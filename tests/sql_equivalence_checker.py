import sqlglot
import pandas as pd
from sqlalchemy import create_engine, text
from typing import Tuple, List, Dict, Optional
import re

class SQLEquivalenceChecker:
    def __init__(self, db_connection=None):
        """
        Initialize the SQL equivalence checker.
        Args:
            db_connection: SQLAlchemy database connection (optional for static analysis)
        """
        self.db_connection = db_connection
        
    def normalize_query(self, query: str) -> str:
        """
        Normalize a SQL query by removing extra whitespace and standardizing syntax.
        """
        # Remove comments
        query = re.sub(r'--.*$', '', query, flags=re.MULTILINE)
        query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)
        
        # Normalize whitespace
        query = ' '.join(query.split())
        
        # Convert to lowercase for case-insensitive comparison
        query = query.lower()
        
        # Standardize quotes
        query = query.replace('"', "'")
        
        return query

    def parse_query(self, query: str) -> Tuple[Optional[sqlglot.expressions.Expression], List[str]]:
        """
        Parse a SQL query and return its AST and any errors encountered.
        """
        errors = []
        try:
            ast = sqlglot.parse_one(query)
            return ast, errors
        except Exception as e:
            errors.append(f"Parsing error: {str(e)}")
            return None, errors

    def compare_static(self, query1: str, query2: str) -> Tuple[bool, List[str]]:
        """
        Perform static analysis comparison of two SQL queries.
        """
        differences = []
        
        # Normalize queries
        norm_query1 = self.normalize_query(query1)
        norm_query2 = self.normalize_query(query2)
        
        # Parse queries
        ast1, errors1 = self.parse_query(norm_query1)
        ast2, errors2 = self.parse_query(norm_query2)
        
        # If there were parsing errors, return them
        if errors1 or errors2:
            return False, errors1 + errors2
        
        # Compare basic structure
        if ast1 == ast2:
            return True, ["Queries are structurally identical after normalization"]
        
        # Analyze differences in structure
        try:
            # Compare table references
            tables1 = set(str(t) for t in ast1.find_all(sqlglot.exp.Table))
            tables2 = set(str(t) for t in ast2.find_all(sqlglot.exp.Table))
            if tables1 != tables2:
                differences.append(f"Different tables referenced: {tables1} vs {tables2}")
            
            # Compare columns
            cols1 = set(str(c) for c in ast1.find_all(sqlglot.exp.Column))
            cols2 = set(str(c) for c in ast2.find_all(sqlglot.exp.Column))
            if cols1 != cols2:
                differences.append(f"Different columns referenced: {cols1} vs {cols2}")
            
            # Compare conditions
            where1 = [str(w) for w in ast1.find_all(sqlglot.exp.Where)]
            where2 = [str(w) for w in ast2.find_all(sqlglot.exp.Where)]
            if where1 != where2:
                differences.append("Different WHERE conditions")
            
        except Exception as e:
            differences.append(f"Error during structural comparison: {str(e)}")
        
        return False, differences

    def compare_results(self, query1: str, query2: str) -> Tuple[bool, List[str]]:
        """
        Compare actual query results using a database connection.
        """
        if not self.db_connection:
            return False, ["No database connection provided for result comparison"]
        
        differences = []
        try:
            # Execute queries
            df1 = pd.read_sql(text(query1), self.db_connection)
            df2 = pd.read_sql(text(query2), self.db_connection)
            
            # Compare basic properties
            if df1.shape != df2.shape:
                differences.append(f"Different result shapes: {df1.shape} vs {df2.shape}")
                return False, differences
            
            # Sort both dataframes
            if not df1.empty and not df2.empty:
                df1 = df1.sort_values(by=list(df1.columns)).reset_index(drop=True)
                df2 = df2.sort_values(by=list(df2.columns)).reset_index(drop=True)
            
            # Compare column names
            if list(df1.columns) != list(df2.columns):
                differences.append(f"Different columns: {list(df1.columns)} vs {list(df2.columns)}")
            
            # Compare actual data
            if not df1.equals(df2):
                # Find specific differences
                for col in df1.columns:
                    if col in df2.columns and not df1[col].equals(df2[col]):
                        differences.append(f"Data differs in column: {col}")
                        
                # Sample of differing rows
                if not df1.equals(df2):
                    diff_mask = ~(df1 == df2).all(axis=1)
                    if diff_mask.any():
                        sample_diff = pd.concat([df1[diff_mask], df2[diff_mask]]).head(3)
                        differences.append(f"Sample of differing rows:\n{sample_diff}")
            
            return len(differences) == 0, differences
            
        except Exception as e:
            return False, [f"Error during result comparison: {str(e)}"]

    def check_equivalence(self, query1: str, query2: str) -> Dict:
        """
        Perform comprehensive equivalence check combining static and dynamic analysis.
        """
        result = {
            "equivalent": False,
            "static_analysis": {},
            "result_comparison": {},
            "overall_differences": []
        }
        
        # Perform static analysis
        static_equivalent, static_differences = self.compare_static(query1, query2)
        result["static_analysis"] = {
            "equivalent": static_equivalent,
            "differences": static_differences
        }
        
        # Perform result comparison if database connection is available
        if self.db_connection:
            result_equivalent, result_differences = self.compare_results(query1, query2)
            result["result_comparison"] = {
                "equivalent": result_equivalent,
                "differences": result_differences
            }
            
            # Queries are equivalent only if both static and result analysis agree
            result["equivalent"] = static_equivalent and result_equivalent
            result["overall_differences"] = static_differences + result_differences
        else:
            # If no database connection, rely only on static analysis
            result["equivalent"] = static_equivalent
            result["overall_differences"] = static_differences
        
        return result

# Example usage:
if __name__ == "__main__":
    # Example with SQLite database
    engine = create_engine('sqlite:///:memory:')
    
    # Create a sample table
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT,
                age INTEGER
            )
        """))
        
        # Insert sample data
        conn.execute(text("""
            INSERT INTO users (name, age) VALUES 
            ('Alice', 25),
            ('Bob', 30),
            ('Charlie', 35)
        """))
        conn.commit()
    
    # Initialize checker
    checker = SQLEquivalenceChecker(engine)
    # Example queries to compare
    query1 = "SELECT * FROM users WHERE age > 25 ORDER BY name"
    query2 = "SELECT id, name, age FROM users WHERE age > 25 ORDER BY name"
    
    # Check equivalence
    result = checker.check_equivalence(query1, query2)
    
    # Print results
    print("\nEquivalence Check Results:")
    print(f"Equivalent: {result['equivalent']}")
    if result['overall_differences']:
        print("\nDifferences found:")
        for diff in result['overall_differences']:
            print(f"- {diff}")