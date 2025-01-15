from abc import ABC, abstractmethod
import os
import aiohttp
import asyncio
from typing import List, Dict, Optional, Tuple, Set
import numpy as np
from dataclasses import dataclass
import yaml
from pathlib import Path
import re
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import sqlparse
from sqlparse.sql import Where, Comparison
import nltk

from interfaces import VectorManager, VectorData, VectorSearchResult, VectorAPIClient, LLMRequest, LLMResponse,VectorDBError,LLMAPIClient
from api_clients import  PineconeVectorAPIClient
from llm_clients import OpenAILLMClient

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

@dataclass
class DomainContext:
    """Represents context for domain-specific interpretation"""
    industry: str
    current_context: Optional[str] = None
    related_terms: Set[str] = None
    abbreviations: Dict[str, str] = None

@dataclass
class QueryIntent:
    """Captures the semantic intent of a query"""
    action_type: str  # e.g., 'SELECT', 'AGGREGATE', 'COMPARE'
    main_entities: List[str]
    conditions: List[dict]
    temporal_context: Optional[str] = None
    aggregation_type: Optional[str] = None

class TextPreprocessor:
    """Preprocesses natural language queries for better analysis"""
    
    def __init__(self, config: dict):
        self.config = config
        self.standardization_rules = config.get('standardization_rules', {})
    
    def preprocess_query(self, query: str, context: DomainContext) -> str:
        """
        Preprocess the query text with standardization and context-aware fixes
        
        Args:
            query: Original query text
            context: Domain context for context-aware preprocessing
            
        Returns:
            Preprocessed query text
        """
        processed = query.strip()
        
        # Apply standardization rules
        processed = self._apply_standardization(processed)
        
        # Expand abbreviations based on context
        if context.abbreviations:
            processed = self._expand_abbreviations(processed, context.abbreviations)
        
        # Handle industry-specific preprocessing
        if context.industry == "banking":
            processed = self._preprocess_banking_query(processed)
        
        return processed
    
    def _apply_standardization(self, text: str) -> str:
        """Apply standard text cleaning rules"""
        result = text
        
        for pattern, replacement in self.standardization_rules.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        # Common standardizations
        result = re.sub(r'\s+', ' ', result)  # Normalize whitespace
        result = result.replace(' ? ', '?').replace(' !', '!')  # Fix punctuation
        result = result.strip()
        
        return result
    
    def _expand_abbreviations(self, text: str, abbreviations: Dict[str, str]) -> str:
        """Expand abbreviations based on context"""
        result = text
        
        # Sort abbreviations by length (longest first) to handle nested abbreviations
        sorted_abbrevs = sorted(
            abbreviations.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        
        for abbr, full_form in sorted_abbrevs:
            # Use word boundaries to avoid partial matches
            pattern = rf'\b{re.escape(abbr)}\b'
            result = re.sub(pattern, full_form, result, flags=re.IGNORECASE)
        
        return result
    
    def _preprocess_banking_query(self, text: str) -> str:
        """Apply banking-specific preprocessing rules"""
        # Example: Standardize common banking terms
        replacements = {
            r'\b(?:cc|credit card)\b': 'credit_card',
            r'\bssn\b': 'social_security_number',
            r'\bdob\b': 'date_of_birth',
            r'\bytd\b': 'year_to_date',
            r'\bmtd\b': 'month_to_date'
        }
        
        result = text
        for pattern, replacement in replacements.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result

class SemanticAnalyzer:
    """Enhanced semantic analysis with context awareness"""
    
    def __init__(self, config: dict):
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        self.config = config
        self.term_patterns = self._compile_term_patterns()
        self.nlp_patterns = self._compile_nlp_patterns()
    
    def _compile_term_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for domain terms"""
        patterns = {}
        domain_terms = self.config.get('domain_terms', {})
        
        for term, info in domain_terms.items():
            # Create pattern for main term
            patterns[term] = re.compile(rf'\b{term}\b', re.IGNORECASE)
            
            # Add patterns for synonyms
            for synonym in info.get('synonyms', []):
                patterns[synonym] = re.compile(rf'\b{synonym}\b', re.IGNORECASE)
                
            # Add patterns for abbreviations
            for abbr in info.get('abbreviations', []):
                patterns[abbr] = re.compile(rf'\b{abbr}\b', re.IGNORECASE)
                
        return patterns
    
    def _compile_nlp_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile patterns for NLP analysis"""
        return {
            'aggregate': [
                re.compile(r'\b(average|avg|mean)\b', re.IGNORECASE),
                re.compile(r'\b(sum|total)\b', re.IGNORECASE),
                re.compile(r'\b(count|number of)\b', re.IGNORECASE),
                re.compile(r'\b(minimum|min|lowest)\b', re.IGNORECASE),
                re.compile(r'\b(maximum|max|highest)\b', re.IGNORECASE)
            ],
            'temporal': [
                re.compile(r'\b(today|yesterday|tomorrow)\b', re.IGNORECASE),
                re.compile(r'\b(last|past|previous)\s+(\d+|few|couple)\s*(day|week|month|year)s?\b', re.IGNORECASE),
                re.compile(r'\b(this|current)\s+(week|month|year)\b', re.IGNORECASE)
            ]
        }
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract domain-specific entities from query"""
        entities = []
        
        try:
            # Try to use NLTK tokenization
            words = word_tokenize(query.lower())
        except Exception as e:
            # Fallback to simple splitting if NLTK fails
            print(f"Warning: NLTK tokenization failed, using fallback: {str(e)}")
            words = query.lower().split()
        
        # Check for matches in term patterns
        for term, pattern in self.term_patterns.items():
            if pattern.search(query):
                entities.append(term)
        
        # Check database schema for table and column matches
        schema = self.config.get('schema', {}).get('tables', {})
        for table, info in schema.items():
            if re.search(rf'\b{table}\b', query, re.IGNORECASE):
                entities.append(table)
            
            # Check columns
            columns = info.get('columns', {})
            if isinstance(columns, dict):
                for col_name in columns.keys():
                    if re.search(rf'\b{col_name}\b', query, re.IGNORECASE):
                        entities.append(col_name)
            elif isinstance(columns, list):
                for column in columns:
                    if isinstance(column, str):
                        if re.search(rf'\b{column}\b', query, re.IGNORECASE):
                            entities.append(column)
                    elif isinstance(column, dict):
                        col_name = list(column.keys())[0]
                        if re.search(rf'\b{col_name}\b', query, re.IGNORECASE):
                            entities.append(col_name)
        
        return list(set(entities))  # Remove duplicates
    
    def _identify_aggregation(self, query: str) -> Optional[str]:
        """Identify aggregation type if present"""
        for pattern in self.nlp_patterns['aggregate']:
            match = pattern.search(query.lower())
            if match:
                agg_term = match.group(0).lower()
                if 'average' in agg_term or 'avg' in agg_term or 'mean' in agg_term:
                    return 'AVG'
                elif 'sum' in agg_term or 'total' in agg_term:
                    return 'SUM'
                elif 'count' in agg_term or 'number' in agg_term:
                    return 'COUNT'
                elif 'min' in agg_term or 'lowest' in agg_term:
                    return 'MIN'
                elif 'max' in agg_term or 'highest' in agg_term:
                    return 'MAX'
        return None
    
    def _extract_temporal_context(self, query: str) -> Optional[str]:
        """Extract temporal context from query"""
        for pattern in self.nlp_patterns['temporal']:
            match = pattern.search(query)
            if match:
                return match.group(0)
        return None
    
    def analyze_query(self, query: str, context: DomainContext) -> QueryIntent:
        """
        Perform comprehensive semantic analysis of the query
        """
        # Extract basic intent
        intent = self._extract_base_intent(query)
        
        # Enhance with domain context
        intent = self._enhance_with_context(intent, context)
        
        # Extract conditions and comparisons
        intent.conditions = self._extract_conditions(query)
        
        # Identify temporal context if any
        intent.temporal_context = self._extract_temporal_context(query)
        
        return intent
    
    def _extract_base_intent(self, query: str) -> QueryIntent:
        """Extract base query intent"""
        # Identify action type
        action_type = self._identify_action_type(query)
        
        # Extract main entities
        entities = self._extract_entities(query)
        
        # Identify aggregation if present
        agg_type = self._identify_aggregation(query)
        
        return QueryIntent(
            action_type=action_type,
            main_entities=entities,
            conditions=[],
            aggregation_type=agg_type
        )
    
    def _identify_action_type(self, query: str) -> str:
        """Identify the type of query action"""
        if any(word in query.lower() for word in ['average', 'sum', 'count', 'min', 'max']):
            return 'AGGREGATE'
        elif any(word in query.lower() for word in ['compare', 'difference', 'versus']):
            return 'COMPARE'
        else:
            return 'SELECT'
    
    def _extract_conditions(self, query: str) -> List[dict]:
        """Extract conditions and comparisons from query"""
        conditions = []
        
        # Look for comparison patterns
        comparison_patterns = [
            (r'(greater|more|higher|above|over) than (\d+)', '>'),
            (r'(less|lower|below|under) than (\d+)', '<'),
            (r'equal to (\d+)', '='),
            (r'at least (\d+)', '>='),
            (r'at most (\d+)', '<=')
        ]
        
        for pattern, operator in comparison_patterns:
            matches = re.finditer(pattern, query.lower())
            for match in matches:
                conditions.append({
                    'value': float(match.group(2)),
                    'operator': operator,
                    'context': query[max(0, match.start()-20):min(len(query), match.end()+20)]
                })
        
        return conditions
    
    def _enhance_with_context(self, intent: QueryIntent, context: DomainContext) -> QueryIntent:
        """
        Enhance query intent with domain context.
        
        Args:
            intent: Base query intent
            context: Domain-specific context
            
        Returns:
            Enhanced query intent
        """
        # Add domain-specific abbreviation expansions
        if context.abbreviations:
            expanded_entities = []
            for entity in intent.main_entities:
                if entity in context.abbreviations:
                    expanded_entities.append(context.abbreviations[entity])
                expanded_entities.append(entity)
            intent.main_entities = list(set(expanded_entities))
        
        # Add related terms from context
        if context.related_terms:
            related_entities = []
            for entity in intent.main_entities:
                # Add directly related terms
                related = [term for term in context.related_terms 
                         if self._are_terms_related(entity, term)]
                related_entities.extend(related)
            
            # Add to main entities if they're relevant
            intent.main_entities.extend(
                [term for term in related_entities 
                 if self._is_term_relevant(term, context.current_context)]
            )
            intent.main_entities = list(set(intent.main_entities))
        
        # Adjust intent based on industry context
        if context.industry == "banking":
            # Add banking-specific interpretations
            self._enhance_banking_context(intent)
        
        return intent
    
    def _are_terms_related(self, term1: str, term2: str) -> bool:
        """Check if two terms are semantically related using WordNet"""
        try:
            # Get WordNet synsets for both terms
            synsets1 = wordnet.synsets(term1)
            synsets2 = wordnet.synsets(term2)
            
            if not synsets1 or not synsets2:
                return False
            
            # Check path similarity between first synsets
            similarity = synsets1[0].path_similarity(synsets2[0])
            return similarity is not None and similarity > 0.5
            
        except Exception as e:
            print(f"Warning: WordNet similarity check failed: {str(e)}")
            return False
    
    def _is_term_relevant(self, term: str, current_context: Optional[str]) -> bool:
        """Check if a term is relevant in the current context"""
        if not current_context:
            return True
            
        # Get domain-specific relevance rules
        relevance_rules = self.config.get('domain_terms', {}).get(term, {}).get('context_rules', [])
        
        for rule in relevance_rules:
            if rule.get('when') == current_context:
                return rule.get('weight', 'normal') != 'ignore'
        
        return True
    
    def _enhance_banking_context(self, intent: QueryIntent):
        """Add banking-specific enhancements to intent"""
        # Example: Add common banking-related joins
        if 'credit_score' in intent.main_entities:
            if 'payment_history' not in intent.main_entities:
                intent.main_entities.append('payment_history')
        
        # Example: Add standard banking conditions
        if intent.action_type == 'AGGREGATE' and 'risk_rating' in intent.main_entities:
            intent.conditions.append({
                'type': 'filter',
                'field': 'risk_rating',
                'operator': 'IN',
                'values': ['LOW', 'MEDIUM', 'HIGH']
            })

class SQLValidator:
    """Validates generated SQL against schema and business rules"""
    
    def __init__(self, config: dict):
        self.schema = config['schema']
        self.business_rules = config['business_rules']
    
    def validate_sql(self, sql: str) -> Tuple[bool, List[str]]:
        """
        Validate SQL against schema and business rules
        Returns (is_valid, list_of_issues)
        """
        issues = []
        
        # Parse SQL
        parsed = sqlparse.parse(sql)[0]
        
        # Validate table names
        issues.extend(self._validate_tables(parsed))
        
        # Validate column names
        issues.extend(self._validate_columns(parsed))
        
        # Validate joins
        issues.extend(self._validate_joins(parsed))
        
        # Validate business rules
        issues.extend(self._validate_business_rules(parsed))
        
        return len(issues) == 0, issues
    
    def _validate_tables(self, parsed) -> List[str]:
        """Validate table names and usage"""
        issues = []
        tables = self._extract_tables(parsed)
        
        for table in tables:
            if table not in self.schema['tables']:
                issues.append(f"Invalid table name: {table}")
        
        return issues
    
    def _validate_business_rules(self, parsed) -> List[str]:
        """Validate compliance with business rules"""
        issues = []
        
        # Extract WHERE clause
        where_clause = next((token for token in parsed.tokens if isinstance(token, Where)), None)
        
        if where_clause:
            # Check each business rule
            for rule in self.business_rules:
                if not self._check_rule_compliance(where_clause, rule):
                    issues.append(f"Business rule violation: {rule['name']}")
        
        return issues
    
    def _extract_tables(self, parsed) -> List[str]:
        """Extract table names from parsed SQL"""
        tables = []
        for token in parsed.tokens:
            if token.ttype is None and hasattr(token, 'get_name'):
                tables.append(token.get_name())
        return tables
    
    def _validate_columns(self, parsed) -> List[str]:
        """Validate column names against schema"""
        issues = []
        columns = self._extract_columns(parsed)
        
        for column in columns:
            if not self._is_valid_column(column):
                issues.append(f"Invalid column name: {column}")
        
        return issues
    
    def _validate_joins(self, parsed) -> List[str]:
        """Validate join conditions"""
        issues = []
        joins = self._extract_joins(parsed)
        
        for join in joins:
            if not self._is_valid_join(join):
                issues.append(f"Invalid join condition: {join}")
        
        return issues
    
    def _check_rule_compliance(self, where_clause, rule: dict) -> bool:
        """Check if WHERE clause complies with a business rule"""
        rule_condition = rule['condition']
        where_text = str(where_clause)
        
        # Basic check - ensure required conditions are present
        if rule_condition.lower() not in where_text.lower():
            return False
        
        return True

class QueryTranslator:
    """Enhanced query translator with full pipeline implementation"""
    
    def __init__(
        self,
        config_path: Path,
        vector_api_client: VectorAPIClient,
        llm_api_client: LLMAPIClient
    ):
        self.config = self._load_config(config_path)
        self.preprocessor = TextPreprocessor(self.config)
        self.semantic_analyzer = SemanticAnalyzer(self.config)
        self.vector_manager = VectorManager(vector_api_client)
        self.llm_client = llm_api_client
        self.sql_validator = SQLValidator(self.config)
        
        # Initialize domain context
        self.domain_context = DomainContext(
            industry="banking",
            abbreviations=self.config.get('abbreviations', {}),
            related_terms=set(self.config.get('related_terms', []))
        )
    
    def _load_config(self, config_path: Path) -> dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration directory
            
        Returns:
            dict: Loaded configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        try:
            config_file = config_path
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_file}")
                
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            # Add default configurations if not present
            config.setdefault('abbreviations', {})
            config.setdefault('related_terms', [])
            config.setdefault('standardization_rules', {})
            
            return config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {str(e)}")
    
    async def translate_to_sql(self, natural_query: str) -> Tuple[str, dict]:
        """
        Translate natural language query to SQL with full pipeline
        Returns (sql_query, analysis_info)
        """
        # Preprocess query
        processed_query = self.preprocessor.preprocess_query(
            natural_query, 
            self.domain_context
        )
        
        # Perform semantic analysis
        query_intent = self.semantic_analyzer.analyze_query(
            processed_query,
            self.domain_context
        )
        
        # Find similar terms using vector search
        query_embedding = self.semantic_analyzer.embedding_model.encode(processed_query)
        similar_terms = await self.vector_manager.find_similar_terms(query_embedding)
        
        # Prepare LLM prompt
        prompt = self._prepare_llm_prompt(
            processed_query,
            query_intent,
            similar_terms
        )
        
        # Generate SQL using LLM
        llm_request = LLMRequest(
            prompt=prompt,
            temperature=0.3,
            additional_context={
                "query_intent": query_intent.__dict__,
                "similar_terms": [t.metadata for t in similar_terms],
                "schema": self.config['schema']
            }
        )
        
        llm_response = await self.llm_client.generate_completion(llm_request)
        
        # Validate generated SQL
        is_valid, issues = self.sql_validator.validate_sql(llm_response.text)
        
        if not is_valid:
            # Handle validation issues
            # You might want to retry with modified prompt or raise an error
            raise ValueError(f"Generated SQL failed validation: {issues}")
        
        return llm_response.text, {
            "intent": query_intent,
            "similar_terms": similar_terms,
            "validation_issues": issues
        }
    
    def _prepare_llm_prompt(
        self,
        query: str,
        intent: QueryIntent,
        similar_terms: List[VectorSearchResult]
    ) -> str:
        """
        Prepare the prompt for the LLM with context and examples.
        
        Args:
            query: Original natural language query
            intent: Analyzed query intent
            similar_terms: Similar terms from vector search
            
        Returns:
            Formatted prompt string
        """
        schema_context = self._build_schema_context(similar_terms)
        
        prompt = f"""
        You are a SQL expert. Convert the following natural language query to SQL.
        
        Database Schema:
        {schema_context}
        
        Query Intent:
        - Action Type: {intent.action_type}
        - Main Entities: {', '.join(intent.main_entities)}
        - Conditions: {intent.conditions}
        - Temporal Context: {intent.temporal_context or 'None'}
        - Aggregation Type: {intent.aggregation_type or 'None'}
        
        Similar Terms Found:
        {self._format_similar_terms(similar_terms)}
        
        Natural Language Query:
        {query}
        
        Generate a valid SQL query that:
        1. Uses the correct table and column names from the schema
        2. Implements the identified aggregations and conditions
        3. Follows standard SQL best practices
        4. Includes appropriate JOIN conditions if multiple tables are needed
        
        SQL Query:
        """
        
        return prompt.strip()
    
    def _build_schema_context(self, similar_terms: List[VectorSearchResult]) -> str:
        """
        Build a string representation of the relevant schema context.
        
        Args:
            similar_terms: Similar terms from vector search to help identify relevant tables
            
        Returns:
            Formatted schema context string
        """
        schema = self.config['schema']
        context_parts = []
        
        # Add table definitions
        for table_name, table_info in schema['tables'].items():
            columns = table_info.get('columns', {})
            relationships = table_info.get('relationships', [])
            
            # Format column definitions
            if isinstance(columns, dict):
                column_defs = [
                    f"  - {col_name} ({info.get('type', 'unknown')}):"
                    f" {info.get('description', '')}"
                    for col_name, info in columns.items()
                ]
            else:  # List format
                column_defs = [f"  - {col}" for col in columns]
            
            # Format relationships
            relation_defs = []
            for rel in relationships:
                rel_str = f"  - Related to {rel['table']}"
                if 'join_conditions' in rel:
                    rel_str += f" ON {' AND '.join(rel['join_conditions'])}"
                relation_defs.append(rel_str)
            
            # Combine into table definition
            table_def = [
                f"Table: {table_name}",
                "Columns:",
                *column_defs
            ]
            
            if relation_defs:
                table_def.extend([
                    "Relationships:",
                    *relation_defs
                ])
            
            context_parts.append("\n".join(table_def))
        
        return "\n\n".join(context_parts)
    
    def _format_similar_terms(self, similar_terms: List[VectorSearchResult]) -> str:
        """Format similar terms for prompt inclusion"""
        if not similar_terms:
            return "No similar terms found."
            
        formatted_terms = []
        for term in similar_terms[:5]:  # Limit to top 5 terms
            metadata = term.metadata
            term_str = f"- {metadata.get('term', 'Unknown Term')}"
            if 'description' in metadata:
                term_str += f": {metadata['description']}"
            if 'table' in metadata:
                term_str += f" (Table: {metadata['table']})"
            formatted_terms.append(term_str)
            
        return "\n".join(formatted_terms)

# Example configuration
config = {
    'schema': {
        'tables': {
            'customer_credit': {
                'columns': ['customer_id', 'credit_score', 'risk_rating'],
                'relationships': [
                    {'table': 'payment_history', 'type': 'one_to_many'}
                ]
            }
        }
    },
    'business_rules': [
        {
            'name': 'credit_score_range',
            'condition': 'credit_score BETWEEN 300 AND 850'
        }
    ]
}

# Example usage
async def main():
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    # Get the base project path and API key from environment variables
    BASE_PROJECT_PATH = os.getenv("BASE_PROJECT_PATH")
    OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    vector_api_client = PineconeVectorAPIClient(
        api_key=PINECONE_API_KEY,
        environment="us-east1-gcp",
        index_name="text2sql"
    )
    
    # Alternative: Initialize Weaviate client
    # vector_api_client = WeaviateVectorAPIClient(
    #     api_key="your-weaviate-api-key",
    #     url="https://your-weaviate-instance.com"
    # )
    
    # Initialize LLM client (OpenAI example)
    llm_api_client = OpenAILLMClient(
        api_key=OPEN_AI_API_KEY,
        model="gpt-4"
    )
    


    translator = QueryTranslator(
        config_path=Path(BASE_PROJECT_PATH) / "src/config/schema.yaml",
        vector_api_client=vector_api_client,
        llm_api_client=llm_api_client
    )
    
    query = "What's the average credit score for customers with late payments?"
    sql, analysis = await translator.translate_to_sql(query)
    print(f"Generated SQL: {sql}")
    print(f"Analysis: {analysis}")

if __name__ == "__main__":
    asyncio.run(main())