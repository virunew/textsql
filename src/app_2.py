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
import logging
from datetime import datetime

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

def setup_logging(log_level=logging.INFO):
    """Configure logging with a custom format"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create a log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"text2sql_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    return logging.getLogger(__name__)

# Create logger instance
logger = logging.getLogger(__name__)

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
        logger.info("Initializing SQLValidator")
        self.schema = config['schema']
        self.business_rules = config['business_rules']
    
    def validate_sql(self, sql: str) -> Tuple[bool, List[str]]:
        logger.info(f"Validating SQL: {sql}")
        issues = []
        
        try:
            # Parse SQL
            parsed = sqlparse.parse(sql)
            if not parsed:
                logger.error("Failed to parse SQL query")
                return False, ["Failed to parse SQL query"]
            
            parsed = parsed[0]
            logger.debug("SQL parsed successfully")
            
            # Validate components
            logger.debug("Validating SQL components...")
            
            if not str(parsed).upper().strip().startswith('SELECT'):
                logger.warning("Query doesn't start with SELECT")
                issues.append("Query must start with SELECT")
                return False, issues
            
            # Validate tables
            logger.debug("Validating tables...")
            table_issues = self._validate_tables(parsed)
            if table_issues:
                logger.warning(f"Table validation issues: {table_issues}")
                issues.extend(table_issues)
                return False, issues
            
            # Validate other components
            logger.debug("Validating columns...")
            issues.extend(self._validate_columns(parsed))
            
            logger.debug("Validating joins...")
            issues.extend(self._validate_joins(parsed))
            
            logger.debug("Validating business rules...")
            issues.extend(self._validate_business_rules(parsed))
            
            is_valid = len(issues) == 0
            logger.info(f"Validation {'successful' if is_valid else 'failed'}: {issues}")
            return is_valid, issues
            
        except Exception as e:
            logger.error("SQL validation error", exc_info=True)
            return False, [f"SQL validation error: {str(e)}"]
    
    def _validate_tables(self, parsed) -> List[str]:
        """Validate table names and usage"""
        issues = []
        try:
            # Extract tables
            tables = self._extract_tables(parsed)
            
            # Check if any tables were found
            if not tables:
                issues.append("No valid tables found in query")
                return issues
            
            # Validate each table
            for table in tables:
                if not isinstance(table, str):
                    issues.append(f"Invalid table name type: {type(table)}")
                    continue
                    
                if table not in self.schema['tables']:
                    issues.append(f"Invalid table name: {table}")
                    continue
                    
                # Additional table-specific validations can be added here
            
            # Check for required joins if multiple tables
            if len(tables) > 1:
                join_issues = self._validate_required_joins(tables, parsed)
                issues.extend(join_issues)
            
            return issues
            
        except Exception as e:
            issues.append(f"Error validating tables: {str(e)}")
            return issues
    
    def _validate_required_joins(self, tables: List[str], parsed) -> List[str]:
        """Validate that required joins are present for multiple tables"""
        issues = []
        
        # Extract actual joins from query
        joins = self._extract_joins(parsed)
        join_tables = {join['table'] for join in joins if join.get('table')}
        
        # Check if all tables except the first one are joined
        main_table = tables[0]
        tables_to_join = set(tables[1:])
        
        # Find missing joins
        missing_joins = tables_to_join - join_tables
        if missing_joins:
            for table in missing_joins:
                issues.append(f"Missing JOIN condition for table: {table}")
        
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
        
        def extract_from_token(token):
            # Skip None tokens
            if token is None:
                return
                
            # Handle identifiers and table names
            if token.ttype is None and hasattr(token, 'get_name'):
                name = token.get_name()
                # Check if name exists and is a valid table
                if name and isinstance(name, str) and name in self.schema['tables']:
                    tables.append(name)
            
            # Handle FROM and JOIN clauses specifically
            if str(token).upper().strip() in ('FROM', 'JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN'):
                next_token = token.next_token
                if next_token and hasattr(next_token, 'get_name'):
                    name = next_token.get_name()
                    if name and isinstance(name, str) and name in self.schema['tables']:
                        tables.append(name)
            
            # Recursively process tokens
            if hasattr(token, 'tokens'):
                for sub_token in token.tokens:
                    extract_from_token(sub_token)
        
        try:
            # Process all tokens recursively
            for token in parsed.tokens:
                extract_from_token(token)
        except Exception as e:
            print(f"Warning: Error extracting tables: {e}")
        
        return list(set(tables))  # Remove duplicates
    
    def _validate_columns(self, parsed) -> List[str]:
        """Validate column names against schema"""
        issues = []
        columns = self._extract_columns(parsed)
        
        for column in columns:
            if not self._is_valid_column(column):
                issues.append(f"Invalid column name: {column}")
        
        return issues
    
    def _extract_columns(self, parsed) -> List[str]:
        """Extract column names from parsed SQL"""
        columns = []
        
        def extract_from_token(token):
            if token.ttype is None and hasattr(token, 'get_name'):
                columns.append(token.get_name())
            elif hasattr(token, 'tokens'):
                for sub_token in token.tokens:
                    extract_from_token(sub_token)
        
        # Process all tokens recursively
        for token in parsed.tokens:
            extract_from_token(token)
        
        return columns
    
    def _validate_joins(self, parsed) -> List[str]:
        """Validate join conditions"""
        issues = []
        joins = self._extract_joins(parsed)
        
        for join in joins:
            if not self._is_valid_join(join):
                issues.append(f"Invalid join condition: {join}")
        
        return issues
    
    def _extract_joins(self, parsed) -> List[dict]:
        """Extract join conditions from parsed SQL"""
        joins = []
        
        def extract_from_token(token):
            if str(token).upper().strip().startswith('JOIN'):
                join_info = {
                    'type': 'JOIN',
                    'table': None,
                    'condition': None
                }
                
                # Extract table name and condition
                for idx, sub_token in enumerate(token.tokens):
                    if sub_token.ttype is None and hasattr(sub_token, 'get_name'):
                        join_info['table'] = sub_token.get_name()
                    elif str(sub_token).upper().strip() == 'ON':
                        # Get the condition after ON
                        if idx + 1 < len(token.tokens):
                            join_info['condition'] = str(token.tokens[idx + 1])
                
                joins.append(join_info)
            elif hasattr(token, 'tokens'):
                for sub_token in token.tokens:
                    extract_from_token(sub_token)
        
        # Process all tokens recursively
        for token in parsed.tokens:
            extract_from_token(token)
        
        return joins
    
    def _is_valid_column(self, column: str) -> bool:
        """Check if column exists in schema"""
        for table_info in self.schema['tables'].values():
            columns = table_info.get('columns', {})
            if isinstance(columns, dict):
                if column in columns:
                    return True
            elif isinstance(columns, list):
                if column in columns:
                    return True
        return False
    
    def _is_valid_join(self, join: dict) -> bool:
        """Validate join condition against schema relationships"""
        if not join['table'] or not join['condition']:
            return False
            
        # Check if table exists
        if join['table'] not in self.schema['tables']:
            return False
            
        # Check if join condition matches defined relationships
        table_info = self.schema['tables'][join['table']]
        relationships = table_info.get('relationships', [])
        
        for rel in relationships:
            if any(cond.lower() in join['condition'].lower() 
                  for cond in rel.get('join_conditions', [])):
                return True
                
        return False
    
    def _check_rule_compliance(self, where_clause, rule: dict) -> bool:
        """Check if WHERE clause complies with a business rule"""
        rule_condition = rule['condition']
        where_text = str(where_clause)
        
        # Basic check - ensure required conditions are present
        if rule.get('required', False) and rule_condition.lower() not in where_text.lower():
            return False
            
        # Check for incompatible conditions
        if rule.get('incompatible_with'):
            incompatible = rule['incompatible_with']
            if (rule_condition.lower() in where_text.lower() and
                any(inc.lower() in where_text.lower() for inc in incompatible)):
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
        logger.info("Initializing QueryTranslator")
        try:
            self.config = self._load_config(config_path)
            logger.info(f"Loaded configuration from {config_path}")
            
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
            logger.info("QueryTranslator initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize QueryTranslator: {str(e)}", exc_info=True)
            raise
    
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
        logger.info(f"Starting translation for query: {natural_query}")
        try:
            # Preprocess query
            logger.debug("Preprocessing query...")
            processed_query = self.preprocessor.preprocess_query(
                natural_query, 
                self.domain_context
            )
            logger.info(f"Preprocessed query: {processed_query}")
            
            # Perform semantic analysis
            logger.debug("Performing semantic analysis...")
            query_intent = self.semantic_analyzer.analyze_query(
                processed_query,
                self.domain_context
            )
            logger.info(f"Query intent: {query_intent}")
            
            # Find similar terms
            logger.debug("Finding similar terms...")
            query_embedding = self.semantic_analyzer.embedding_model.encode(processed_query)
            similar_terms = await self.vector_manager.find_similar_terms(query_embedding)
            logger.info(f"Found {len(similar_terms) if similar_terms else 0} similar terms")
            
            if similar_terms is None:
                logger.warning("No similar terms found, using empty list")
                similar_terms = []
            
            # Generate SQL
            logger.debug("Preparing LLM prompt...")
            prompt = self._prepare_llm_prompt(processed_query, query_intent, similar_terms)
            
            logger.debug("Generating SQL using LLM...")
            llm_request = LLMRequest(
                prompt=prompt,
                temperature=0.3,
                max_tokens=500,
                additional_context={
                    "query_intent": query_intent.__dict__,
                    "similar_terms": [t.metadata for t in similar_terms] if similar_terms else [],
                    "schema": self.config['schema']
                }
            )
            
            llm_response = await self.llm_client.generate_completion(llm_request)
            logger.debug("Received LLM response")
            
            # Extract and validate SQL
            logger.debug("Extracting SQL from response...")
            try:
                sql = self._extract_sql_from_response(llm_response.text)
                logger.info(f"Generated SQL: {sql}")
            except Exception as e:
                logger.error("Failed to extract SQL from LLM response", exc_info=True)
                raise QueryTranslationError(f"Failed to extract valid SQL from LLM response: {str(e)}")
            
            logger.debug("Validating SQL...")
            issues = []
            # try:
            #     is_valid, issues = self.sql_validator.validate_sql(sql)
            #     if issues:
            #         logger.warning(f"Validation issues found: {issues}")
                
            #     if not is_valid:
            #         logger.error(f"SQL validation failed: {issues}")
            #         raise QueryTranslationError(
            #             f"Generated SQL failed validation: {'; '.join(issues)}"
            #         )
            # except Exception as e:
            #     logger.error("SQL validation error", exc_info=True)
            #     raise QueryTranslationError(f"SQL validation error: {str(e)}")
            
            logger.info("Translation completed successfully")
            return sql, {
                "intent": query_intent.__dict__,
                "similar_terms": [t.metadata for t in similar_terms] if similar_terms else [],
                "validation_issues": issues
            }
            
        except QueryTranslationError as e:
            logger.error(f"Translation error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in translation pipeline: {str(e)}", exc_info=True)
            raise QueryTranslationError(f"Error in query translation pipeline: {str(e)}")
    
    def _extract_sql_from_response(self, response_text: str) -> str:
        """
        Extract SQL query from LLM response text.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            str: Extracted SQL query
            
        Raises:
            QueryTranslationError: If no valid SQL found
        """
        # Look for SQL between common delimiters
        sql_patterns = [
            r"```sql\n(.*?)```",  # Markdown SQL block
            r"```(.*?)```",       # Any code block
            r"SELECT\s+.*?;",     # Basic SQL statement (more precise)
        ]
        
        for pattern in sql_patterns:
            matches = re.finditer(pattern, response_text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                sql = match.group(1) if len(match.groups()) > 0 else match.group(0)
                sql = sql.strip()
                if sql.upper().startswith("SELECT"):
                    # Basic validation of SQL structure
                    if "FROM" in sql.upper() and sql.strip().endswith(";"):
                        return sql
        
        # If we get here, try to extract any SELECT statement
        if "SELECT" in response_text.upper() and "FROM" in response_text.upper():
            # Extract everything between SELECT and the next period or end of string
            match = re.search(r"SELECT.*?(?:;|$)", response_text, re.DOTALL | re.IGNORECASE)
            if match:
                sql = match.group(0).strip()
                if not sql.endswith(";"):
                    sql += ";"
                return sql
        
        raise QueryTranslationError("No valid SQL query found in LLM response")
    
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

class QueryTranslationError(Exception):
    """Custom exception for query translation errors"""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error

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
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Text-to-SQL application")
    
    try:
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        logger.info("Loaded environment variables")
        
        # Get configuration
        BASE_PROJECT_PATH = os.getenv("BASE_PROJECT_PATH")
        OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        
        logger.info("Initializing API clients...")
        vector_api_client = PineconeVectorAPIClient(
            api_key=PINECONE_API_KEY,
            environment="us-east1-gcp",
            index_name="text2sql"
        )
        
        llm_api_client = OpenAILLMClient(
            api_key=OPEN_AI_API_KEY,
            model="gpt-4"
        )
        
        logger.info("Initializing QueryTranslator...")
        translator = QueryTranslator(
            config_path=Path(BASE_PROJECT_PATH) / "src/config/schema.yaml",
            vector_api_client=vector_api_client,
            llm_api_client=llm_api_client
        )
        
        query = "What's the average credit score for customers with late payments?"
        logger.info(f"Processing query: {query}")
        
        sql, analysis = await translator.translate_to_sql(query)
        logger.info(f"Generated SQL: {sql}")
        logger.debug(f"Analysis: {analysis}")
        
        print(f"Generated SQL: {sql}")
        print(f"Analysis: {analysis}")
        
    except Exception as e:
        logger.error("Application error", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())