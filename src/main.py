from abc import ABC, abstractmethod
import os
import aiohttp
import asyncio
from typing import List, Dict, Optional, Tuple, Set, Any
import numpy as np
from dataclasses import dataclass
import torch
import yaml
from pathlib import Path
import re
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import sqlparse
from sqlparse.sql import Where, Comparison
import nltk
import logging
from datetime import datetime
from config import config

from interfaces import VectorManager, VectorData, VectorSearchResult, VectorAPIClient, LLMRequest, LLMResponse,VectorDBError,LLMAPIClient, LLMWareAPIClient, LLMWareEmbeddingClient
from api_clients import  PineconeVectorAPIClient, ChromaVectorAPIClient
from llm_clients import OpenAILLMClient
from constants import (
    LLM_TEMPERATURE, LLM_MAX_TOKENS, LLMWARE_EMBEDDING_MODEL, 
    LOG_FORMAT, LOG_DATE_FORMAT, LOG_FILE_PREFIX, LOG_DIR,
    SQL_EXTRACTION_PATTERNS,
    VECTOR_DIMENSION,
    LLMWARE_LLM_MODEL
)

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Configure logging before importing models
logging.basicConfig(level=config.get_logging_level_by_module('models'))

# Now import the model
from llmware.models import GGUFGenerativeModel, HFEmbeddingModel, ModelCatalog

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
    # Create logs directory if it doesn't exist
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(exist_ok=True)
    
    # Create a log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{LOG_FILE_PREFIX}_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
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
    """Analyzes semantic meaning of queries"""
    
    def __init__(self, config: dict):
        self.config = config
        hf_tokenizer = AutoTokenizer.from_pretrained(LLMWARE_EMBEDDING_MODEL)
        hf_model = AutoModel.from_pretrained(LLMWARE_EMBEDDING_MODEL)
        self.embedding_model = HFEmbeddingModel(
            model=hf_model, 
            tokenizer=hf_tokenizer, 
            model_name=LLMWARE_EMBEDDING_MODEL
        )
        self.term_patterns = self._compile_term_patterns()
        
    def _compile_term_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for domain terms"""
        patterns = {}
        domain_terms = self.config.get('domain_terms', [])
        
        for term_info in domain_terms:  # Changed from dict to list iteration
            term = term_info['term']
            synonyms = term_info.get('synonyms', [])
            
            # Create pattern that matches the term or any of its synonyms
            term_variants = [term] + synonyms
            # Escape special characters and join with OR
            pattern_str = '|'.join(map(re.escape, term_variants))
            patterns[term] = re.compile(pattern_str, re.IGNORECASE)
            
        return patterns
        
    def analyze_query(self, query: str, context: DomainContext) -> QueryIntent:
        """
        Analyze query to determine intent and entities
        
        Args:
            query: Preprocessed query text
            context: Domain context for interpretation
            
        Returns:
            QueryIntent object with analysis results
        """
        embedding = self.embedding_model.embedding(query)
        if len(embedding.shape) == 2:
            embedding = embedding[0]  # Take the first vector if it's a batch
        embedding = embedding.flatten().tolist()  # Convert to list of floats
        
        tokens = word_tokenize(query.lower())
        action_type = self._determine_action_type(tokens)
        aggregation_type = self._determine_aggregation(tokens)
        main_entities = []
        for term, pattern in self.term_patterns.items():
            if pattern.search(query):
                main_entities.append(term)
        
        conditions = self._extract_conditions(query, context)
        temporal_context = self._extract_temporal_context(tokens)
        
        return QueryIntent(
            action_type=action_type,
            main_entities=main_entities,
            conditions=conditions,
            temporal_context=temporal_context,
            aggregation_type=aggregation_type
        )
    
    def _determine_action_type(self, tokens: List[str]) -> str:
        """Determine the type of action requested"""
        if any(word in tokens for word in ['average', 'avg', 'mean']):
            return 'AGGREGATE'
        if any(word in tokens for word in ['compare', 'difference', 'versus']):
            return 'COMPARE'
        return 'SELECT'
    
    def _determine_aggregation(self, tokens: List[str]) -> Optional[str]:
        """Determine aggregation type if present"""
        agg_keywords = {
            'average': 'AVG',
            'avg': 'AVG',
            'sum': 'SUM',
            'total': 'SUM',
            'count': 'COUNT',
            'number': 'COUNT',
            'maximum': 'MAX',
            'max': 'MAX',
            'minimum': 'MIN',
            'min': 'MIN'
        }
        
        for token in tokens:
            if token.lower() in agg_keywords:
                return agg_keywords[token.lower()]
        return None
    
    def _extract_conditions(self, query: str, context: DomainContext) -> List[dict]:
        """Extract conditions from query"""
        conditions = []
        domain_terms = self.config.get('domain_terms', [])
        
        # Look for conditions based on domain terms
        for term_info in domain_terms:  # Changed from dict to list iteration
            term = term_info['term']
            if term_info.get('value') and term_info['value'].lower() in query.lower():
                conditions.append({
                    'field': term_info['column'],
                    'table': term_info['table'],
                    'operator': '=',
                    'value': term_info['value']
                })
        
        return conditions
    
    def _extract_temporal_context(self, tokens: List[str]) -> Optional[str]:
        """Extract temporal context if present"""
        temporal_indicators = {
            'today': 'CURRENT_DATE',
            'tomorrow': 'CURRENT_DATE + 1',
            'next week': 'CURRENT_DATE + 7',
            'next month': 'CURRENT_DATE + 30',
            'next quarter': 'CURRENT_DATE + 90',
            'next year': 'CURRENT_DATE + 365',
            'today': 'CURRENT_DATE',
            'yesterday': 'CURRENT_DATE - 1',
            'this month': 'CURRENT_MONTH',
            'last month': 'PREVIOUS_MONTH',
            'this year': 'CURRENT_YEAR',
            'last year': 'PREVIOUS_YEAR',
            'next week': 'CURRENT_DATE + 7',
            'next month': 'CURRENT_DATE + 30',
            'next quarter': 'CURRENT_DATE + 90',
            'next year': 'CURRENT_DATE + 365',
            'yesterday': 'CURRENT_DATE - 1',
            'this month': 'CURRENT_MONTH',
            'last month': 'PREVIOUS_MONTH',
            'this year': 'CURRENT_YEAR',
            'last year': 'PREVIOUS_YEAR'
        }
        
        query_text = ' '.join(tokens)
        for indicator, sql_value in temporal_indicators.items():
            if indicator in query_text:
                return sql_value
        return None

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
            query_embedding = self.semantic_analyzer.embedding_model.embedding(processed_query)
            if len(query_embedding.shape) == 2:
                query_embedding = query_embedding[0]  # Take the first vector if it's a batch
            query_embedding = query_embedding.flatten().tolist()  # Convert to list of floats
            similar_terms = await self.vector_manager.find_similar_terms(query_embedding)
            
            if similar_terms:
                term_scores = [
                    {
                        'term': t.metadata.get('term', 'unknown'),
                        'score': t.score,
                        'table': t.metadata.get('table'),
                        'column': t.metadata.get('column')
                    }
                    for t in similar_terms
                ]
                logger.info(f"Found similar terms: {term_scores}")
            else:
                logger.warning("No similar terms found above threshold")
            
            # Generate SQL
            logger.debug("Preparing LLM prompt...")
            prompt = self._prepare_llm_prompt(processed_query, query_intent, similar_terms)
            logger.debug(f"LLM prompt: {prompt}")
            logger.debug("Generating SQL using LLM...")
            llm_request = LLMRequest(
                prompt=prompt,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                additional_context={
                    "query_intent": query_intent.__dict__,
                    "similar_terms": [t.metadata for t in similar_terms] if similar_terms else [],
                    "schema": self.config['schema']
                }
            )
            
            if isinstance(self.llm_client, LLMWareAPIClient) and isinstance(self.llm_client.model, GGUFGenerativeModel):
                # For GGUF models, use inference directly
                response_text = self.llm_client.model.inference(
                    prompt=llm_request.prompt,
                    add_context=llm_request.additional_context.get("context"),
                    inference_dict={
                        "temperature": llm_request.temperature,
                    }
                )

                # Handle dictionary response from GGUF model
                if isinstance(response_text, dict):
                    response_text = response_text.get("llm_response", "")

                llm_response = LLMResponse(
                    text=response_text,
                    metadata={
                        "model": self.llm_client.model.model_name,
                        "raw_response": response_text,
                        "finish_reason": "completed",
                        "created": datetime.now().isoformat()
                    },
                    usage={
                        "prompt_tokens": self.llm_client.model.usage.get("input", 0),
                        "completion_tokens": self.llm_client.model.usage.get("output", 0),
                        "total_tokens": self.llm_client.model.usage.get("total", 0)
                    }
                )
            else:
                # For other LLM clients, use generate_completion
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
        """Extract SQL query from LLM response"""
        if not response_text:
            raise QueryTranslationError("Empty response from LLM")
        
        # Handle if response_text is a dictionary
        if isinstance(response_text, dict):
            response_text = response_text.get("llm_response", "")
        
        # Log the raw response for debugging
        logger.debug(f"Extracting SQL from response:\n{response_text}")
        
        # Try different patterns to extract SQL
        for pattern in SQL_EXTRACTION_PATTERNS:
            matches = re.finditer(pattern, response_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            for match in matches:
                sql = match.group(1) if len(match.groups()) > 0 else match.group(0)
                sql = sql.strip()
                if sql:
                    # Validate the extracted SQL
                    try:
                        parsed = sqlparse.parse(sql)
                        if parsed and len(parsed) > 0:
                            logger.debug(f"Successfully extracted SQL: {sql}")
                            return sql
                    except Exception as e:
                        logger.warning(f"Failed to parse extracted SQL: {e}")
                        continue
        
        # If no SQL found in patterns, check if the entire response is a valid SQL query
        try:
            parsed = sqlparse.parse(response_text)
            if parsed and len(parsed) > 0 and parsed[0].get_type() == 'SELECT':
                logger.debug("Using full response as SQL query")
                return response_text.strip()
        except Exception as e:
            logger.warning(f"Failed to parse full response as SQL: {e}")
        
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
        5. Use the least number of joins and conditions to achieve the desired result
        
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
        """Format similar terms for the prompt"""
        formatted_terms = []
        for term in similar_terms:
            metadata = term.metadata
            synonyms = metadata.get('synonyms', '')
            if isinstance(synonyms, str):
                synonyms = synonyms.split(',')
            
            term_info = (
                f"- {metadata.get('term', '')}: {metadata.get('description', '')} "
                f"(Table: {metadata.get('table', '')}) "
                f"(Column: {metadata.get('column', '')}) "
                f"(Also known as: {', '.join(s.strip() for s in synonyms if s.strip())})"
            )
            formatted_terms.append(term_info)
        
        return "\n".join(formatted_terms)

class QueryTranslationError(Exception):
    """Custom exception for query translation errors"""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error

# Example usage
async def initialize_vector_db(vector_api_client: VectorAPIClient, config: dict):
    """Initialize vector database with domain terms from config"""
    logger.info("Initializing vector database with domain terms...")
    
    try:
        # Check if vectors already exist by doing a simple search
        try:
            test_query = np.zeros(VECTOR_DIMENSION)
            logger.debug(f"Testing vector database with zero vector of dimension {VECTOR_DIMENSION}")
            existing_vectors = await vector_api_client.search_vectors(
                query_vector=test_query,
                top_k=1  # Just check for any existing vectors
            )
            
            if existing_vectors:
                logger.info("Vector database already initialized")
                return
                
        except Exception as e:
            logger.error(f"Error checking vector database: {str(e)}", exc_info=True)
            raise VectorDBError(f"Failed to check vector database: {str(e)}")
            
        # Continue with initialization if no vectors exist
        domain_terms = config.get('domain_terms', [])
        if not domain_terms:
            logger.warning("No domain terms found in config")
            return
        
        # Create vector data for each term
        vector_data = []
        hf_tokenizer = AutoTokenizer.from_pretrained(LLMWARE_EMBEDDING_MODEL)
        hf_model = AutoModel.from_pretrained(LLMWARE_EMBEDDING_MODEL)
        model = HFEmbeddingModel(model=hf_model, tokenizer=hf_tokenizer, model_name=LLMWARE_EMBEDDING_MODEL)
        
        for term in domain_terms:
            description = term['description']
            if 'synonyms' in term:
                description += f" (Also known as: {', '.join(term['synonyms'])})"
            
            text_to_embed = f"{term['term']} - {description}"
            
            # Get embedding and properly format it
            embedding = model.embedding(text_to_embed)
            
            # Convert the embedding to the correct format
            # If embedding is a 2D array with shape (1, dimension)
            if len(embedding.shape) == 2:
                embedding = embedding[0]  # Take the first (and only) vector
            
            # Convert to list of floats
            embedding_list = embedding.flatten().tolist()
            
            # Create metadata
            metadata = {
                'term': term['term'],
                'description': term['description'],
                'synonyms': term.get('synonyms', [])
            }
            
            if term.get('table'):
                metadata['table'] = term['table']
            if term.get('column'):
                metadata['column'] = term['column']
            if term.get('value'):
                metadata['value'] = term['value']
            
            # Create vector data with properly formatted embedding
            vector_data.append(VectorData(
                id=f"term_{term['term'].replace(' ', '_')}",
                vector=embedding_list,  # Use the properly formatted embedding
                metadata=metadata
            ))
            
            logger.debug(f"Created vector data for term: {term['term']} with embedding dimension {len(embedding_list)}")
        
        # Store vectors in database
        logger.info(f"Attempting to store {len(vector_data)} vectors in database")
        success = await vector_api_client.store_vectors(vector_data)
        
        if success:
            logger.info(f"Successfully stored {len(vector_data)} terms in vector database")
        else:
            logger.error("Failed to store terms in vector database")
            
    except Exception as e:
        logger.error(f"Error initializing vector database: {str(e)}", exc_info=True)
        raise

async def main():
    # Setup logging
    logger = setup_logging(logging.DEBUG)
    logger.info("Starting Text-to-SQL application")
    
    try:
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        logger.info("Loaded environment variables")
        
        # Get configuration
        BASE_PROJECT_PATH = os.getenv("BASE_PROJECT_PATH")
        OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")


        PINECONE_API_KEY = os.getenv("¸")
        
        # Load config file
        config_path = Path(BASE_PROJECT_PATH) / "src/config/schema.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        logger.info("Initializing API clients...")
        
        # Initialize ChromaDB vector store
        vector_api_client = ChromaVectorAPIClient(
            collection_name="banking_terms"
        )
        
        # Initialize llmware LLM client
        llm_api_client = LLMWareAPIClient(
            model_name=LLMWARE_LLM_MODEL
        )
        
        # Initialize vector database with domain terms
        await initialize_vector_db(vector_api_client, config)
        
        logger.info("Initializing QueryTranslator...")
        translator = QueryTranslator(
            config_path=config_path,
            vector_api_client=vector_api_client,
            llm_api_client=llm_api_client
        )
        
        query = "What's the average credit worthiness for customers with late payments?"
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