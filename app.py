import os
import asyncio
from threading import Thread
import logging
import re
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Create a patch for the imports in the src directory
# This will modify the sys.modules to fix relative imports
def patch_imports():
    # Create __init__.py in src directory if it doesn't exist
    src_init_path = os.path.join(project_root, 'src', '__init__.py')
    if not os.path.exists(src_init_path):
        with open(src_init_path, 'w') as f:
            f.write('# This file marks the directory as a Python package\n')
    
    # Add src to path
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

# Apply the patch
patch_imports()

# Now import the modules
from src.config import config
from src.constants import LLMWARE_LLM_MODEL
from src.interfaces import LLMWareAPIClient
from src.api_clients import ChromaVectorAPIClient
from src.main import setup_logging, QueryTranslator, initialize_vector_db, QueryIntent

# Load environment variables
load_dotenv()

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to store initialized components
translator = None
initialization_complete = False
initialization_error = None
config_data = None

def run_async_task(coro):
    """Run an async task in the current event loop or create a new one"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

class EnhancedSemanticAnalyzer:
    """Enhanced version of the semantic analyzer that properly populates the intent object"""
    
    def __init__(self, original_analyzer, config_data):
        self.original_analyzer = original_analyzer
        self.config_data = config_data
        self.domain_terms = config_data.get('domain_terms', [])
        self.schema = config_data.get('schema', {})
        self.tables = self.schema.get('tables', {})
        
        # Copy all attributes from the original analyzer
        for attr_name in dir(original_analyzer):
            # Skip private attributes and methods
            if not attr_name.startswith('_') and not callable(getattr(original_analyzer, attr_name)):
                setattr(self, attr_name, getattr(original_analyzer, attr_name))
        
        # Explicitly copy the embedding_model
        self.embedding_model = original_analyzer.embedding_model
        # Copy the term_patterns
        self.term_patterns = original_analyzer.term_patterns
    
    def analyze_query(self, query, context):
        # Get the original intent
        original_intent = self.original_analyzer.analyze_query(query, context)
        
        # Extract main entities (tables and columns mentioned in the query)
        main_entities = self._extract_main_entities(query)
        
        # Extract conditions with proper operators
        conditions = self._extract_conditions(query)
        
        # Create a new intent with the enhanced data
        enhanced_intent = QueryIntent(
            action_type=original_intent.action_type,
            main_entities=main_entities,
            conditions=conditions,
            temporal_context=original_intent.temporal_context,
            aggregation_type=original_intent.aggregation_type
        )
        
        return enhanced_intent
    
    def _extract_main_entities(self, query):
        """Extract main entities (tables and columns) from the query"""
        main_entities = []
        query_lower = query.lower()
        
        # Check for table names in the query
        for table_name, table_info in self.tables.items():
            if table_name.lower() in query_lower:
                main_entities.append(table_name)
            
            # Check for column names in the query
            for column_name, column_info in table_info.get('columns', {}).items():
                if column_name.lower() in query_lower:
                    main_entities.append(f"{table_name}.{column_name}")
                
                # Check for column synonyms
                synonyms = column_info.get('synonyms', [])
                if synonyms:
                    for synonym in synonyms:
                        if synonym.lower() in query_lower:
                            main_entities.append(f"{table_name}.{column_name}")
        
        # Check for domain terms in the query
        for term_info in self.domain_terms:
            term = term_info.get('term', '')
            if term and term.lower() in query_lower:
                table = term_info.get('table', '')
                column = term_info.get('column', '')
                if table and column:
                    main_entities.append(f"{table}.{column}")
            
            # Check for term synonyms
            synonyms = term_info.get('synonyms', [])
            if synonyms:
                for synonym in synonyms:
                    if synonym.lower() in query_lower:
                        table = term_info.get('table', '')
                        column = term_info.get('column', '')
                        if table and column:
                            main_entities.append(f"{table}.{column}")
        
        return list(set(main_entities))  # Remove duplicates
    
    def _extract_conditions(self, query):
        """Extract conditions with proper operators from the query"""
        conditions = []
        query_lower = query.lower()
        
        # Define patterns for different operators
        operator_patterns = [
            (r'greater than or equal to\s+\$?(\d+)', '>='),
            (r'less than or equal to\s+\$?(\d+)', '<='),
            (r'greater than\s+\$?(\d+)', '>'),
            (r'less than\s+\$?(\d+)', '<'),
            (r'equal to\s+\$?(\d+)', '='),
            (r'not equal to\s+\$?(\d+)', '!='),
            (r'at least\s+(\d+)', '>='),
            (r'at most\s+(\d+)', '<='),
            (r'exactly\s+(\d+)', '='),
            (r'more than\s+\$?(\d+)', '>'),
            (r'over\s+\$?(\d+)', '>'),
            (r'under\s+\$?(\d+)', '<'),
            (r'above\s+\$?(\d+)', '>'),
            (r'below\s+\$?(\d+)', '<'),
        ]
        
        # Check for domain terms and apply operator patterns
        for term_info in self.domain_terms:
            term = term_info.get('term', '')
            table = term_info.get('table', '')
            column = term_info.get('column', '')
            
            if not (term and table and column):
                continue
                
            # Check if the term or its synonyms are in the query
            term_in_query = term.lower() in query_lower
            synonyms = term_info.get('synonyms', [])
            if synonyms:
                for synonym in synonyms:
                    if synonym.lower() in query_lower:
                        term_in_query = True
                        break
            
            if not term_in_query:
                continue
            
            # Check for predefined value in the term
            if 'value' in term_info and term_info['value'].lower() in query_lower:
                conditions.append({
                    'field': column,
                    'table': table,
                    'operator': '=',
                    'value': term_info['value']
                })
                continue
            
            # Check for operator patterns
            for pattern, operator in operator_patterns:
                matches = re.findall(pattern, query_lower)
                if matches:
                    for match in matches:
                        # Check if this pattern is related to the current term
                        # by looking for the term near the pattern
                        term_pos = query_lower.find(term.lower())
                        pattern_pos = query_lower.find(pattern.split(r'\s+')[0].lower())
                        
                        # If term is found and is reasonably close to the pattern
                        if term_pos >= 0 and pattern_pos >= 0 and abs(term_pos - pattern_pos) < 50:
                            conditions.append({
                                'field': column,
                                'table': table,
                                'operator': operator,
                                'value': match
                            })
        
        # Look for specific conditions in the query
        if 'late' in query_lower and 'payment' in query_lower:
            if 'don\'t have' in query_lower or 'do not have' in query_lower:
                conditions.append({
                    'field': 'payment_status',
                    'table': 'payment_history',
                    'operator': '!=',
                    'value': 'Late'
                })
            else:
                conditions.append({
                    'field': 'payment_status',
                    'table': 'payment_history',
                    'operator': '=',
                    'value': 'Late'
                })
        
        if 'high risk' in query_lower:
            if 'except' in query_lower or 'not' in query_lower:
                conditions.append({
                    'field': 'risk_rating',
                    'table': 'customer_credit',
                    'operator': '!=',
                    'value': 'HIGH'
                })
            else:
                conditions.append({
                    'field': 'risk_rating',
                    'table': 'customer_credit',
                    'operator': '=',
                    'value': 'HIGH'
                })
        
        return conditions

def initialize_in_background():
    """Initialize the QueryTranslator in a background thread"""
    global translator, initialization_complete, initialization_error, config_data
    
    try:
        logger.info("Initializing API clients...")
        
        # Get configuration
        BASE_PROJECT_PATH = os.getenv("BASE_PROJECT_PATH", os.getcwd())
        
        # Load config file
        config_path = Path(BASE_PROJECT_PATH) / "src/config/schema.yaml"
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        
        # Initialize ChromaDB vector store
        vector_api_client = ChromaVectorAPIClient(
            collection_name="banking_terms"
        )
        
        # Initialize llmware LLM client
        llm_api_client = LLMWareAPIClient(
            model_name=LLMWARE_LLM_MODEL
        )
        
        # Initialize vector database with domain terms
        run_async_task(initialize_vector_db(vector_api_client, config_data))
        
        logger.info("Initializing QueryTranslator...")
        translator = QueryTranslator(
            config_path=config_path,
            vector_api_client=vector_api_client,
            llm_api_client=llm_api_client
        )
        
        # Enhance the semantic analyzer
        enhanced_analyzer = EnhancedSemanticAnalyzer(
            translator.semantic_analyzer,
            config_data
        )
        translator.semantic_analyzer = enhanced_analyzer
        
        initialization_complete = True
        logger.info("Initialization complete")
        
    except Exception as e:
        initialization_error = str(e)
        logger.error(f"Initialization error: {str(e)}", exc_info=True)

@app.route('/translate', methods=['POST'])
def translate():
    """Endpoint to translate natural language to SQL"""
    global translator, initialization_complete, initialization_error
    
    # Check if initialization is complete
    if not initialization_complete:
        if initialization_error:
            return jsonify({
                'error': f'Initialization failed: {initialization_error}'
            }), 500
        else:
            return jsonify({
                'error': 'Server is still initializing. Please try again in a moment.'
            }), 503
    
    try:
        # Get question from request
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({
                'error': 'Missing required field: question'
            }), 400
        
        question = data['question']
        logger.info(f"Received translation request: {question}")
        
        # Translate question to SQL using async function
        sql, analysis = run_async_task(translator.translate_to_sql(question))
        
        # Return response
        return jsonify({
            'sql': sql,
            'analysis': analysis
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global initialization_complete, initialization_error
    
    if initialization_complete:
        return jsonify({
            'status': 'ready'
        })
    elif initialization_error:
        return jsonify({
            'status': 'error',
            'error': initialization_error
        }), 500
    else:
        return jsonify({
            'status': 'initializing'
        }), 503

if __name__ == '__main__':
    # Start initialization in a background thread
    logger.info("Starting initialization in background...")
    init_thread = Thread(target=initialize_in_background)
    init_thread.daemon = True
    init_thread.start()
    
    # Run the Flask app
    port = int(os.getenv('PORT', 5001))
    logger.info(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False) 