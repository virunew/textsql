import os
import asyncio
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import yaml
from dotenv import load_dotenv
from threading import Thread

# Import from existing code
from src.main import setup_logging, QueryTranslator, initialize_vector_db
from src.api_clients import ChromaVectorAPIClient
from src.interfaces import LLMWareAPIClient
from src.constants import LLMWARE_LLM_MODEL

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

def run_async_task(coro):
    """Run an async task in the current event loop or create a new one"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

def initialize_in_background():
    """Initialize the QueryTranslator in a background thread"""
    global translator, initialization_complete, initialization_error
    
    try:
        logger.info("Initializing API clients...")
        
        # Get configuration
        BASE_PROJECT_PATH = os.getenv("BASE_PROJECT_PATH", os.getcwd())
        
        # Load config file
        config_path = Path(BASE_PROJECT_PATH) / "src/config/schema.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Initialize ChromaDB vector store
        vector_api_client = ChromaVectorAPIClient(
            collection_name="banking_terms"
        )
        
        # Initialize llmware LLM client
        llm_api_client = LLMWareAPIClient(
            model_name=LLMWARE_LLM_MODEL
        )
        
        # Initialize vector database with domain terms
        run_async_task(initialize_vector_db(vector_api_client, config))
        
        logger.info("Initializing QueryTranslator...")
        translator = QueryTranslator(
            config_path=config_path,
            vector_api_client=vector_api_client,
            llm_api_client=llm_api_client
        )
        
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
    port = int(os.getenv('PORT', 5000))
    logger.info(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False) 