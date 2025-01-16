# Vector Database Constants
VECTOR_NAMESPACE = "banking-terms"
VECTOR_DIMENSION = 768  # dimension for 'all-mpnet-base-v2' model
VECTOR_METRIC = "cosine"
VECTOR_SIMILARITY_THRESHOLD = 0.2

# LLM Constants
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 500
LLM_MODEL = "gpt-4"

# Database Schema Constants
DEFAULT_SCHEMA_PATH = "src/config/schema.yaml"

# SQL Patterns
SQL_EXTRACTION_PATTERNS = [
    r"```sql\n(.*?)```",  # Markdown SQL block
    r"```(.*?)```",       # Any code block
    r"SELECT\s+.*?;",     # Basic SQL statement
]

# Logging Constants
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_FILE_PREFIX = "text2sql"
LOG_DIR = "logs"

# Environment Variables



# Pinecone Constants
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-west-2"
PINECONE_ENVIRONMENT = "us-east1-gcp" 

# Add llmware constants
LLMWARE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Default embedding model
LLMWARE_LLM_MODEL = "llmware/bling-1.4b-0.1"  # Default LLM model
LLMWARE_LIBRARY = "textsql"  # Default library name 