# Vector Database Constants
VECTOR_NAMESPACE = "banking-terms"
VECTOR_DIMENSION = 768  # dimension for llmware embedding model
VECTOR_METRIC = "cosine"
VECTOR_SIMILARITY_THRESHOLD = 0.2
LLMWARE_LLM_MODEL = "slim-sql-tool"  # Default LLM model for SQL generation

# LLM Constants
LLM_TEMPERATURE = 0.0  # GGUF models don't use temperature
LLM_MAX_TOKENS = 2048  # Default token limit for responses
LLM_MODEL = LLMWARE_LLM_MODEL  # Use llmware model as default

# Database Schema Constants
DEFAULT_SCHEMA_PATH = "src/config/schema.yaml"

# SQL Patterns
SQL_EXTRACTION_PATTERNS = [
    r"```sql\n(.*?)```",     # Markdown SQL block
    r"```(.*?)```",          # Any code block
    r"SELECT\s+.*?;",        # Basic SQL statement
    r"SELECT\s+.*$",         # SQL without semicolon
    r".*SELECT\s+.*?;.*",    # SQL anywhere in text
    r".*SELECT\s+.*$"        # SQL anywhere without semicolon
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

# LLMWare Constants
LLMWARE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Default embedding model
LLMWARE_LIBRARY = "textsql"  # Default library name

# ChromaDB Constants
CHROMA_PERSIST_DIR = "chroma_db" 