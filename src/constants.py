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
ENV_BASE_PROJECT_PATH = "BASE_PROJECT_PATH"
ENV_OPENAI_API_KEY = "OPEN_AI_API_KEY"
ENV_PINECONE_API_KEY = "PINECONE_API_KEY"

# Pinecone Constants
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-west-2"
PINECONE_ENVIRONMENT = "us-east1-gcp" 