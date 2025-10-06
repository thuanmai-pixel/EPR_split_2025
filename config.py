import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Application configuration"""
    
    # Environment variables
    WEAVIATE_URL = os.environ.get("WEAVIATE_URL")
    WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    
    # Flask settings
    PORT = int(os.environ.get('PORT', 5000))
    DEBUG = False
    
    # Weaviate settings
    WEAVIATE_CLASS_NAME = "LlamaIndex_auto_EPR"
    
    # Retriever settings
    SIMILARITY_TOP_K = 6
    EMPTY_QUERY_TOP_K = 10
    MAX_SOURCES = 3
    MAX_SOURCES_FALLBACK = 5
    
    # LLM settings
    LLM_MODEL = "gpt-3.5-turbo"
    LLM_TEMPERATURE = 0.1
    FAQ_MATCHING_MODEL = "gpt-4o-mini"
    
    # File paths
    FAQ_JSON_PATH = 'faq.json'
    PDF_CATALOG_JSON_PATH = 'pdf_catalog.json'
    
    # CORS settings
    CORS_ORIGINS = "*"
    CORS_METHODS = ["GET", "POST", "OPTIONS"]
    CORS_HEADERS = ["Content-Type"]
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        required = [cls.WEAVIATE_URL, cls.WEAVIATE_API_KEY, cls.OPENAI_API_KEY]
        if not all(required):
            raise ValueError("Missing required environment variables: WEAVIATE_URL, WEAVIATE_API_KEY, OPENAI_API_KEY")
        return True