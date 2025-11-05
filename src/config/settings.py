"""
Configuration module for RAG Chatbot
Handles all environment variables and configuration settings
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Central configuration class"""
    
    # Base Paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_RESOURCES_DIR = BASE_DIR / "data_resources"
    CHROMA_DB_DIR = BASE_DIR / "chroma_db"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_API_KEY: Optional[str] = os.getenv("OLLAMA_API_KEY")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral:latest")
    
    # ChromaDB Configuration
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", str(CHROMA_DB_DIR))
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")
    
    # IndoBERT Configuration
    INDOBERT_MODEL: str = os.getenv("INDOBERT_MODEL", "indobenchmark/indobert-base-p1")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "768"))
    
    # Data Processing Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    MAX_CHUNKS_PER_DOCUMENT: int = int(os.getenv("MAX_CHUNKS_PER_DOCUMENT", "100"))
    
    # RAG Configuration
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # Streamlit Configuration
    APP_TITLE: str = os.getenv("APP_TITLE", "RAG Chatbot Indonesia")
    APP_ICON: str = os.getenv("APP_ICON", "")
    PAGE_CONFIG: str = os.getenv("PAGE_CONFIG", "wide")
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", str(LOGS_DIR / "app.log"))
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.DATA_RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate(cls) -> bool:
        """Validate critical configuration"""
        if not cls.OLLAMA_BASE_URL:
            raise ValueError("OLLAMA_BASE_URL must be set")
        
        return True

# Initialize directories on import
Config.ensure_directories()
