import os
from typing import Optional

class Config:
    """Configuration management for the Contextual Personal Assistant."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    
    # Database Configuration
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "database/assistant.db")
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "database/chroma_db")
    
    # Application Configuration
    APP_NAME: str = "Serani - Contextual Personal Assistant"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Agent Configuration
    MAX_CONVERSATION_HISTORY: int = int(os.getenv("MAX_CONVERSATION_HISTORY", "10"))
    SUMMARIZATION_THRESHOLD: int = int(os.getenv("SUMMARIZATION_THRESHOLD", "20"))
    CONTEXT_RETRIEVAL_LIMIT: int = int(os.getenv("CONTEXT_RETRIEVAL_LIMIT", "3"))
    
    # Streamlit Configuration
    STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    STREAMLIT_HOST: str = os.getenv("STREAMLIT_HOST", "localhost")
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configuration is present."""
        required_vars = []
        
        # Note: OPENAI_API_KEY is not required at startup as users can input it in UI
        if not cls.DATABASE_PATH:
            required_vars.append("DATABASE_PATH")
        
        if required_vars:
            print(f"Missing required configuration: {', '.join(required_vars)}")
            return False
        
        return True
    
    @classmethod
    def get_database_config(cls) -> dict:
        """Get database configuration."""
        return {
            "database_path": cls.DATABASE_PATH,
            "chroma_db_path": cls.CHROMA_DB_PATH
        }
    
    @classmethod
    def get_openai_config(cls) -> dict:
        """Get OpenAI configuration."""
        return {
            "api_key": cls.OPENAI_API_KEY,
            "model": cls.OPENAI_MODEL,
            "temperature": cls.OPENAI_TEMPERATURE
        }
    
    @classmethod
    def get_agent_config(cls) -> dict:
        """Get agent configuration."""
        return {
            "max_conversation_history": cls.MAX_CONVERSATION_HISTORY,
            "summarization_threshold": cls.SUMMARIZATION_THRESHOLD,
            "context_retrieval_limit": cls.CONTEXT_RETRIEVAL_LIMIT
        }