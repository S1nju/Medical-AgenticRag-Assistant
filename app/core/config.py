"""Configuration module for the Medical RAG Assistant."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    dashscope_api_key: str = Field(default="", env="DASHSCOPE_API_KEY")
    
    # ============================================================================
    # DATABASE (Qdrant) - French Drug Information
    # ============================================================================
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(default="french_drugs", env="QDRANT_COLLECTION_NAME")
    
    # ============================================================================
    # EMBEDDING MODELS (French-optimized)
    # ============================================================================
    dense_model_name: str = Field(
        default="BAAI/bge-m3", 
        env="DENSE_MODEL_NAME"
    )
    sparse_model_name: str = Field(
        default="prithivida/Splade_PP_en_v1", 
        env="SPARSE_MODEL_NAME"
    )
    reranker_model_name: str = Field(
        default="BAAI/bge-reranker-base", 
        env="RERANKER_MODEL_NAME"
    )
    reranker_top_k: int = Field(default=3, env="RERANKER_TOP_K")
    retrieval_top_k: int = Field(default=5, env="RETRIEVAL_TOP_K")
    
    # ============================================================================
    # LLM MODELS
    # ============================================================================
    # Main LLM for responses
    llm_model_name: str = Field(default="gpt-4", env="LLM_MODEL_NAME")
    llm_temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")
    
    # Judge LLM for validation
    judge_model_name: str = Field(default="gpt-3.5-turbo", env="JUDGE_MODEL_NAME")
    judge_temperature: float = Field(default=0.0, env="JUDGE_TEMPERATURE")
    
    # Translator LLM
    translator_model_name: str = Field(default="gpt-3.5-turbo", env="TRANSLATOR_MODEL_NAME")
    
    # ============================================================================
    # LANGUAGE SETTINGS
    # ============================================================================
    # Source languages: Algerian Darija, French, Arabic, English
    # Vector DB language: French (fr)
    db_language: str = Field(default="fr", env="DB_LANGUAGE")  # French for vector DB
    
    # ============================================================================
    # NeDRex API
    # ============================================================================
    nedrex_api_url: str = Field(
        default="https://api.nedrex.net", 
        env="NEDREX_API_URL"
    )
    nedrex_timeout: int = Field(default=30, env="NEDREX_TIMEOUT")
    
    # ============================================================================
    # CHAINLIT & API
    # ============================================================================
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    chainlit_host: str = Field(default="0.0.0.0", env="CHAINLIT_HOST")
    chainlit_port: int = Field(default=8001, env="CHAINLIT_PORT")
    
    # ============================================================================
    # LANGSMITH - LLM Observability & Tracing
    # ============================================================================
    langsmith_enabled: bool = Field(default=True, env="LANGSMITH_ENABLED")
    langsmith_api_key: Optional[str] = Field(default=None, env="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="medical-rag-assistant", env="LANGCHAIN_PROJECT")
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com", env="LANGCHAIN_ENDPOINT")
    langsmith_tracing: bool = Field(default=True, env="LANGCHAIN_TRACING_V2")
    
    # ============================================================================
    # APPLICATION
    # ============================================================================
    app_name: str = "Medical RAG Assistant"
    debug_mode: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
# Global settings instance
settings = Settings()
