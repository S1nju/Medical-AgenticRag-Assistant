"""
LangSmith Tracing Configuration

This module sets up LangSmith tracing for the entire application.
It tracks:
- LLM calls and responses
- Tool usage (NeDRex, Qdrant, web search)
- Agent workflow steps
- Performance metrics
- Errors and exceptions
"""

import os
import logging
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


def setup_langsmith_tracing():
    """
    Configure LangSmith tracing for the application.
    
    Sets up environment variables required for LangChain/LangSmith integration.
    This should be called at application startup.
    """
    if not settings.langsmith_enabled:
        logger.info("LangSmith tracing is disabled")
        return
    
    if not settings.langsmith_api_key:
        logger.warning(
            "LangSmith is enabled but LANGSMITH_API_KEY is not set. "
            "Tracing will not work. Get your API key from https://smith.langchain.com"
        )
        return
    
    # Set LangSmith environment variables
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langsmith_tracing).lower()
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    
    logger.info(
        f"✅ LangSmith tracing enabled for project: {settings.langsmith_project}"
    )
    logger.info(f"   Dashboard: https://smith.langchain.com/o/medical-rag-assistant/projects/p/{settings.langsmith_project}")


def get_langsmith_config() -> dict:
    """
    Get current LangSmith configuration.
    
    Returns:
        Dictionary with LangSmith settings
    """
    return {
        "enabled": settings.langsmith_enabled,
        "project": settings.langsmith_project,
        "endpoint": settings.langsmith_endpoint,
        "tracing": settings.langsmith_tracing,
        "api_key_set": bool(settings.langsmith_api_key),
    }


def create_run_name(node_name: str, query: str) -> str:
    """
    Create a descriptive run name for LangSmith traces.
    
    Args:
        node_name: Name of the node/step (e.g., "translate", "router")
        query: User query (truncated for readability)
    
    Returns:
        Formatted run name
    """
    query_preview = query[:50] + "..." if len(query) > 50 else query
    return f"{node_name}: {query_preview}"


# Initialize tracing on import if enabled
if settings.langsmith_enabled:
    setup_langsmith_tracing()
