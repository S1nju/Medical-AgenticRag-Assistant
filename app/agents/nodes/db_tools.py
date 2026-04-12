"""
DB Tools Node - Retrieves information from the French database.

Handles all database-related retrieval operations.
"""

import logging
from typing import Dict, Any
from app.tools.agent_tools import retrieve_from_french_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DBToolsNode:
    """Retrieves information from the French medical database."""
    
    def __init__(self):
        logger.info("DBToolsNode initialized")
    
    def query_db(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute database queries.
        
        Args:
            state: Current agent state with query
            
        Returns:
            Updated state with db_results
        """
        try:
            query = state.get("french_query", "")
            
            logger.info("DB Tools node executing")
            
            # Call database
            db_results = retrieve_from_french_db(query)
            state["db_results"] = db_results
            logger.info(f"Retrieved {len(db_results)} documents from French DB")
            
        except Exception as e:
            logger.error(f"DB Tools node error: {e}")
            # Provide fallback defaults
            state["db_results"] = []
        
        return state
