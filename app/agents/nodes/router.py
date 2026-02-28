"""
Router Node - Routes queries to appropriate retrieval method.

Determines whether to use:
- Vector Database (French drug information)
- NeDRex API (disease and symptom information)  
- Hybrid (both sources)
"""

import logging
from typing import Dict, Any, Literal
from app.tools.agent_tools import retrieve_from_french_db, search_nedrex_disease
from ..llm import ChatModelSinglton
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class RouterNode:
    """Routes query to appropriate tools: DB, NeDRex, or both."""
    
    def __init__(self):
        self.model = ChatModelSinglton().get_model_instance(
            model_name=settings.judge_model_name,
            temperature=0.0
        )
        logger.info("RouterNode initialized")
    
    def route(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide whether to use:
        - 'db': Vector database (drug info, dosages, side effects)
        - 'nedrex': NeDRex API (diseases, symptoms, genes)
        - 'both': Both sources
        """
        query = state.get("french_query", "")
        
        system_prompt = """Analyze this medical query and decide which tool to use:

db = French drug database (for: dosages, side effects, drug interactions, medications)
nedrex = Disease/symptom database (for: diseases, symptoms, genes, condition info)
both = Use both sources

Respond with ONLY one word: 'db', 'nedrex', or 'both' or 'chitchat' if it's a general question not requiring retrieval.

Examples:
- "dosage de l'aspirine" → db
- "symptômes de la grippe" → nedrex
- "bonjour, comment ça va?" → chitchat
- "side effects of ibuprofen for my condition" → both"""
        
        try:
            # Format prompt as string for ChatQwen
            prompt = f"{system_prompt}\n\nQuery: {query}"
            
            decision_result = self.model.invoke(prompt)
            decision_text = (
                decision_result.content
                if hasattr(decision_result, "content")
                else str(decision_result)
            )
            decision = decision_text.strip().lower()
            
          
        except Exception as e:
            logger.error(f"Tools error: {e}")
            # Default to both tools
            decision = "chitchat" 
        state["tool_choice"] = decision
        return state

