"""
Simplified Nodes for Medical RAG Agent.

Nodes (decision points):
1. Translator - Translates input to French for vector DB
2. Router - Decides which tool(s) to use
3. Synthesizer - Structures the response
4. Judge - Validates response against retrieved info
"""

from typing import Dict, Any, Literal
import logging
from ..llm import ChatModelSinglton
from app.core.config import settings
from app.tools.agent_tools import retrieve_from_french_db, search_nedrex_disease

logger = logging.getLogger(__name__)

class TranslatorNode:
    """Translates user query to French, keeping drug/scientific terms."""
    
    def __init__(self):
        self.model = ChatModelSinglton().get_model_instance(
            model_name=settings.translator_model_name,
            temperature=0.0
        )
        logger.info("TranslatorNode initialized")
    
    def translate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate user query to French.
        
        Handles:
        - Algerian Darija → French
        - Arabic → French
        - English → French
        - Already French → Keep as is
        """
        original_query = state.get("question", "")
        
        if not original_query:
            logger.warning("No question in state")
            return state
        
        system_prompt = """You are a medical translation expert. 
        
Translate the user's query to French while:
1. Keeping drug names, scientific terms, and medical terminology UNCHANGED
2. Translating only natural language words
3. If already in French, return as-is

Example:
- Input: "ما هي أعراض الإنفلونزا و علاج بالأسبرين"
- Output: "Quels sont les symptômes de la grippe et le traitement par l'aspirine"

Return ONLY the French translation, nothing else."""
        
        try:
            # Format prompt as string for ChatQwen
            prompt = f"{system_prompt}\n\nUser: {original_query}\n\nAssistant:"
            
            translation_result = self.model.invoke(prompt)
            translation_text = (
                translation_result.content
                if hasattr(translation_result, "content")
                else str(translation_result)
            )
            french_query = translation_text.strip()
            
            logger.info(f"Translated: {original_query[:50]}... → {french_query[:50]}...")
            
            state["original_query"] = original_query
            state["french_query"] = french_query
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            # Fallback: use original query
            state["french_query"] = original_query
        
        return state

