"""
Synthesizer Node - Generates structured medical responses.

This node:
1. Takes retrieved information (from DB or NeDRex or web search)
2. Structures it into a coherent, medical-grade response
3. Includes sources, dosages, warnings, and contraindications
4. Formats for optimal readability
"""

from typing import Dict, Any, List
import logging
from ..llm import ChatModelSinglton
from app.core.config import settings
from app.tools.nedrexapi import get_nedrex_api

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatNode:
    """Synthesizes a structured response from retrieved information."""
    
    def __init__(self):
        self.model = ChatModelSinglton().get_model_instance(
            model_name=settings.llm_model_name,
            temperature=settings.llm_temperature
        )
    
    def chat(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a structured medical response in the user's original language.
        """
        original_query = state.get("original_query", "")
        
        # Format prompt as string for ChatQwen
        system_message = "You are a helpful medical assistant."
        prompt = f"{system_message}\n\nUser: {original_query}\n\nAssistant:"
        
        response = self.model.invoke(prompt)
        response_text = (
            response.content
            if hasattr(response, "content")
            else str(response)
        )
        state["response"] = response_text
        return state
