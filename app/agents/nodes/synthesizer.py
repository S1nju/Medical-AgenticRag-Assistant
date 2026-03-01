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


class SynthesizerNode:
    """Synthesizes a structured response from retrieved information."""
    
    def __init__(self):
        self.model = ChatModelSinglton().get_model_instance(
            model_name=settings.llm_model_name,
            temperature=settings.llm_temperature
        )
        self.nedrex_api = get_nedrex_api()
        logger.info("SynthesizerNode initialized")
    
    def _format_nedrex_results(self, nedrex_results: Dict[str, Any]) -> str:
        """
        Format NeDRex results from multiple collections into readable text.
        
        Args:
            nedrex_results: Dictionary with collections and their results
            
        Returns:
            Formatted string for context
        """
        if not nedrex_results.get("success", False):
            return "NeDRex: No results available.\n"
        
        formatted = "From NeDRex Medical Knowledge Base:\n\n"
        collections = nedrex_results.get("collections", {})
        
        for collection_name, result in collections.items():
            if not result.get("success", False):
                continue
                
            items = result.get("results", [])
            if not items:
                continue
            
            # Format collection name for display
            display_name = collection_name.replace("_", " ").title()
            formatted += f"** {display_name} **:\n"
            
            for i, item in enumerate(items[:3], 1):  # Top 3 items per collection
                name = item.get("name", "Unknown")
                score = item.get("score", 0.0)
                description = item.get("description")
                entity_type = item.get("type", "")
                synonyms = item.get("synonyms", [])
                
                # Format entry
                type_str = f" [{entity_type}]" if entity_type else ""
                formatted += f"  {i}. {name}{type_str} (relevance: {score:.3f})\n"
                
                # Add synonyms if available
                if synonyms and isinstance(synonyms, list) and len(synonyms) > 0:
                    syn_list = ", ".join(synonyms[:3])
                    formatted += f"     Also known as: {syn_list}\n"
                
                # Add description if available
                if description and description != "No description available":
                    desc_text = description[:150] + "..." if len(description) > 150 else description
                    formatted += f"     {desc_text}\n"
            
            formatted += "\n"
        
        return formatted if collections else "NeDRex: No relevant results found.\n"
    
    def synthesize(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a structured medical response in the user's original language.
        """
        original_query = state.get("original_query", "")
        french_query = state.get("french_query", "")
        db_results = state.get("db_results", [])
        nedrex_results = state.get("nedrex_results", {})
        collections_used = state.get("nedrex_collections_used", [])
        
        # Build context from retrieved information
        context = "Retrieved Information:\n\n"
        has_context = False
        
        if not db_results or  db_results[0].get("content", "").strip() == "":
            has_context = False
        else:
            has_context = True
            context += "From French Drug Database:\n"
            for doc in db_results[:3]:
                  # Top 3 docs
                metadata = doc.get("metadata", {})
                content = doc.get('content', '')
                if isinstance(content, str) and content.strip() != "":
                    context += f"- {content[:200]}\n"
                if isinstance(metadata, dict) :
                    meta_str = ", ".join(f"{k}: {v}" for k, v in metadata.items())
                    context += f"- {meta_str}\n"
        
        if nedrex_results and nedrex_results.get("success", False):
            has_context = True
            context += "\n" + self._format_nedrex_results(nedrex_results)
            if collections_used:
                context += f"\n(Searched collections: {', '.join(collections_used)})\n"
        
        # Check if we have any context
        if not has_context:
            state["response"] = "Sorry, I couldn't find relevant information to answer your question."
            return state
        
        system_prompt = f"""You are a medical information assistant.

Answer the user's question using the provided information.
Structure your response with:
1. Direct answer
2. Key points
3. Important notes/warnings

{context}

if there is no context or it qritten in point dont answer the question and say that you don't have enough information to answer the question. say i didnt find the information in the retrieved documents.
Respond in the same language as the user's original question.
Be clear, accurate, and healthcare-focused."""
        
        try:
            # Format prompt as string for ChatQwen
            prompt = f"{system_prompt}\n\nUser Question: {original_query}\n\nAssistant:"
            
            response = self.model.invoke(prompt)
            
            response_text = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )
            
            logger.info("Response synthesized successfully")
            state["response"] = response_text
            
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            state["response"] = f"Error generating response: {e}"
        
        return state
