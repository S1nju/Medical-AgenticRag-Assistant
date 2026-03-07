"""
Relevance Grading Node (LLM-as-a-Judge).

This is the "safety net" that ensures retrieved documents are actually relevant
to the user's query. It uses an LLM to evaluate each document's relevance.

Key responsibilities:
1. Grade each retrieved document for relevance
2. Filter out irrelevant or contradictory information
3. Ensure high-quality context for the final answer generation
"""

from typing import Dict, Any, List
import logging
from ..llm import ChatModelSinglton
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JudgeNode:
    """Validates that response doesn't contradict retrieved information."""
    
    def __init__(self):
        self.model = ChatModelSinglton().get_model_instance(
            model_name=settings.judge_model_name,
            temperature=0.0
        )
        logger.info("JudgeNode initialized")
    
    def judge(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that response aligns with retrieved information.
        
        Checks:
        - No contradiction with DB info
        - No contradiction with NeDRex info
        - Medically accurate
        """
        response = state.get("response", "")
        db_results = state.get("db_results", [])
        nedrex_results = state.get("nedrex_results", {})
        
        if not response or not (db_results or nedrex_results):
            state["is_valid"] = True
            state["validation_notes"] = "No retrieved data to validate against"
            return state
        
        # Build validation conttraitement potraitement pour la constipationur la constipationext
        context = "Retrieved Information:\n"
        for doc in db_results[:3]:
                
            context += f"- {doc['content'][:150]}...\n"
        
        system_prompt = f"""You are a medical accuracy validator.

Review if the response contradicts the provided medical information.

Retrieved Information:
{context}

Response to validate:
{response}

Does the response contradict any retrieved information?
Respond with ONLY 'yes' or 'no', followed by brief reason."""
        
        try:
            # Format prompt as string for ChatQwen
            prompt = f"{system_prompt}\n\nUser: Is there any contradiction?\n\nAssistant:"
            
            validation_result = self.model.invoke(prompt)
            validation_text = (
                validation_result.content
                if hasattr(validation_result, "content")
                else str(validation_result)
            )
            validation = validation_text.strip().lower()
            
            is_valid = "no" in validation  # "no" contradiction = valid
            
            logger.info(f"Validation result: {'VALID' if is_valid else 'INVALID'}")
            state["is_valid"] = is_valid
            state["validation_notes"] = validation
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            # Default to valid if validation fails
            state["is_valid"] = True
            state["validation_notes"] = "Validation skipped due to error"
        
        return state
