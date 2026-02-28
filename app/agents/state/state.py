"""Agent state definition for LangGraph workflow."""

from typing_extensions import TypedDict
from typing import List, Dict, Any, Optional


class AgentState(TypedDict, total=False):
    # ========== INPUT ==========
    question: str  # User's original question (any language: Darija, French, English, Arabic)
    original_query: str  # Same as question, preserved
    
    # ========== TRANSLATION ==========
    french_query: str  # Query translated/converted to French for vector DB
    english_query: str  # Query translated to English for NeDRex API
    
    # ========== ROUTING & TOOLS ==========
    tool_choice: str  # 'db', 'nedrex', or 'both'
    
    # ========== RETRIEVAL RESULTS ==========
    db_results: List[Dict[str, Any]]  # Documents from French drug database
    nedrex_results: Dict[str, Any]  # Results from NeDRex API (single or combined)
    nedrex_collections_used: List[str]  # Which NeDRex collections were queried
    
    # ========== SYNTHESIS ==========
    response: str  # Final synthesized response
    
    # ========== VALIDATION ==========
    is_valid: bool  # Whether response contradicts retrieved info
    validation_notes: str  # Validation feedback
    
    # ========== METADATA ==========
    conversation_id: str
    steps_completed: List[str]  # Track which nodes ran

