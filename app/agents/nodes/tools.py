"""
Tools Node - Intelligently selects and calls appropriate NeDRex collection functions.

Uses LLM to determine which NeDRex collections to query based on the user's question.
"""

import logging
from typing import Dict, Any, List
from app.tools.agent_tools import (
    retrieve_from_french_db, 
    NEDREX_FUNCTION_MAP,
    search_nedrex_disorder,
    search_nedrex_drug,
    search_nedrex_gene,
    search_nedrex_protein,
    search_nedrex_phenotype,
)
from ..llm import ChatModelSinglton
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolsNode:
    """Retrieves information using DB and/or NeDRex with intelligent collection selection."""
    
    def __init__(self):
        self.model = ChatModelSinglton().get_model_instance(
            model_name=settings.judge_model_name,
            temperature=0.0
        )
        logger.info("ToolsNode initialized with LLM-based collection selection")
    
    def _translate_to_english(self, french_query: str) -> str:
        """
        Translate French query to English for NeDRex API.
        
        Args:
            french_query: Query in French
            
        Returns:
            Query translated to English
        """
        system_prompt = """You are a medical translator. Translate the French medical query to English.

Keep medical terms, drug names, and scientific terminology accurate.
Return ONLY the English translation, nothing else.

Examples:
- "symptômes de la maladie d'Alzheimer" → "symptoms of Alzheimer's disease"
- "effets secondaires de l'aspirine" → "side effects of aspirin"
- "gènes associés au diabète" → "genes associated with diabetes"
"""
        
        try:
            prompt = f"{system_prompt}\n\nFrench query: {french_query}\n\nEnglish translation:"
            
            result = self.model.invoke(prompt)
            english_query = (
                result.content
                if hasattr(result, "content")
                else str(result)
            ).strip()
            
            logger.info(f"Translated for NeDRex: '{french_query}' → '{english_query}'")
            return english_query
            
        except Exception as e:
            logger.error(f"Translation error: {e}, using original query")
            return french_query
    
    def _select_nedrex_collections(self, query: str) -> List[str]:
        """
        Use LLM to determine which NeDRex collections to query.
        
        Args:
            query: The user's medical question
            
        Returns:
            List of collection names to query (e.g., ['disorder', 'drug_indication'])
        """
        system_prompt = """You are a medical information retrieval expert. Given a medical query, select which NeDRex database collections to search.

Available collections (select 1-3 most relevant):
- disorder: diseases, conditions, medical disorders
- drug: medications, pharmaceuticals, treatments
- gene: genetic information, gene names
- protein: protein information, protein names
- phenotype: symptoms, observable characteristics
- side_effect: adverse effects, side effects
- pathway: biological pathways, molecular pathways
- tissue: tissue types, organ information
- go: gene ontology terms
- disorder_phenotype: which symptoms belong to which disorders
- drug_indication: what drugs treat which conditions
- drug_contraindication: when drugs should NOT be used
- drug_side_effect: side effects of specific drugs
- drug_target: what proteins/genes drugs target
- gene_disorder: which genes are associated with disorders
- gene_tissue: which genes are expressed in which tissues
- protein_gene: which proteins are encoded by which genes

Respond with ONLY a comma-separated list of collection names (1-3 collections max).

Examples:
Query: "What are the symptoms of Alzheimer's disease?"
Answer: disorder, disorder_phenotype, phenotype

Query: "What drugs treat diabetes?"
Answer: disorder, drug, drug_indication

Query: "Side effects of aspirin"
Answer: drug, drug_side_effect, side_effect

Query: "Which genes cause breast cancer?"
Answer: disorder, gene, gene_disorder

Query: "What is ibuprofen used for?"
Answer: drug, drug_indication"""

        try:
            prompt = f"{system_prompt}\n\nQuery: {query}\nAnswer:"
            
            result = self.model.invoke(prompt)
            response_text = (
                result.content
                if hasattr(result, "content")
                else str(result)
            )
            
            # Parse comma-separated collections
            collections = [c.strip().lower() for c in response_text.split(",")]
            
            # Validate collections
            valid_collections = [c for c in collections if c in NEDREX_FUNCTION_MAP]
            
            if not valid_collections:
                # Default fallback
                logger.warning(f"No valid collections selected, using defaults: {collections}")
                valid_collections = ["disorder", "drug"]
            
            logger.info(f"Selected NeDRex collections: {valid_collections}")
            return valid_collections[:3]  # Limit to 3 collections max
            
        except Exception as e:
            logger.error(f"Error selecting collections: {e}")
            # Default fallback
            return ["disorder", "drug"]
    
    def use_tools(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute tools based on router decision and LLM collection selection.
        
        Args:
            state: Current agent state with tool_choice and query
            
        Returns:
            Updated state with db_results and nedrex_results
        """
        try:
            decision = state.get("tool_choice", "both").strip().lower()
            query = state.get("french_query", "")
            
            # Validate decision
            if decision not in ["db", "nedrex", "both"]:
                logger.warning(f"Invalid decision '{decision}', defaulting to 'both'")
                decision = "both"
            
            logger.info(f"Tools node executing with decision: {decision}")
            state["tool_choice"] = decision
            
            # Initialize results
            state["nedrex_results"] = {}
            state["nedrex_collections_used"] = []
            
            # Call database if needed
            if decision in ["db", "both"]:
                db_results = retrieve_from_french_db(query)
                state["db_results"] = db_results
                logger.info(f"Retrieved {len(db_results)} documents from French DB")
            
            # Call NeDRex if needed
            if decision in ["nedrex", "both"]:
                # Translate French query to English for NeDRex API
                english_query = self._translate_to_english(query)
                state["english_query"] = english_query  # Store in state
                logger.info(f"Using English query for NeDRex: '{english_query}'")
                
                # Use LLM to select which collections to query
                collections_to_query = self._select_nedrex_collections(
                    state.get("question", english_query)
                )
                
                # Query each selected collection
                combined_results = {
                    "collections": {},
                    "query": english_query,
                    "success": True
                }
                
                for collection_name in collections_to_query:
                    try:
                        # Get the function for this collection
                        search_func = NEDREX_FUNCTION_MAP.get(collection_name)
                        
                        if search_func:
                            logger.info(f"Querying NeDRex collection: {collection_name}")
                            # Use English query for NeDRex
                            result = search_func(english_query, top_k=5)
                            
                            if result.get("success", False):
                                combined_results["collections"][collection_name] = result
                                state["nedrex_collections_used"].append(collection_name)
                                logger.info(
                                    f"Retrieved {result.get('count', 0)} results "
                                    f"from {collection_name}"
                                )
                        else:
                            logger.warning(f"No function found for collection: {collection_name}")
                            
                    except Exception as e:
                        logger.error(f"Error querying collection {collection_name}: {e}")
                        combined_results["collections"][collection_name] = {
                            "success": False,
                            "error": str(e)
                        }
                
                state["nedrex_results"] = combined_results
                logger.info(
                    f"Completed NeDRex queries for collections: "
                    f"{state['nedrex_collections_used']}"
                )
            
        except Exception as e:
            logger.error(f"Tools node error: {e}")
            # Provide fallback defaults
            state["tool_choice"] = "both"
            state["db_results"] = retrieve_from_french_db(state.get("french_query", ""))
            state["nedrex_results"] = {
                "success": False,
                "error": str(e),
                "collections": {}
            }
            state["nedrex_collections_used"] = []
        
        return state

