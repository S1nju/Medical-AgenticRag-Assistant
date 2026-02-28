"""
Agent Tools - Simple functions for the agent to use.

These are NOT nodes - they are tools that nodes can call to perform actions.
"""

from typing import Dict, Any, List, Optional
import logging
from app.tools.retriever import HybridRetriever
from app.tools.nedrexapi import NeDRexAPI
from app.core.config import settings

logger = logging.getLogger(__name__)

# ============================================================================
# RETRIEVAL TOOL - French Vector Database
# ============================================================================

_retriever_instance = None


def get_retriever() -> HybridRetriever:
    """Get singleton retriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance


def retrieve_from_french_db(query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Retrieve documents from French drug database.
    
    Args:
        query: Query in French
        top_k: Number of documents to retrieve (defaults to config)
    
    Returns:
        List of retrieved documents with scores and content
    """
    retriever = get_retriever()
    top_k = top_k or settings.retrieval_top_k
    
    logger.info(f"Retrieving documents for query: {query[:80]}...")
    
    try:
        results = retriever.hybrid_search(query, limit=top_k)
        
        if results:
            # Re-rank results
            reranked = retriever.rerank(query, results)
            logger.info(f"Retrieved and re-ranked {len(reranked)} documents")
            return reranked
        else:
            logger.warning("No documents retrieved from database")
            return []
            
    except Exception as e:
        logger.error(f"Error retrieving from database: {e}")
        return []


# ============================================================================
# NeDRex TOOLS - All Collection Types
# ============================================================================

_nedrex_instance = None


def get_nedrex() -> NeDRexAPI:
    """Get singleton NeDRex instance."""
    global _nedrex_instance
    if _nedrex_instance is None:
        _nedrex_instance = NeDRexAPI()
    return _nedrex_instance


# Entity Collections
def search_nedrex_disorder(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search for disorder/disease information."""
    nedrex = get_nedrex()
    logger.info(f"Searching NeDRex Disorder: {query}")
    try:
        return nedrex.search_disorder(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Error searching disorder: {e}")
        return {"success": False, "error": str(e)}


def search_nedrex_drug(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search for drug information."""
    nedrex = get_nedrex()
    logger.info(f"Searching NeDRex Drug: {query}")
    try:
        return nedrex.search_drug(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Error searching drug: {e}")
        return {"success": False, "error": str(e)}


def search_nedrex_gene(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search for gene information."""
    nedrex = get_nedrex()
    logger.info(f"Searching NeDRex Gene: {query}")
    try:
        return nedrex.search_gene(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Error searching gene: {e}")
        return {"success": False, "error": str(e)}


def search_nedrex_protein(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search for protein information."""
    nedrex = get_nedrex()
    logger.info(f"Searching NeDRex Protein: {query}")
    try:
        return nedrex.search_protein(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Error searching protein: {e}")
        return {"success": False, "error": str(e)}


def search_nedrex_phenotype(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search for phenotype/symptom information."""
    nedrex = get_nedrex()
    logger.info(f"Searching NeDRex Phenotype: {query}")
    try:
        return nedrex.search_phenotype(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Error searching phenotype: {e}")
        return {"success": False, "error": str(e)}


def search_nedrex_side_effect(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search for side effect information."""
    nedrex = get_nedrex()
    logger.info(f"Searching NeDRex SideEffect: {query}")
    try:
        return nedrex.search_side_effect(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Error searching side effect: {e}")
        return {"success": False, "error": str(e)}


def search_nedrex_pathway(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search for biological pathway information."""
    nedrex = get_nedrex()
    logger.info(f"Searching NeDRex Pathway: {query}")
    try:
        return nedrex.search_pathway(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Error searching pathway: {e}")
        return {"success": False, "error": str(e)}


def search_nedrex_tissue(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search for tissue information."""
    nedrex = get_nedrex()
    logger.info(f"Searching NeDRex Tissue: {query}")
    try:
        return nedrex.search_tissue(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Error searching tissue: {e}")
        return {"success": False, "error": str(e)}


def search_nedrex_go(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search for Gene Ontology terms."""
    nedrex = get_nedrex()
    logger.info(f"Searching NeDRex GO: {query}")
    try:
        return nedrex.search_go(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Error searching GO: {e}")
        return {"success": False, "error": str(e)}


def search_nedrex_signature(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search for molecular signatures."""
    nedrex = get_nedrex()
    logger.info(f"Searching NeDRex Signature: {query}")
    try:
        return nedrex.search_signature(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Error searching signature: {e}")
        return {"success": False, "error": str(e)}


# Relationship Collections
def search_nedrex_disorder_phenotype(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search for disorder-phenotype relationships."""
    nedrex = get_nedrex()
    logger.info(f"Searching NeDRex DisorderHasPhenotype: {query}")
    try:
        return nedrex.search_disorder_phenotype(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Error searching disorder-phenotype: {e}")
        return {"success": False, "error": str(e)}


def search_nedrex_disorder_subtype(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search for disorder subtype relationships."""
    nedrex = get_nedrex()
    logger.info(f"Searching NeDRex DisorderIsSubtypeOfDisorder: {query}")
    try:
        return nedrex.search_disorder_subtype(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Error searching disorder subtype: {e}")
        return {"success": False, "error": str(e)}


def search_nedrex_drug_indication(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search for drug indications (what drugs treat)."""
    nedrex = get_nedrex()
    logger.info(f"Searching NeDRex DrugHasIndication: {query}")
    try:
        return nedrex.search_drug_indication(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Error searching drug indication: {e}")
        return {"success": False, "error": str(e)}


def search_nedrex_drug_contraindication(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search for drug contraindications."""
    nedrex = get_nedrex()
    logger.info(f"Searching NeDRex DrugHasContraindication: {query}")
    try:
        return nedrex.search_drug_contraindication(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Error searching drug contraindication: {e}")
        return {"success": False, "error": str(e)}


def search_nedrex_drug_side_effect(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search for drug side effects."""
    nedrex = get_nedrex()
    logger.info(f"Searching NeDRex DrugHasSideEffect: {query}")
    try:
        return nedrex.search_drug_side_effect(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Error searching drug side effect: {e}")
        return {"success": False, "error": str(e)}


def search_nedrex_drug_target(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search for drug targets (proteins/genes)."""
    nedrex = get_nedrex()
    logger.info(f"Searching NeDRex DrugHasTarget: {query}")
    try:
        return nedrex.search_drug_target(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Error searching drug target: {e}")
        return {"success": False, "error": str(e)}


def search_nedrex_gene_disorder(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search for gene-disorder associations."""
    nedrex = get_nedrex()
    logger.info(f"Searching NeDRex GeneAssociatedWithDisorder: {query}")
    try:
        return nedrex.search_gene_disorder(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Error searching gene-disorder: {e}")
        return {"success": False, "error": str(e)}


def search_nedrex_gene_tissue(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search for gene expression in tissues."""
    nedrex = get_nedrex()
    logger.info(f"Searching NeDRex GeneExpressedInTissue: {query}")
    try:
        return nedrex.search_gene_tissue(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Error searching gene-tissue: {e}")
        return {"success": False, "error": str(e)}


def search_nedrex_protein_gene(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search for protein-gene encoding relationships."""
    nedrex = get_nedrex()
    logger.info(f"Searching NeDRex ProteinEncodedByGene: {query}")
    try:
        return nedrex.search_protein_gene(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Error searching protein-gene: {e}")
        return {"success": False, "error": str(e)}


def search_nedrex_protein_go(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search for protein GO annotations."""
    nedrex = get_nedrex()
    logger.info(f"Searching NeDRex ProteinHasGoAnnotation: {query}")
    try:
        return nedrex.search_protein_go(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Error searching protein-GO: {e}")
        return {"success": False, "error": str(e)}


def search_nedrex_side_effect_phenotype(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search for side effect-phenotype relationships."""
    nedrex = get_nedrex()
    logger.info(f"Searching NeDRex SideEffectSameAsPhenotype: {query}")
    try:
        return nedrex.search_side_effect_phenotype(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Error searching side effect-phenotype: {e}")
        return {"success": False, "error": str(e)}


def search_nedrex_variant_disorder(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search for genetic variant-disorder associations."""
    nedrex = get_nedrex()
    logger.info(f"Searching NeDRex VariantAssociatedWithDisorder: {query}")
    try:
        return nedrex.search_variant_disorder(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Error searching variant-disorder: {e}")
        return {"success": False, "error": str(e)}


# Mapping of collection names to functions for dynamic calling
NEDREX_FUNCTION_MAP = {
    "disorder": search_nedrex_disorder,
    "drug": search_nedrex_drug,
    "gene": search_nedrex_gene,
    "protein": search_nedrex_protein,
    "phenotype": search_nedrex_phenotype,
    "side_effect": search_nedrex_side_effect,
    "pathway": search_nedrex_pathway,
    "tissue": search_nedrex_tissue,
    "go": search_nedrex_go,
    "signature": search_nedrex_signature,
    "disorder_phenotype": search_nedrex_disorder_phenotype,
    "disorder_subtype": search_nedrex_disorder_subtype,
    "drug_indication": search_nedrex_drug_indication,
    "drug_contraindication": search_nedrex_drug_contraindication,
    "drug_side_effect": search_nedrex_drug_side_effect,
    "drug_target": search_nedrex_drug_target,
    "gene_disorder": search_nedrex_gene_disorder,
    "gene_tissue": search_nedrex_gene_tissue,
    "protein_gene": search_nedrex_protein_gene,
    "protein_go": search_nedrex_protein_go,
    "side_effect_phenotype": search_nedrex_side_effect_phenotype,
    "variant_disorder": search_nedrex_variant_disorder,
}


# Legacy function for backward compatibility
def search_nedrex_disease(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Legacy function - use search_nedrex_disorder instead."""
    return search_nedrex_disorder(query, top_k)


def search_nedrex_symptom(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Legacy function - use search_nedrex_phenotype instead."""
    return search_nedrex_phenotype(query, top_k)


# ============================================================================
# NeDRex DISORDER HIERARCHY & LOOKUP TOOLS
# ============================================================================

def get_disorder_by_icd10(icd10_codes: List[str]) -> Dict[str, Any]:
    """
    Get disorders by ICD-10 codes.
    
    Args:
        icd10_codes: List of ICD-10 codes (3 or 4 character)
        
    Returns:
        Disorders matching the ICD-10 codes
        
    Example:
        >>> get_disorder_by_icd10(["E10", "E11.9"])
    """
    nedrex = get_nedrex()
    logger.info(f"Getting disorders by ICD-10: {icd10_codes}")
    try:
        return nedrex.get_disorder_by_icd10(icd10_codes)
    except Exception as e:
        logger.error(f"Error getting disorders by ICD-10: {e}")
        return {"success": False, "error": str(e)}


def get_disorder_hierarchy(disorder_id: str, relation_type: str = "children") -> Dict[str, Any]:
    """
    Get disorder hierarchy relationships.
    
    Args:
        disorder_id: Disorder ID (e.g., mondo.0005252, omim.130020)
        relation_type: Type of relationship - 'children', 'parents', 'ancestors', or 'descendants'
        
    Returns:
        Dictionary with related disorders
    """
    nedrex = get_nedrex()
    logger.info(f"Getting {relation_type} for disorder: {disorder_id}")
    
    try:
        if relation_type == "children":
            return nedrex.get_disorder_children([disorder_id])
        elif relation_type == "parents":
            return nedrex.get_disorder_parents([disorder_id])
        elif relation_type == "ancestors":
            return nedrex.get_disorder_ancestors([disorder_id])
        elif relation_type == "descendants":
            return nedrex.get_disorder_descendants([disorder_id])
        else:
            return {"success": False, "error": f"Invalid relation_type: {relation_type}"}
    except Exception as e:
        logger.error(f"Error getting disorder {relation_type}: {e}")
        return {"success": False, "error": str(e)}


def get_disorder_ancestors(disorder_ids: List[str]) -> Dict[str, Any]:
    """Get ancestor disorders."""
    nedrex = get_nedrex()
    logger.info(f"Getting ancestors for: {disorder_ids}")
    try:
        return nedrex.get_disorder_ancestors(disorder_ids)
    except Exception as e:
        logger.error(f"Error getting ancestors: {e}")
        return {"success": False, "error": str(e)}


def get_disorder_descendants(disorder_ids: List[str]) -> Dict[str, Any]:
    """Get descendant disorders."""
    nedrex = get_nedrex()
    logger.info(f"Getting descendants for: {disorder_ids}")
    try:
        return nedrex.get_disorder_descendants(disorder_ids)
    except Exception as e:
        logger.error(f"Error getting descendants: {e}")
        return {"success": False, "error": str(e)}


def get_disorder_parents(disorder_ids: List[str]) -> Dict[str, Any]:
    """Get parent disorders."""
    nedrex = get_nedrex()
    logger.info(f"Getting parents for: {disorder_ids}")
    try:
        return nedrex.get_disorder_parents(disorder_ids)
    except Exception as e:
        logger.error(f"Error getting parents: {e}")
        return {"success": False, "error": str(e)}


def get_disorder_children(disorder_ids: List[str]) -> Dict[str, Any]:
    """Get child disorders."""
    nedrex = get_nedrex()
    logger.info(f"Getting children for: {disorder_ids}")
    try:
        return nedrex.get_disorder_children(disorder_ids)
    except Exception as e:
        logger.error(f"Error getting children: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# COMBINED TOOL
# ============================================================================

def search_medical_info(query: str, use_db: bool = True, use_nedrex: bool = True) -> Dict[str, Any]:
    """
    Search both database and NeDRex for medical information.
    
    Args:
        query: Search query in French
        use_db: Whether to search vector database
        use_nedrex: Whether to search NeDRex API
    
    Returns:
        Combined results from both sources
    """
    result = {
        "db_results": [],
        "nedrex_results": {}
    }
    
    if use_db:
        result["db_results"] = retrieve_from_french_db(query)
    
    if use_nedrex:
        # Use simplified embeddings search
        nedrex = get_nedrex()
        result["nedrex_results"] = nedrex.query(query, top_k=5)
    
    return result
