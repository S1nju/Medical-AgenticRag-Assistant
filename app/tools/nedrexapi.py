"""
NeDRex API Integration - Network-based Drug Repurposing and Exploration.

Simplified integration using NeDRex embeddings query endpoint:
https://api.nedrex.net/open/embeddings/query

This tool queries the NeDRex knowledge base for medical information.
"""

from typing import Dict, Any, List, Optional
import requests
import logging
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeDRexAPI:
    """
    Simplified client for NeDRex embeddings API.
    
    Uses vector similarity search to find relevant medical information
    about diseases, drugs, genes, and proteins.
    """
   
    BASE_URL = "https://api.nedrex.net"
    EMBEDDINGS_ENDPOINT = f"{BASE_URL}/open/embeddings/query"
    
    # Disorder endpoints
    DISORDER_BY_ICD10_ENDPOINT = f"{BASE_URL}/open/disorder/get_by_icd10"
    DISORDER_DESCENDANTS_ENDPOINT = f"{BASE_URL}/open/disorder/descendants"
    DISORDER_ANCESTORS_ENDPOINT = f"{BASE_URL}/open/disorder/ancestors"
    DISORDER_PARENTS_ENDPOINT = f"{BASE_URL}/open/disorder/parents"
    DISORDER_CHILDREN_ENDPOINT = f"{BASE_URL}/open/disorder/children"
    
    DEFAULT_COLLECTION = "Disorder"  # Default collection name
    DEFAULT_TOP_K = 10
    
    def __init__(self, timeout: int = None):
        """
        Initialize NeDRex API client.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout or settings.nedrex_timeout
        logger.info(f"NeDRexAPI initialized with embeddings endpoint")
    
    def query(
        self,
        query: str,
        collection: str = None,
        top_k: int = None
    ) -> Dict[str, Any]:
        """
        Query NeDRex embeddings for relevant medical information.
        
        Args:
            query: Natural language query (e.g., "What is Alzheimer's disease?")
            collection: Collection to search in (default: "string")
            top_k: Number of results to return (default: 10)
        
        Returns:
            Dictionary containing:
            - results: List of relevant items with similarity scores
            - query: Original query
            - collection: Collection searched
            - count: Number of results
        
        Example:
            >>> api = NeDRexAPI()
            >>> results = api.query("What is AD5?")
            >>> for item in results['results']:
            >>>     print(f"{item['name']}: {item['score']}")
        """
        collection = collection or self.DEFAULT_COLLECTION
        top_k = top_k or self.DEFAULT_TOP_K
        
        logger.info(f"Querying NeDRex: '{query}' (top_k={top_k}, collection={collection})")
        headers = {
    "Content-Type": "application/json",
    "Accept": "application/json"
}
        payload = {
            "query": query,
            "collection": collection,
            "top": top_k
        }
        
        try:
                response = requests.post(
                headers=headers,
                url=self.EMBEDDINGS_ENDPOINT,
                json=payload,
                timeout=self.timeout
            )
                response.raise_for_status()
                raw_result = response.json()
                
                # Parse nested array structure [[{...}, {...}]]
                parsed_results = []
                if isinstance(raw_result, list) and len(raw_result) > 0:
                    if isinstance(raw_result[0], list):
                        # Nested array [[{...}]]
                        for item in raw_result[0]:
                            parsed_results.append({
                                "name": item.get("n.displayName", "Unknown"),
                                "description": item.get("n.description", "No description available"),
                                "score": item.get("score", 0.0),
                                "type": item.get("n.type", "Unknown"),
                                "synonyms": item.get("n.synonyms", []),
                                "domain_ids": item.get("n.domainIds", []),
                                "data_sources": item.get("n.dataSources", []),
                                "primary_id": item.get("n.primaryDomainId", "")
                            })
                
                # Log successful response
                results_count = len(parsed_results)
                logger.info(f"NeDRex returned {results_count} results")
                
                return {
                    "results": parsed_results,
                    "query": query,
                    "collection": collection,
                    "count": results_count,
                    "success": True
                }
                
        except requests.HTTPError as e:
            logger.error(f"NeDRex HTTP error {e.response.status_code}: {str(e)}")
            return {
                "results": [],
                "query": query,
                "error": f"HTTP {e.response.status_code}",
                "details": str(e),
                "success": False
            }
        except Exception as e:
            logger.error(f"NeDRex request error: {str(e)}")
            return {
                "results": [],
                "query": query,
                "error": "Request failed",
                "details": str(e),
                "success": False
            }
    
    # ============================================================================
    # COLLECTION-SPECIFIC SEARCH METHODS
    # ============================================================================
    
    def search_disorder(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for disorder/disease information."""
        logger.info(f"Searching Disorder: '{query}'")
        return self.query(query, collection="Disorder", top_k=top_k)
    
    def search_disorder_phenotype(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for disorder-phenotype relationships."""
        logger.info(f"Searching DisorderHasPhenotype: '{query}'")
        return self.query(query, collection="DisorderHasPhenotype", top_k=top_k)
    
    def search_disorder_subtype(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for disorder subtype relationships."""
        logger.info(f"Searching DisorderIsSubtypeOfDisorder: '{query}'")
        return self.query(query, collection="DisorderIsSubtypeOfDisorder", top_k=top_k)
    
    def search_drug(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for drug information."""
        logger.info(f"Searching Drug: '{query}'")
        return self.query(query, collection="Drug", top_k=top_k)
    
    def search_drug_contraindication(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for drug contraindications."""
        logger.info(f"Searching DrugHasContraindication: '{query}'")
        return self.query(query, collection="DrugHasContraindication", top_k=top_k)
    
    def search_drug_indication(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for drug indications (what drugs treat)."""
        logger.info(f"Searching DrugHasIndication: '{query}'")
        return self.query(query, collection="DrugHasIndication", top_k=top_k)
    
    def search_drug_side_effect(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for drug side effects."""
        logger.info(f"Searching DrugHasSideEffect: '{query}'")
        return self.query(query, collection="DrugHasSideEffect", top_k=top_k)
    
    def search_drug_target(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for drug targets (proteins/genes)."""
        logger.info(f"Searching DrugHasTarget: '{query}'")
        return self.query(query, collection="DrugHasTarget", top_k=top_k)
    
    def search_gene(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for gene information."""
        logger.info(f"Searching Gene: '{query}'")
        return self.query(query, collection="Gene", top_k=top_k)
    
    def search_gene_disorder(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for gene-disorder associations."""
        logger.info(f"Searching GeneAssociatedWithDisorder: '{query}'")
        return self.query(query, collection="GeneAssociatedWithDisorder", top_k=top_k)
    
    def search_gene_tissue(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for gene expression in tissues."""
        logger.info(f"Searching GeneExpressedInTissue: '{query}'")
        return self.query(query, collection="GeneExpressedInTissue", top_k=top_k)
    
    def search_go(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for Gene Ontology terms."""
        logger.info(f"Searching GO: '{query}'")
        return self.query(query, collection="GO", top_k=top_k)
    
    def search_pathway(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for biological pathways."""
        logger.info(f"Searching Pathway: '{query}'")
        return self.query(query, collection="Pathway", top_k=top_k)
    
    def search_phenotype(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for phenotype information."""
        logger.info(f"Searching Phenotype: '{query}'")
        return self.query(query, collection="Phenotype", top_k=top_k)
    
    def search_protein(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for protein information."""
        logger.info(f"Searching Protein: '{query}'")
        return self.query(query, collection="Protein", top_k=top_k)
    
    def search_protein_gene(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for protein-gene encoding relationships."""
        logger.info(f"Searching ProteinEncodedByGene: '{query}'")
        return self.query(query, collection="ProteinEncodedByGene", top_k=top_k)
    
    def search_protein_go(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for protein GO annotations."""
        logger.info(f"Searching ProteinHasGoAnnotation: '{query}'")
        return self.query(query, collection="ProteinHasGoAnnotation", top_k=top_k)
    
    def search_side_effect(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for side effect information."""
        logger.info(f"Searching SideEffect: '{query}'")
        return self.query(query, collection="SideEffect", top_k=top_k)
    
    def search_side_effect_phenotype(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for side effect-phenotype relationships."""
        logger.info(f"Searching SideEffectSameAsPhenotype: '{query}'")
        return self.query(query, collection="SideEffectSameAsPhenotype", top_k=top_k)
    
    def search_signature(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for molecular signatures."""
        logger.info(f"Searching Signature: '{query}'")
        return self.query(query, collection="Signature", top_k=top_k)
    
    def search_tissue(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for tissue information."""
        logger.info(f"Searching Tissue: '{query}'")
        return self.query(query, collection="Tissue", top_k=top_k)
    
    def search_variant_disorder(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for genetic variant-disorder associations."""
        logger.info(f"Searching VariantAssociatedWithDisorder: '{query}'")
        return self.query(query, collection="VariantAssociatedWithDisorder", top_k=top_k)
    
    # ============================================================================
    # DISORDER HIERARCHY & LOOKUP METHODS
    # ============================================================================
    
    def get_disorder_by_icd10(self, icd10_codes: List[str]) -> Dict[str, Any]:
        """
        Get disorders by ICD-10 codes.
        
        Args:
            icd10_codes: List of ICD-10 codes (3 or 4 character)
            
        Returns:
            Dictionary with disorders matching the ICD-10 codes
            
        Example:
            >>> api.get_disorder_by_icd10(["E10", "E11.9"])
        """
        logger.info(f"Getting disorders by ICD-10: {icd10_codes}")
        
        try:
            params = {"q": icd10_codes}
            response = requests.get(
                self.DISORDER_BY_ICD10_ENDPOINT,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"Found {len(result) if isinstance(result, list) else 0} disorders")
            return {
                "success": True,
                "disorders": result,
                "icd10_codes": icd10_codes
            }
            
        except Exception as e:
            logger.error(f"Error getting disorders by ICD-10: {e}")
            return {
                "success": False,
                "error": str(e),
                "icd10_codes": icd10_codes
            }
    
    def get_disorder_descendants(self, disorder_ids: List[str]) -> Dict[str, Any]:
        """
        Get descendant disorders for given disorder IDs.
        
        Args:
            disorder_ids: List of disorder IDs (any domain ID, e.g., omim.130020, mondo.0005252)
            
        Returns:
            Dictionary mapping each disorder to its descendants
            
        Example:
            >>> api.get_disorder_descendants(["mondo.0005252"])
            Returns: {"mondo.0005252": ["mondo.0006727", "mondo.0004595", ...]}
        """
        logger.info(f"Getting descendants for disorders: {disorder_ids}")
        
        try:
            params = {"q": disorder_ids}
            response = requests.get(
                self.DISORDER_DESCENDANTS_ENDPOINT,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"Retrieved descendants for {len(result)} disorders")
            return {
                "success": True,
                "descendants": result,
                "query_ids": disorder_ids
            }
            
        except Exception as e:
            logger.error(f"Error getting disorder descendants: {e}")
            return {
                "success": False,
                "error": str(e),
                "query_ids": disorder_ids
            }
    
    def get_disorder_ancestors(self, disorder_ids: List[str]) -> Dict[str, Any]:
        """
        Get ancestor disorders for given disorder IDs.
        
        Args:
            disorder_ids: List of disorder IDs (any domain ID)
            
        Returns:
            Dictionary mapping each disorder to its ancestors
            
        Example:
            >>> api.get_disorder_ancestors(["mondo.0007523"])
            Returns: {"mondo.0007523": ["mondo.0020066", ...]}
        """
        logger.info(f"Getting ancestors for disorders: {disorder_ids}")
        
        try:
            params = {"q": disorder_ids}
            response = requests.get(
                self.DISORDER_ANCESTORS_ENDPOINT,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"Retrieved ancestors for {len(result)} disorders")
            return {
                "success": True,
                "ancestors": result,
                "query_ids": disorder_ids
            }
            
        except Exception as e:
            logger.error(f"Error getting disorder ancestors: {e}")
            return {
                "success": False,
                "error": str(e),
                "query_ids": disorder_ids
            }
    
    def get_disorder_parents(self, disorder_ids: List[str]) -> Dict[str, Any]:
        """
        Get parent disorders for given disorder IDs.
        
        Args:
            disorder_ids: List of disorder IDs (any domain ID)
            
        Returns:
            Dictionary mapping each disorder to its immediate parents
            
        Example:
            >>> api.get_disorder_parents(["mondo.0007523"])
            Returns: {"mondo.0007523": ["mondo.0020066"]}
        """
        logger.info(f"Getting parents for disorders: {disorder_ids}")
        
        try:
            params = {"q": disorder_ids}
            response = requests.get(
                self.DISORDER_PARENTS_ENDPOINT,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"Retrieved parents for {len(result)} disorders")
            return {
                "success": True,
                "parents": result,
                "query_ids": disorder_ids
            }
            
        except Exception as e:
            logger.error(f"Error getting disorder parents: {e}")
            return {
                "success": False,
                "error": str(e),
                "query_ids": disorder_ids
            }
    
    def get_disorder_children(self, disorder_ids: List[str]) -> Dict[str, Any]:
        """
        Get child disorders for given disorder IDs.
        
        Args:
            disorder_ids: List of disorder IDs (any domain ID)
            
        Returns:
            Dictionary mapping each disorder to its immediate children
            
        Example:
            >>> api.get_disorder_children(["mondo.0020066"])
            Returns: {"mondo.0020066": ["mondo.0007523", ...]}
        """
        logger.info(f"Getting children for disorders: {disorder_ids}")
        
        try:
            params = {"q": disorder_ids}
            response = requests.get(
                self.DISORDER_CHILDREN_ENDPOINT,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"Retrieved children for {len(result)} disorders")
            return {
                "success": True,
                "children": result,
                "query_ids": disorder_ids
            }
            
        except Exception as e:
            logger.error(f"Error getting disorder children: {e}")
            return {
                "success": False,
                "error": str(e),
                "query_ids": disorder_ids
            }
    
    def format_results_for_llm(self, results: Dict[str, Any]) -> str:
        """
        Format NeDRex results into a readable string for LLM context.
        
        Args:
            results: Results from query()
        
        Returns:
            Formatted string with key information
        """
        if not results.get("success", False):
            return f"NeDRex Error: {results.get('error', 'Unknown error')}"
        
        items = results.get("results", [])
        if not items:
            return "No results found in NeDRex."
        
        formatted = f"NeDRex Results (Collection: {results.get('collection', 'Unknown')}):\n\n"
        
        for i, item in enumerate(items, 1):
            name = item.get("name", "Unknown")
            score = item.get("score", 0.0)
            description = item.get("description")
            entity_type = item.get("type", "Unknown")
            synonyms = item.get("synonyms", [])
            
            formatted += f"{i}. **{name}** [{entity_type}] (relevance: {score:.3f})\n"
            
            if synonyms and synonyms is not None:
                formatted += f"   Synonyms: {', '.join(synonyms[:3])}\n"
            
            if description and description is not None:
                desc_text = description[:200] + "..." if len(description) > 200 else description
                formatted += f"   {desc_text}\n"
            
            formatted += "\n"
        
        return formatted


# Singleton instance
_nedrex_instance: Optional[NeDRexAPI] = None


def get_nedrex_api() -> NeDRexAPI:
    """Get or create a singleton instance of the NeDRex API client."""
    global _nedrex_instance
    if _nedrex_instance is None:
        _nedrex_instance = NeDRexAPI()
    return _nedrex_instance
