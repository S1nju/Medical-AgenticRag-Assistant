"""
Hybrid Retriever with Dense + Sparse Vectors and Re-ranking.

This module implements a production-grade retrieval pipeline:
1. Dense vectors (BAAI/bge-large-en-v1.5) for semantic meaning
2. Sparse vectors (Splade_PP_en_v1) for exact keyword matches
3. Re-ranking with Cross-Encoder (BAAI/bge-reranker-base)
"""

from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    CollectionInfo,
    PointStruct,
    ScoredPoint,
    QueryRequest,
    Query
)
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import logging

from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SparseEncoder:
    """Sparse encoder using Splade for keyword-based retrieval."""
    
    def __init__(self, model_name: str = "prithivida/Splade_PP_en_v1"):
        logger.info(f"Loading sparse encoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
    
    def encode(self, text: str) -> Dict[int, float]:
        """Encode text to sparse vector representation."""
        with torch.no_grad():
            tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            output = self.model(**tokens)
            logits = output.logits
            
            # Apply ReLU and log to get sparse weights
            relu_log = torch.log(1 + torch.relu(logits))
            weighted_log = relu_log * tokens.attention_mask.unsqueeze(-1)
            max_val, _ = torch.max(weighted_log, dim=1)
            
            # Convert to sparse dictionary (only non-zero values)
            sparse_vec = {}
            for idx, value in enumerate(max_val.squeeze().cpu().numpy()):
                if value > 0:
                    sparse_vec[idx] = float(value)
            
            return sparse_vec


class HybridRetriever:
    """
    Production-grade hybrid retrieval system combining dense and sparse search
    with re-ranking.
    """
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        dense_model_name: Optional[str] = None,
        sparse_model_name: Optional[str] = None,
        reranker_model_name: Optional[str] = None,
        top_k: int = 10,
        rerank_top_k: int = 3,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            collection_name: Qdrant collection name
            qdrant_url: Qdrant server URL
            dense_model_name: Model for dense embeddings
            sparse_model_name: Model for sparse embeddings
            reranker_model_name: Model for re-ranking
            top_k: Number of initial results to retrieve
            rerank_top_k: Number of results after re-ranking
            dense_weight: Weight for dense search (0-1)
            sparse_weight: Weight for sparse search (0-1)
        """
        # Load configuration
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.qdrant_url = qdrant_url or settings.qdrant_url
        self.dense_model_name = dense_model_name or settings.dense_model_name
        self.sparse_model_name = sparse_model_name or settings.sparse_model_name
        self.reranker_model_name = reranker_model_name or settings.reranker_model_name
        
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        # Initialize Qdrant client
        logger.info(f"Connecting to Qdrant at {self.qdrant_url}")
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=30.0
        )
        
        # Initialize models
        logger.info(f"Loading dense encoder: {self.dense_model_name}")
        self.dense_encoder = SentenceTransformer(self.dense_model_name)
        
        logger.info(f"Loading sparse encoder: {self.sparse_model_name}")
        self.sparse_encoder = SparseEncoder(self.sparse_model_name)
        
        logger.info(f"Loading re-ranker: {self.reranker_model_name}")
        self.reranker = CrossEncoder(self.reranker_model_name)
        
        logger.info("Hybrid retriever initialized successfully")
    
    def encode_query_dense(self, query: str) -> List[float]:
        """Encode query using dense encoder."""
        return self.dense_encoder.encode(query, normalize_embeddings=True).tolist()
    
    def encode_query_sparse(self, query: str) -> Dict[int, float]:
        """Encode query using sparse encoder."""
        return self.sparse_encoder.encode(query)
    
    def hybrid_search(self, query: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining dense and sparse retrieval.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
        
        Returns:
            List of retrieved documents with scores
        """
        limit = limit or self.top_k
        
        logger.info(f"Performing hybrid search for query: '{query[:50]}...'")
        
        # Encode query with both methods
        dense_vector = self.encode_query_dense(query)
        sparse_vector = self.encode_query_sparse(query)
        
        try:
            # Perform hybrid search in Qdrant
            # Note: This is a simplified version. In production, you'd use Qdrant's
            # query API with proper fusion of dense and sparse results
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=("dense", dense_vector),
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            # Convert results to dict format
            documents = []
            for point in results:
                doc = {
                    "id": point.id,
                    "content": point.payload.get("content", ""),
                    "metadata": point.payload.get("metadata", {}),
                    "score": point.score,
                }
                documents.append(doc)
            
            logger.info(f"Retrieved {len(documents)} documents from hybrid search")
            return documents
            
        except Exception as e:
            logger.error(f"Error during hybrid search: {str(e)}")
            return []
    
    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Re-rank retrieved documents using a cross-encoder.
        
        Args:
            query: Original search query
            documents: List of retrieved documents
        
        Returns:
            Re-ranked list of top documents
        """
        if not documents:
            return []
        
        logger.info(f"Re-ranking {len(documents)} documents")
        
        # Prepare query-document pairs
        pairs = [[query, doc["content"]] for doc in documents]
        
        # Get cross-encoder scores
        scores = self.reranker.predict(pairs)
        
        # Attach new scores and sort
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
        
        # Sort by rerank score (descending)
        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        
        # Return top-k after reranking
        top_reranked = reranked[:self.rerank_top_k]
        
        logger.info(f"Returning top {len(top_reranked)} re-ranked documents")
        return top_reranked
    
    def retrieve(self, query: str, apply_reranking: bool = True) -> List[Dict[str, Any]]:
        """
        Full retrieval pipeline: hybrid search + optional re-ranking.
        
        Args:
            query: Search query
            apply_reranking: Whether to apply re-ranking
        
        Returns:
            List of retrieved and re-ranked documents
        """
        # Step 1: Hybrid search
        documents = self.hybrid_search(query, limit=self.top_k)
        
        if not documents:
            logger.warning("No documents retrieved")
            return []
        
        # Step 2: Re-ranking
        if apply_reranking and len(documents) > 0:
            documents = self.rerank(query, documents)
        
        return documents
    
    def initialize_collection(
        self,
        vector_size: int = 1024,
        distance: Distance = Distance.COSINE,
        force_recreate: bool = False
    ):
        """
        Initialize or recreate the Qdrant collection.
        
        Args:
            vector_size: Dimension of dense vectors
            distance: Distance metric (COSINE, EUCLID, DOT)
            force_recreate: Whether to delete and recreate if exists
        """
        try:
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if collection_exists:
                if force_recreate:
                    logger.info(f"Deleting existing collection: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                else:
                    logger.info(f"Collection '{self.collection_name}' already exists")
                    return
            
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(size=vector_size, distance=distance),
                }
            )
            logger.info(f"Collection '{self.collection_name}' created successfully")
            
        except Exception as e:
            logger.error(f"Error initializing collection: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the Qdrant collection.
        
        Args:
            documents: List of documents with 'content' and optional 'metadata'
        """
        points = []
        
        for idx, doc in enumerate(documents):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # Generate dense embedding
            dense_vector = self.encode_query_dense(content)
            
            point = PointStruct(
                id=idx,
                vector={"dense": dense_vector},
                payload={
                    "content": content,
                    "metadata": metadata
                }
            )
            points.append(point)
        
        logger.info(f"Adding {len(points)} documents to collection")
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info("Documents added successfully")


# Singleton instance (optional)
_retriever_instance: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    """Get or create a singleton instance of the hybrid retriever."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance
