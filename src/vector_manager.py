from typing import List, Dict, Any
from dataclasses import dataclass
from interfaces import VectorAPIClient, LLMWareEmbeddingClient
import logging
from constants import VECTOR_SIMILARITY_THRESHOLD

# Add llmware imports
from llmware.models import ModelCatalog
from llmware.embeddings import EmbeddingHandler

logger = logging.getLogger(__name__)

@dataclass
class VectorData:
    id: str
    vector: List[float]
    metadata: Dict[str, Any]

@dataclass
class VectorSearchResult:
    id: str
    score: float
    metadata: Dict[str, Any]

class VectorManager:
    def __init__(self, vector_api_client: VectorAPIClient):
        self.vector_api_client = vector_api_client
        
    @classmethod
    def from_llmware(cls, model_name: str, library_name: str = "default"):
        """Create VectorManager using llmware embedding model"""
        client = LLMWareEmbeddingClient(model_name, library_name)
        return cls(client)
        
    async def find_similar_terms(self, query_embedding: List[float]) -> List[VectorSearchResult]:
        """Find similar terms using vector similarity search"""
        try:
            results = await self.vector_api_client.search_vectors(query_embedding)
            
            # Log all scores and metadata for debugging
            for result in results:
                logger.debug(f"Result ID: {result.id}, Score: {result.score}, Metadata: {result.metadata}")
            
            filtered_results = [
                result for result in results 
                if result.score >= VECTOR_SIMILARITY_THRESHOLD
            ]
            logger.debug(f"Vector search found {len(results)} results, {len(filtered_results)} above threshold")
            return filtered_results
        except Exception as e:
            logger.error(f"Error in vector search: {e}", exc_info=True)
            return []
        
    async def store_term_vectors(self, terms: List[Dict[str, Any]], vectors: List[List[float]]) -> bool:
        """Store term vectors with their metadata"""
        vector_data = [
            VectorData(
                id=str(i),
                vector=vector,
                metadata=term
            )
            for i, (term, vector) in enumerate(zip(terms, vectors))
        ]
        return await self.vector_api_client.store_vectors(vector_data) 