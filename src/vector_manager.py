from typing import List, Dict, Any
from dataclasses import dataclass
from interfaces import VectorAPIClient
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
        
    async def find_similar_terms(self, query_embedding: List[float]) -> List[VectorSearchResult]:
        """Find similar terms using vector similarity search"""
        return await self.vector_api_client.search_vectors(query_embedding)
        
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