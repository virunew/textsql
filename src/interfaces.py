from typing import List
import aiohttp
import asyncio
# Example configurations for different services
# Vector Database API Cfrom typing import Listlient Implementations


from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any
import numpy as np
from dataclasses import dataclass

@dataclass
class LLMRequest:
    """
    Represents a request to an LLM API.
    
    Attributes:
        prompt: The text prompt to send to the LLM
        temperature: Controls randomness in the response (0.0 to 1.0)
        max_tokens: Maximum number of tokens in the response
        additional_context: Optional dictionary of additional context
    """
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 500
    additional_context: Optional[Dict[str, Any]] = None

@dataclass
class LLMResponse:
    """
    Represents a response from an LLM API.
    
    Attributes:
        text: The generated text response
        metadata: Dictionary containing response metadata such as:
            - model: The model used for generation
            - finish_reason: Why the generation stopped
            - usage: Token usage statistics
            - created: Timestamp of creation
            - message_id: Unique identifier for the response (if provided)
            - stop_reason: Alternative finish reason for some providers
            - deployment: Deployment information for Azure
    """
    text: str
    metadata: Dict[str, Any]

@dataclass
class VectorData:
    """
    Represents a vector and its metadata for storage.
    
    Attributes:
        id: Unique identifier for the vector
        vector: The numerical vector representation
        metadata: Additional information about the vector
        namespace: Optional grouping for vectors
    """
    id: str
    vector: Union[List[float], np.ndarray]
    metadata: Dict[str, Any]
    namespace: Optional[str] = None

@dataclass
class VectorSearchResult:
    """
    Represents a search result from vector similarity search.
    
    Attributes:
        id: Identifier of the matched vector
        score: Similarity score (typically between 0 and 1)
        metadata: Associated metadata of the matched vector
        vector: Optional original vector (some APIs return this)
    """
    id: str
    score: float
    metadata: Dict[str, Any]
    vector: Optional[Union[List[float], np.ndarray]] = None

class VectorAPIClient(ABC):
    """Abstract base class for vector database interactions"""
    
    @abstractmethod
    async def store_vectors(self, vectors: List[VectorData]) -> bool:
        """
        Store multiple vectors in the vector database.

        Args:
            vectors: List of VectorData objects containing vectors and metadata

        Returns:
            bool: True if storage was successful, False otherwise

        Raises:
            VectorDBError: If there's an error communicating with the vector database
        """
        pass

    @abstractmethod
    async def search_vectors(
        self,
        query_vector: Union[List[float], np.ndarray],
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter_metadata: Optional[Dict[str, any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[VectorSearchResult]:
        """
        Search for similar vectors in the database.

        Args:
            query_vector: Vector to search for
            top_k: Number of similar vectors to return
            namespace: Optional namespace to search within
            filter_metadata: Optional metadata filters to apply

        Returns:
            List[VectorSearchResult]: List of matching vectors with similarity scores

        Raises:
            VectorDBError: If there's an error communicating with the vector database
        """
        pass

    @abstractmethod
    async def delete_vectors(
        self,
        vector_ids: List[str],
        namespace: Optional[str] = None
    ) -> bool:
        """
        Delete vectors from the database.

        Args:
            vector_ids: List of vector IDs to delete
            namespace: Optional namespace the vectors belong to

        Returns:
            bool: True if deletion was successful, False otherwise

        Raises:
            VectorDBError: If there's an error communicating with the vector database
        """
        pass

class LLMAPIClient(ABC):
    """Abstract base class for LLM API interactions"""
    
    @abstractmethod
    async def generate_completion(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a completion from the LLM
        
        Args:
            request: LLMRequest containing the prompt and parameters
            
        Returns:
            LLMResponse containing the generated text and metadata
            
        Raises:
            LLMError: If there's an error calling the LLM API
        """
        pass

class VectorDBError(Exception):
    """Custom exception for vector database operations"""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error

class LLMError(Exception):
    """Custom exception for LLM API operations"""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error

class VectorManager:
    """Manages vector operations using the configured API client"""
    
    def __init__(self, api_client: VectorAPIClient):
        self.api_client = api_client
    
    async def store_term_vectors(self, domain_terms: List[dict]) -> bool:
        """Store vectors for domain terms"""
        vectors = [
            VectorData(
                id=f"term_{term['term'].replace(' ', '_')}",
                vector=term['embedding'],
                metadata={
                    'term': term['term'],
                    'description': term['description'],
                    'table': term.get('table'),
                    'column': term.get('column')
                }
            )
            for term in domain_terms
        ]
        
        return await self.api_client.store_vectors(vectors)
    
    async def find_similar_terms(self, query_vector: List[float], threshold: float = 0.7) -> List[VectorSearchResult]:
        """Find similar terms using vector similarity search"""
        results = await self.api_client.search_vectors(query_vector)
        return results