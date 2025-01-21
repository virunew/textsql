from typing import List, Dict, Optional, Union, Any
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass
import logging
# Add llmware imports
from llmware.models import ModelCatalog
from llmware.embeddings import EmbeddingHandler
from llmware.configs import LLMWareConfig

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
    model_name: Optional[str] = None # Added to support llmware models

@dataclass
class LLMResponse:
    """Represents a response from an LLM API"""
    text: str
    metadata: Dict[str, Any]
    usage: Optional[Dict[str, Any]] = None  # Add usage field with default None

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

# Add new llmware client classes
class LLMWareAPIClient(LLMAPIClient):
    """Implementation for llmware's LLM models"""
    
    def __init__(self, model_name: str):
        try:
            self.model = ModelCatalog().load_model(model_name)
            logging.info(f"Successfully loaded llmware model: {model_name}")
            logging.debug(f"Model type: {type(self.model)}")
            logging.debug(f"Available methods: {dir(self.model)}")
        except Exception as e:
            logging.error(f"Failed to load llmware model {model_name}: {e}")
            raise
        
    async def generate_completion(self, request: LLMRequest) -> LLMResponse:
        try:
            logging.debug(f"Generating completion with prompt:\n{request.prompt}")
            
            # Try different generation methods based on model type
            model_type = type(self.model).__name__
            logging.debug(f"Model type: {model_type}")
            
            if hasattr(self.model, 'generate_with_params'):  # GGUFGenerativeModel
                logging.debug("Using generate_with_params")
                response = self.model.generate_with_params(
                    prompt=request.prompt,
                    n_predict=request.max_tokens,
                    temp=request.temperature
                )
            elif hasattr(self.model, 'inference'):  # HFGenerativeModel
                logging.debug("Using inference")
                response = self.model.inference(
                    prompt=request.prompt,
                )
            else:
                logging.error(f"Unsupported model type: {model_type}")
                raise AttributeError(f"Model {model_type} has no supported generation method")
            
            logging.debug(f"Raw response type: {type(response)}")
            logging.debug(f"Raw response: {response}")
            
            # Handle different response formats based on model type
            if model_type == "GGUFGenerativeModel":
                text = response.get("llm_response", str(response))
            elif model_type == "HFGenerativeModel":
                text = response.get("generated_text", str(response))
            else:
                text = str(response)
            
            # Clean up the response text
            text = text.strip()
            if text.lower().startswith("sql query:"):
                text = text[len("sql query:"):].strip()
            
            logging.debug(f"Processed response text:\n{text}")
            
            return LLMResponse(
                text=text,
                metadata={
                    "model": self.model.model_name,
                    "model_type": model_type,
                    "raw_response": text
                }
            )
        except Exception as e:
            logging.error(f"Error generating completion with LLMWare model: {str(e)}", exc_info=True)
            raise

class LLMWareEmbeddingClient(VectorAPIClient):
    """Implementation for llmware's embedding models"""
    
    def __init__(self, model_name: str, library_name: str = "default"):
        self.model = ModelCatalog().load_model(model_name)
        self.embedding_handler = EmbeddingHandler(library_name)
        
    async def store_vectors(self, vectors: List[VectorData]) -> bool:
        try:
            # Convert to llmware format
            llmware_vectors = [{
                "id": v.id,
                "vector": v.vector,
                "metadata": v.metadata
            } for v in vectors]
            
            return await self.embedding_handler.store_vectors(llmware_vectors)
        except Exception as e:
            logging.error(f"Error storing vectors: {e}")
            return False
            
    async def search_vectors(self, query_vector, **kwargs) -> List[VectorSearchResult]:
        results = await self.embedding_handler.search_vectors(
            query_vector,
            top_k=kwargs.get("top_k", 10)
        )
        return [
            VectorSearchResult(
                id=r["id"],
                score=r["score"],
                metadata=r["metadata"]
            ) for r in results
        ]