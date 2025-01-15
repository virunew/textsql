from abc import ABC, abstractmethod
import aiohttp
import asyncio
from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass
import yaml
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the base project path and API key from environment variables
BASE_PROJECT_PATH = os.getenv("BASE_PROJECT_PATH")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Data Models
@dataclass
class VectorData:
    """Represents vector data for storage and retrieval"""
    id: str
    vector: List[float]
    metadata: Dict[str, any]

@dataclass
class VectorSearchResult:
    """Represents a vector similarity search result"""
    id: str
    score: float
    metadata: Dict[str, any]

@dataclass
class LLMRequest:
    """Represents a request to the LLM service"""
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 500
    additional_context: Dict[str, any] = None

@dataclass
class LLMResponse:
    """Represents a response from the LLM service"""
    text: str
    metadata: Dict[str, any] = None

# Abstract base classes for API clients
class VectorAPIClient(ABC):
    """Abstract base class for vector database API interactions"""
    
    @abstractmethod
    async def store_vectors(self, vectors: List[VectorData]) -> bool:
        """Store vectors in the vector database"""
        pass
    
    @abstractmethod
    async def search_vectors(self, query_vector: List[float], top_k: int = 10) -> List[VectorSearchResult]:
        """Search for similar vectors"""
        pass
    
    @abstractmethod
    async def delete_vectors(self, vector_ids: List[str]) -> bool:
        """Delete vectors from the database"""
        pass

class LLMAPIClient(ABC):
    """Abstract base class for LLM API interactions"""
    
    @abstractmethod
    async def generate_completion(self, request: LLMRequest) -> LLMResponse:
        """Generate completion from LLM"""
        pass

# Concrete implementations of API clients
class CustomVectorAPIClient(VectorAPIClient):
    """Implementation for your organization's vector database API"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def store_vectors(self, vectors: List[VectorData]) -> bool:
        """
        Store vectors using your organization's API endpoint
        
        Example API endpoint: POST /api/vectors/batch
        """
        async with aiohttp.ClientSession() as session:
            try:
                payload = {
                    "vectors": [
                        {
                            "id": v.id,
                            "values": v.vector,
                            "metadata": v.metadata
                        }
                        for v in vectors
                    ]
                }
                
                async with session.post(
                    f"{self.base_url}/api/vectors/batch",
                    headers=self.headers,
                    json=payload
                ) as response:
                    return response.status == 200
            except Exception as e:
                print(f"Error storing vectors: {e}")
                return False
    
    async def search_vectors(self, query_vector: List[float], top_k: int = 10) -> List[VectorSearchResult]:
        """
        Search vectors using your organization's API endpoint
        
        Example API endpoint: POST /api/vectors/search
        """
        async with aiohttp.ClientSession() as session:
            try:
                payload = {
                    "vector": query_vector,
                    "top_k": top_k
                }
                
                async with session.post(
                    f"{self.base_url}/api/vectors/search",
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        results = await response.json()
                        return [
                            VectorSearchResult(
                                id=r["id"],
                                score=r["score"],
                                metadata=r["metadata"]
                            )
                            for r in results["matches"]
                        ]
                    return []
            except Exception as e:
                print(f"Error searching vectors: {e}")
                return []
    
    async def delete_vectors(self, vector_ids: List[str]) -> bool:
        """
        Delete vectors using your organization's API endpoint
        
        Example API endpoint: DELETE /api/vectors/batch
        """
        async with aiohttp.ClientSession() as session:
            try:
                payload = {"ids": vector_ids}
                
                async with session.delete(
                    f"{self.base_url}/api/vectors/batch",
                    headers=self.headers,
                    json=payload
                ) as response:
                    return response.status == 200
            except Exception as e:
                print(f"Error deleting vectors: {e}")
                return False

class CustomLLMAPIClient(LLMAPIClient):
    """Implementation for your organization's LLM API"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate_completion(self, request: LLMRequest) -> LLMResponse:
        """
        Generate completion using your organization's LLM API endpoint
        
        Example API endpoint: POST /api/llm/complete
        """
        async with aiohttp.ClientSession() as session:
            try:
                payload = {
                    "prompt": request.prompt,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "context": request.additional_context
                }
                
                async with session.post(
                    f"{self.base_url}/api/llm/complete",
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return LLMResponse(
                            text=result["text"],
                            metadata=result.get("metadata")
                        )
                    raise Exception(f"LLM API error: {response.status}")
            except Exception as e:
                print(f"Error generating completion: {e}")
                raise

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
        return [r for r in results if r.score >= threshold]

class QueryTranslator:
    """Main class for translating natural language queries to SQL"""
    
    def __init__(
        self,
        config_path: str,
        vector_api_client: VectorAPIClient,
        llm_api_client: LLMAPIClient
    ):
        self.config_manager = self._load_config(config_path)
        self.vector_manager = VectorManager(vector_api_client)
        self.llm_client = llm_api_client
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML files"""
        config = {}
        for config_file in Path(config_path).glob("*.yaml"):
            with open(config_file, "r") as f:
                config[config_file.stem] = yaml.safe_load(f)
        return config
    
    async def translate_to_sql(self, natural_query: str) -> str:
        """Translate natural language query to SQL"""
        try:
            # Find similar terms using vector search
            query_embedding = self._generate_embedding(natural_query)
            similar_terms = await self.vector_manager.find_similar_terms(query_embedding)
            
            # Prepare LLM request
            prompt = self._prepare_llm_prompt(natural_query, similar_terms)
            request = LLMRequest(
                prompt=prompt,
                temperature=0.3,  # Lower temperature for more focused SQL generation
                additional_context={
                    "similar_terms": [t.metadata for t in similar_terms],
                    "schema": self.config_manager["schema"]
                }
            )
            
            # Generate SQL using LLM
            response = await self.llm_client.generate_completion(request)
            
            return response.text
        
        except Exception as e:
            print(f"Error translating query: {e}")
            raise

# Example usage
async def main():
    # Initialize API clients with your organization's endpoints
    vector_api_client = CustomVectorAPIClient(
        base_url="https://your-vector-api.com",
        api_key="your-vector-api-key"
    )
    
    llm_api_client = CustomLLMAPIClient(
        base_url="https://your-llm-api.com",
        api_key="your-llm-api-key"
    )
    
    # Initialize translator
    translator = QueryTranslator(
        config_path=os.path.join(BASE_PROJECT_PATH, "src/config/schema.yaml"),
        vector_api_client=vector_api_client,
        llm_api_client=llm_api_client
    )
    
    # Example query
    query = "What's the average credit score for customers with late payments?"
    sql = await translator.translate_to_sql(query)
    print(f"Generated SQL: {sql}")

if __name__ == "__main__":
    asyncio.run(main())