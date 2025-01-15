import asyncio
from typing import List, Dict, Optional, Union
import aiohttp
import numpy as np
from pinecone import Pinecone
import logging
from interfaces import VectorAPIClient, VectorData, VectorSearchResult, VectorDBError

# Get logger instance
logger = logging.getLogger(__name__)

# Example configurations for different services
# Vector Database API Cfrom typing import Listlient Implementations
from interfaces import VectorAPIClient, LLMAPIClient,VectorData,VectorSearchResult,LLMRequest,LLMResponse,VectorDBError
from llm_clients import OpenAILLMClient,AzureOpenAIClient
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union
import numpy as np
from dataclasses import dataclass
from pinecone import Pinecone


# Example implementation for a specific vector database (e.g., Pinecone)
class PineconeVectorAPIClient(VectorAPIClient):
    """
    Pinecone implementation of the VectorAPIClient interface using official SDK.
    """
    
    def __init__(self, api_key: str, environment: str, index_name: str):
        """
        Initialize Pinecone client using the official SDK.

        Args:
            api_key: Pinecone API key
            environment: Pinecone environment (e.g., 'us-east1-gcp')
            index_name: Name of the Pinecone index to use
        """
        logger.info(f"Initializing PineconeVectorAPIClient with index: {index_name}")
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=api_key)
        
        # Get the index
        try:
            self.index = self.pc.Index(index_name)
            logger.info("Successfully connected to existing Pinecone index")
        except Exception as e:
            # If index doesn't exist, create it
            if index_name not in self.pc.list_indexes().names():
                logger.info(f"Creating new Pinecone index: {index_name}")
                self.pc.create_index(
                    name=index_name,
                    dimension=768,  # dimension for 'all-mpnet-base-v2' model
                    metric='cosine',
                    spec=self.pc.ServerlessSpec(
                        cloud='aws',
                        region='us-west-2'
                    )
                )
                self.index = self.pc.Index(index_name)
                logger.info("Successfully created new Pinecone index")
            else:
                logger.error(f"Failed to initialize Pinecone index: {str(e)}")
                raise VectorDBError(f"Failed to initialize Pinecone index: {str(e)}")
        
        self.namespace = "banking-terms"
        logger.info("PineconeVectorAPIClient initialization completed")
    
    async def store_vectors(self, vector_data: List[VectorData]) -> bool:
        """Store vectors using Pinecone SDK"""
        try:
            logger.info(f"Storing {len(vector_data)} vectors in Pinecone")
            
            # Convert VectorData to Pinecone format
            vectors = []
            for data in vector_data:
                # Ensure metadata values are valid types
                cleaned_metadata = {}
                for key, value in data.metadata.items():
                    if value is not None:  # Only include non-null values
                        if isinstance(value, (str, int, float, bool)):
                            cleaned_metadata[key] = value
                        elif isinstance(value, list):
                            # Ensure all list elements are strings
                            cleaned_metadata[key] = [str(v) for v in value if v is not None]
                
                vectors.append({
                    'id': str(data.id),
                    'values': data.vector,
                    'metadata': cleaned_metadata
                })
            
            # Batch upsert to Pinecone
            self.index.upsert(
                vectors=vectors,
                namespace=self.namespace
            )
            logger.info(f"Successfully stored {len(vectors)} vectors in Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store vectors in Pinecone: {str(e)}", exc_info=True)
            raise VectorDBError(f"Failed to store vectors in Pinecone: {str(e)}")
    
    async def search_vectors(
        self,
        query_vector: Union[List[float], np.ndarray],
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter_metadata: Optional[Dict[str, any]] = None
    ) -> List[VectorSearchResult]:
        """Search vectors using Pinecone SDK"""
        try:
            # Convert numpy array to list if necessary
            if isinstance(query_vector, np.ndarray):
                query_vector = query_vector.tolist()
            
            logger.debug(f"Searching vectors with query of dimension {len(query_vector)}")
            
            # Perform query with lower threshold
            response = self.index.query(
                namespace=namespace or self.namespace,
                vector=query_vector,
                top_k=top_k,
                include_values=True,
                include_metadata=True,
                filter=filter_metadata,
                score_threshold=0.5  # Lower threshold for more matches
            )
            
            logger.debug(f"Vector search response: {response}")
            
            # Convert Pinecone response to VectorSearchResult objects
            if not response or "matches" not in response:
                logger.warning("No matches found in vector search")
                return []
                
            results = [
                VectorSearchResult(
                    id=match["id"],
                    score=match["score"],
                    metadata=match.get("metadata", {}),
                    vector=match.get("values")
                )
                for match in response["matches"]
            ]
            
            logger.info(f"Found {len(results)} similar terms with scores: {[r.score for r in results]}")
            return results
            
        except Exception as e:
            logger.error(f"Error in Pinecone search: {e}", exc_info=True)
            return []

    async def delete_vectors(
        self,
        vector_ids: List[str],
        namespace: Optional[str] = None
    ) -> bool:
        """Delete vectors using Pinecone SDK"""
        try:
            self.index.delete(
                ids=vector_ids,
                namespace=namespace or "banking-terms"
            )
            return True
            
        except Exception as e:
            raise VectorDBError(f"Failed to delete vectors from Pinecone: {str(e)}")

    # Implementation of other required methods...

 
class WeaviateVectorAPIClient(VectorAPIClient):
    """Implementation for Weaviate vector database"""
    
    def __init__(self, api_key: str, url: str):
        self.base_url = url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def store_vectors(self, vectors: List[VectorData]) -> bool:
        async with aiohttp.ClientSession() as session:
            try:
                # Weaviate batch import endpoint
                payload = {
                    "objects": [
                        {
                            "class": "Term",  # Your class name
                            "id": v.id,
                            "vector": v.vector,
                            "properties": v.metadata
                        }
                        for v in vectors
                    ]
                }
                
                async with session.post(
                    f"{self.base_url}/v1/batch/objects",
                    headers=self.headers,
                    json=payload
                ) as response:
                    return response.status == 200
            except Exception as e:
                print(f"Error storing vectors in Weaviate: {e}")
                return False

    async def delete_vectors(
        self,
        vector_ids: List[str],
        namespace: Optional[str] = None
    ) -> bool:
        """
        Delete vectors from Weaviate database.

        Args:
            vector_ids: List of vector IDs to delete
            namespace: Optional namespace/class name

        Returns:
            bool: True if deletion was successful

        Raises:
            VectorDBError: If there's an error communicating with Weaviate
        """
        async with aiohttp.ClientSession() as session:
            try:
                # Weaviate requires separate DELETE requests for each object
                for vector_id in vector_ids:
                    class_name = namespace or "Term"
                    async with session.delete(
                        f"{self.base_url}/v1/objects/{class_name}/{vector_id}",
                        headers=self.headers
                    ) as response:
                        if response.status not in (200, 204):
                            error_body = await response.text()
                            raise VectorDBError(
                                f"Failed to delete vector {vector_id}. Status: {response.status}, Body: {error_body}"
                            )
                return True
                
            except aiohttp.ClientError as e:
                raise VectorDBError("Failed to connect to Weaviate", e)
            except Exception as e:
                raise VectorDBError("Unexpected error deleting vectors", e)

# LLM API Client Implementations
class OpenAILLMClient(LLMAPIClient):
    """Implementation for OpenAI's API"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.model = model
    
    async def generate_completion(self, request: LLMRequest) -> LLMResponse:
        async with aiohttp.ClientSession() as session:
            try:
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a SQL expert."},
                        {"role": "user", "content": request.prompt}
                    ],
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens
                }
                
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return LLMResponse(
                            text=result["choices"][0]["message"]["content"],
                            metadata={
                                "model": self.model,
                                "finish_reason": result["choices"][0]["finish_reason"]
                            }
                        )
                    raise Exception(f"OpenAI API error: {response.status}")
            except Exception as e:
                print(f"Error generating completion with OpenAI: {e}")
                raise

class AzureOpenAIClient(LLMAPIClient):
    """Implementation for Azure OpenAI"""
    
    def __init__(
        self,
        api_key: str,
        endpoint: str,
        deployment_name: str
    ):
        self.base_url = f"{endpoint}/openai/deployments/{deployment_name}"
        self.headers = {
            "api-key": api_key,
            "Content-Type": "application/json"
        }
    
    async def generate_completion(self, request: LLMRequest) -> LLMResponse:
        async with aiohttp.ClientSession() as session:
            try:
                payload = {
                    "messages": [
                        {"role": "system", "content": "You are a SQL expert."},
                        {"role": "user", "content": request.prompt}
                    ],
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens
                }
                
                async with session.post(
                    f"{self.base_url}/chat/completions?api-version=2023-05-15",
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return LLMResponse(
                            text=result["choices"][0]["message"]["content"],
                            metadata={
                                "deployment": self.deployment_name,
                                "finish_reason": result["choices"][0]["finish_reason"]
                            }
                        )
                    raise Exception(f"Azure OpenAI API error: {response.status}")
            except Exception as e:
                print(f"Error generating completion with Azure OpenAI: {e}")
                raise

# Example usage showing how to initialize the clients
async def main():
    # Initialize vector database client (Pinecone example)
    vector_api_client = PineconeVectorAPIClient(
        api_key="your-pinecone-api-key",
        environment="us-east1-gcp",
        index_name="banking-terms"
    )
    
    # Alternative: Initialize Weaviate client
    # vector_api_client = WeaviateVectorAPIClient(
    #     api_key="your-weaviate-api-key",
    #     url="https://your-weaviate-instance.com"
    # )
    
    # Initialize LLM client (OpenAI example)
    llm_api_client = OpenAILLMClient(
        api_key="your-openai-api-key",
        model="gpt-4"
    )
    
    # Alternative: Initialize Azure OpenAI client
    # llm_api_client = AzureOpenAIClient(
    #     api_key="your-azure-openai-key",
    #     endpoint="https://your-resource.openai.azure.com",
 
# Configuration example
config = {
    "vector_db": {
        "provider": "pinecone",  # or "weaviate"
        "api_key": "your-vector-db-api-key",
        "environment": "us-east1-gcp",  # for Pinecone
        "index_name": "banking-terms"
    },
    "llm": {
        "provider": "openai",  # or "azure"
        "api_key": "your-llm-api-key",
        "model": "gpt-4",  # for OpenAI
        # For Azure:
        # "endpoint": "https://your-resource.openai.azure.com",
        # "deployment_name": "your-deployment"
    }
}

if __name__ == "__main__":
    asyncio.run(main())