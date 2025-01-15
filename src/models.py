from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass

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

@dataclass
class LLMRequest:
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 2048
    additional_context: Dict[str, Any] = None

@dataclass
class LLMResponse:
    text: str
    metadata: Dict[str, Any]

class VectorAPIClient(ABC):
    @abstractmethod
    async def store_vectors(self, vectors: List[VectorData]) -> bool:
        pass
    
    @abstractmethod
    async def search_vectors(self, query_vector: List[float], top_k: int = 10) -> List[VectorSearchResult]:
        pass

class LLMAPIClient(ABC):
    @abstractmethod
    async def generate_completion(self, request: LLMRequest) -> LLMResponse:
        pass 