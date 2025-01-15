from typing import Dict, Any
from .llm_clients import OpenAILLMClient, AnthropicLLMClient, AzureOpenAIClient, LLMAPIClient

class LLMClientFactory:
    """Factory for creating LLM API clients"""
    
    @staticmethod
    def create_client(config: Dict[str, Any]) -> LLMAPIClient:
        """
        Create an LLM client based on configuration
        
        Args:
            config: Dictionary containing LLM configuration
                Required keys:
                - provider: str ("openai", "anthropic", or "azure")
                - api_key: str
                Provider-specific keys:
                - For OpenAI: model (optional)
                - For Anthropic: model (optional)
                - For Azure: endpoint, deployment_name
        
        Returns:
            LLMAPIClient: Configured LLM client
            
        Raises:
            ValueError: If configuration is invalid
        """
        provider = config.get("provider", "").lower()
        api_key = config.get("api_key")
        
        if not api_key:
            raise ValueError("API key is required")
            
        if provider == "openai":
            return OpenAILLMClient(
                api_key=api_key,
                model=config.get("model", "gpt-4"),
                organization=config.get("organization")
            )
            
        elif provider == "anthropic":
            return AnthropicLLMClient(
                api_key=api_key,
                model=config.get("model", "claude-3-opus-20240229")
            )
            
        elif provider == "azure":
            if not config.get("endpoint") or not config.get("deployment_name"):
                raise ValueError("Azure OpenAI requires endpoint and deployment_name")
                
            return AzureOpenAIClient(
                api_key=api_key,
                endpoint=config["endpoint"],
                deployment_name=config["deployment_name"],
                api_version=config.get("api_version", "2024-02-15-preview")
            )
            
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}") 