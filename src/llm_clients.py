from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import aiohttp
from dataclasses import dataclass
from interfaces import LLMAPIClient, LLMRequest, LLMResponse

class LLMError(Exception):
    """Custom exception for LLM API errors"""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error

class OpenAILLMClient(LLMAPIClient):
    """Implementation for OpenAI's API"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        organization: Optional[str] = None
    ):
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        if organization:
            self.headers["OpenAI-Organization"] = organization
        self.model = model
    
    async def generate_completion(self, request: LLMRequest) -> LLMResponse:
        """Generate completion using OpenAI's API"""
        async with aiohttp.ClientSession() as session:
            try:
                messages = [
                    {"role": "system", "content": "You are a SQL expert specializing in banking and financial queries."},
                    {"role": "user", "content": request.prompt}
                ]
                
                # Add any additional context as system messages
                if request.additional_context:
                    messages.insert(1, {
                        "role": "system",
                        "content": str(request.additional_context)
                    })
                
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "top_p": 1,
                    "frequency_penalty": 0,
                    "presence_penalty": 0
                }
                
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_body = await response.text()
                        raise LLMError(
                            f"OpenAI API error: {response.status} - {error_body}"
                        )
                    
                    result = await response.json()
                    return LLMResponse(
                        text=result["choices"][0]["message"]["content"],
                        metadata={
                            "model": self.model,
                            "finish_reason": result["choices"][0]["finish_reason"],
                            "usage": result.get("usage", {}),
                            "created": result.get("created")
                        }
                    )
                    
            except aiohttp.ClientError as e:
                raise LLMError("Failed to connect to OpenAI API", e)
            except Exception as e:
                raise LLMError(f"Unexpected error in OpenAI API call: {str(e)}", e)

class AnthropicLLMClient(LLMAPIClient):
    """Implementation for Anthropic's Claude API"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-opus-20240229"
    ):
        self.base_url = "https://api.anthropic.com/v1"
        self.headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        self.model = model
    
    async def generate_completion(self, request: LLMRequest) -> LLMResponse:
        """Generate completion using Anthropic's API"""
        async with aiohttp.ClientSession() as session:
            try:
                system_prompt = "You are Claude, an expert in SQL and banking domain knowledge."
                if request.additional_context:
                    system_prompt += f"\n\nAdditional context: {request.additional_context}"
                
                payload = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": request.prompt
                        }
                    ],
                    "system": system_prompt,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens
                }
                
                async with session.post(
                    f"{self.base_url}/messages",
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_body = await response.text()
                        raise LLMError(
                            f"Anthropic API error: {response.status} - {error_body}"
                        )
                    
                    result = await response.json()
                    return LLMResponse(
                        text=result["content"][0]["text"],
                        metadata={
                            "model": self.model,
                            "stop_reason": result.get("stop_reason"),
                            "usage": result.get("usage", {}),
                            "message_id": result.get("id")
                        }
                    )
                    
            except aiohttp.ClientError as e:
                raise LLMError("Failed to connect to Anthropic API", e)
            except Exception as e:
                raise LLMError(f"Unexpected error in Anthropic API call: {str(e)}", e)

class AzureOpenAIClient(LLMAPIClient):
    """Implementation for Azure OpenAI"""
    
    def __init__(
        self,
        api_key: str,
        endpoint: str,
        deployment_name: str,
        api_version: str = "2024-02-15-preview"
    ):
        self.base_url = f"{endpoint}/openai/deployments/{deployment_name}"
        self.headers = {
            "api-key": api_key,
            "Content-Type": "application/json"
        }
        self.deployment_name = deployment_name
        self.api_version = api_version
    
    async def generate_completion(self, request: LLMRequest) -> LLMResponse:
        """Generate completion using Azure OpenAI"""
        async with aiohttp.ClientSession() as session:
            try:
                messages = [
                    {"role": "system", "content": "You are a SQL expert specializing in banking and financial queries."},
                    {"role": "user", "content": request.prompt}
                ]
                
                if request.additional_context:
                    messages.insert(1, {
                        "role": "system",
                        "content": str(request.additional_context)
                    })
                
                payload = {
                    "messages": messages,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "top_p": 1,
                    "frequency_penalty": 0,
                    "presence_penalty": 0
                }
                
                async with session.post(
                    f"{self.base_url}/chat/completions?api-version={self.api_version}",
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_body = await response.text()
                        raise LLMError(
                            f"Azure OpenAI API error: {response.status} - {error_body}"
                        )
                    
                    result = await response.json()
                    return LLMResponse(
                        text=result["choices"][0]["message"]["content"],
                        metadata={
                            "deployment": self.deployment_name,
                            "finish_reason": result["choices"][0]["finish_reason"],
                            "usage": result.get("usage", {}),
                            "created": result.get("created")
                        }
                    )
                    
            except aiohttp.ClientError as e:
                raise LLMError("Failed to connect to Azure OpenAI API", e)
            except Exception as e:
                raise LLMError(f"Unexpected error in Azure OpenAI API call: {str(e)}", e) 