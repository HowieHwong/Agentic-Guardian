"""
Inference Module

This module provides unified inference capabilities for both OpenAI and externalAPI APIs.
Used by both generation and injection modules to reduce code duplication.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from pydantic import BaseModel, Field
import json
import requests
from loguru import logger
from openai import OpenAI

class OpenAIConfig(BaseModel):
    """Configuration for OpenAI API-based inference."""
    api_key: str
    api_base: Optional[str] = None
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 2048

class externalAPIConfig(BaseModel):
    """Configuration for externalAPI API-based inference."""
    api_url: str = "https://inference-3scale-apicast-production.apps.externalAPI.fmaas.res.ibm.com/mixtral-8x22b-instruct-a100/v1/chat/completions"
    api_key: str = ""
    model: str = "mistralai/mixtral-8x22B-instruct-v0.1"
    temperature: float = 1.0
    max_tokens: int = 2048

class InferenceManager:
    """
    Manages inference requests to OpenAI or externalAPI APIs.
    Provides a unified interface for both APIs.
    """
    def __init__(self, use_internal_inference: bool = False, 
                 openai_config: Optional[OpenAIConfig] = None,
                 externalAPI_config: Optional[externalAPIConfig] = None):
        """
        Initialize the inference manager.
        
        Args:
            use_internal_inference: Whether to use internal inference (externalAPI)
            openai_config: OpenAI API configuration
            externalAPI_config: externalAPI API configuration
        """
        self.use_internal_inference = use_internal_inference
        self.openai_config = openai_config
        self.externalAPI_config = externalAPI_config
        self.client = None
        # Initialize token tracking
        self.token_tracker = TokenTracker()
        
        # Setup OpenAI client if not using internal inference
        if not use_internal_inference:
            if not openai_config or not openai_config.api_key:
                raise ValueError("OpenAI API key is required for external inference")
            
            # Create OpenAI client instance
            self.client = OpenAI(api_key=openai_config.api_key)
            if openai_config.api_base:
                self.client.base_url = openai_config.api_base
                
            logger.info(f"Initialized OpenAI inference with model: {openai_config.model}")
        else:
            # Check externalAPI configuration
            if not externalAPI_config or not externalAPI_config.api_key:
                raise ValueError("externalAPI API key is required for internal inference")
                
            logger.info(f"Initialized externalAPI inference with model: {externalAPI_config.model}")

    def generate_text(self, 
                      prompt: str, 
                      system_message: str = "You are an AI assistant that responds to user requests.",
                      response_format: Optional[Dict[str, str]] = None,
                      temperature: Optional[float] = None,
                      return_usage: bool = False) -> Union[str, Tuple[str, Dict[str, int], float]]:
        """
        Generate text using either OpenAI or externalAPI API.
        
        Args:
            prompt: The prompt to send to the model
            system_message: Optional system message
            response_format: Optional response format (e.g., {"type": "json_object"})
            temperature: Optional temperature override
            
        Returns:
            If return_usage is False: Generated text (str)
            If return_usage is True: Tuple of (text, usage_dict, request_cost_usd)
        """
        if self.use_internal_inference:
            return self._generate_with_externalAPI(prompt, system_message, return_usage=return_usage)
        else:
            return self._generate_with_openai(
                prompt, system_message, response_format, temperature, return_usage=return_usage
            )

    def _generate_with_openai(self, 
                             prompt: str, 
                             system_message: str = "You are an AI assistant that responds to user requests.",
                             response_format: Optional[Dict[str, str]] = None,
                             temperature: Optional[float] = None,
                             return_usage: bool = False) -> Union[str, Tuple[str, Dict[str, int], float]]:
        """
        Generate text using OpenAI API.
        
        Args:
            prompt: The prompt to send to the model
            system_message: System message
            response_format: Optional response format
            temperature: Optional temperature override
            
        Returns:
            Generated text
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        # Create kwargs dict for the API call
        kwargs = {
            "model": self.openai_config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.openai_config.temperature,
            "max_tokens": self.openai_config.max_tokens
        }
        
        # Add response_format if provided
        if response_format:
            kwargs["response_format"] = response_format
        
        # Call the API
        completion = self.client.chat.completions.create(**kwargs)

        text = completion.choices[0].message.content

        # Extract usage if available
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        try:
            if hasattr(completion, "usage") and completion.usage is not None:
                usage["prompt_tokens"] = getattr(completion.usage, "prompt_tokens", 0) or 0
                usage["completion_tokens"] = getattr(completion.usage, "completion_tokens", 0) or 0
                usage["total_tokens"] = getattr(completion.usage, "total_tokens", 0) or (
                    usage["prompt_tokens"] + usage["completion_tokens"]
                )
        except Exception:
            # Keep default zeros if structure differs
            pass

        # Update tracker and compute cost
        self.token_tracker.add_usage(self.openai_config.model, usage["prompt_tokens"], usage["completion_tokens"])
        cost = self.token_tracker.compute_cost(self.openai_config.model, usage["prompt_tokens"], usage["completion_tokens"])

        if return_usage:
            return text, usage, cost
        return text

    def _generate_with_externalAPI(self,
                       prompt: str,
                       system_message: str = "You are an AI assistant that responds to user requests.",
                       return_usage: bool = False) -> Union[str, Tuple[str, Dict[str, int], float]]:
        """
        Generate text using externalAPI API via OpenAI client.
        
        Args:
            prompt: The prompt to send to the model
            system_message: System message
            
        Returns:
            Generated text
        """
        if not self.externalAPI_config or not self.externalAPI_config.api_key:
            raise ValueError("externalAPI API key is required for internal inference")

        # Initialize OpenAI client
        client = OpenAI(
            api_key=self.externalAPI_config.api_key,
            base_url=self.externalAPI_config.api_url,
            default_headers={"externalAPI_API_KEY": self.externalAPI_config.api_key}
        )

        try:
            # 构造完整 prompt
            full_prompt = f"System: {system_message}\n\nUser: {prompt}\n\nAssistant:"

            # 调用 completions API
            response = client.completions.create(
                model=self.externalAPI_config.model,
                prompt=full_prompt,
                temperature=self.externalAPI_config.temperature,
                max_tokens=self.externalAPI_config.max_tokens,
            )

            text = response.choices[0].text.strip()

            # Try to read usage if the backend returns it; otherwise leave zeros
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            try:
                if hasattr(response, "usage") and response.usage is not None:
                    usage["prompt_tokens"] = getattr(response.usage, "prompt_tokens", 0) or 0
                    usage["completion_tokens"] = getattr(response.usage, "completion_tokens", 0) or 0
                    usage["total_tokens"] = getattr(response.usage, "total_tokens", 0) or (
                        usage["prompt_tokens"] + usage["completion_tokens"]
                    )
            except Exception:
                pass

            # Update tracker using external model pricing if configured (often unknown, defaults to 0)
            self.token_tracker.add_usage(self.externalAPI_config.model, usage["prompt_tokens"], usage["completion_tokens"])
            cost = self.token_tracker.compute_cost(self.externalAPI_config.model, usage["prompt_tokens"], usage["completion_tokens"])

            if return_usage:
                return text, usage, cost
            return text

        except Exception as e:
            error_msg = f"externalAPI API error: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) 


class TokenPricing(BaseModel):
    """Pricing information per 1M tokens for a model."""
    input_per_million: float = 0.0
    output_per_million: float = 0.0


def _default_pricing_table() -> Dict[str, TokenPricing]:
    """Simple pricing table. Values are USD per 1M tokens. Extend as needed."""
    return {
        # Common OpenAI models (prices subject to change)
        "gpt-4o": TokenPricing(input_per_million=5.0, output_per_million=15.0),
        "gpt-4o-mini": TokenPricing(input_per_million=0.15, output_per_million=0.6),
        "gpt-3.5-turbo": TokenPricing(input_per_million=0.5, output_per_million=1.5),
    }


class TokenTracker:
    """Tracks token usage and estimated costs across requests."""

    def __init__(self) -> None:
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_requests: int = 0
        self.total_cost_usd: float = 0.0
        self.last_request_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.last_request_cost_usd: float = 0.0
        self.pricing_table: Dict[str, TokenPricing] = _default_pricing_table()

    def get_model_pricing(self, model: str) -> TokenPricing:
        # Match by substring to allow variants like gpt-4o-2024-xx
        for key, pricing in self.pricing_table.items():
            if key in (model or ""):
                return pricing
        return TokenPricing()

    def compute_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = self.get_model_pricing(model)
        input_cost = (prompt_tokens / 1_000_000) * pricing.input_per_million
        output_cost = (completion_tokens / 1_000_000) * pricing.output_per_million
        return round(input_cost + output_cost, 8)

    def add_usage(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
        self.total_prompt_tokens += prompt_tokens or 0
        self.total_completion_tokens += completion_tokens or 0
        self.total_requests += 1
        self.last_request_usage = {
            "prompt_tokens": prompt_tokens or 0,
            "completion_tokens": completion_tokens or 0,
            "total_tokens": total_tokens,
        }
        self.last_request_cost_usd = self.compute_cost(model, prompt_tokens or 0, completion_tokens or 0)
        self.total_cost_usd += self.last_request_cost_usd

    def summary(self) -> Dict[str, Any]:
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_requests": self.total_requests,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "last_request_usage": self.last_request_usage,
            "last_request_cost_usd": round(self.last_request_cost_usd, 6),
        }