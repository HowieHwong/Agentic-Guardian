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
                      temperature: Optional[float] = None) -> str:
        """
        Generate text using either OpenAI or externalAPI API.
        
        Args:
            prompt: The prompt to send to the model
            system_message: Optional system message
            response_format: Optional response format (e.g., {"type": "json_object"})
            temperature: Optional temperature override
            
        Returns:
            Generated text
        """
        if self.use_internal_inference:
            return self._generate_with_externalAPI(prompt, system_message)
        else:
            return self._generate_with_openai(prompt, system_message, response_format, temperature)

    def _generate_with_openai(self, 
                             prompt: str, 
                             system_message: str = "You are an AI assistant that responds to user requests.",
                             response_format: Optional[Dict[str, str]] = None,
                             temperature: Optional[float] = None) -> str:
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
        
        # Return the generated text
        return completion.choices[0].message.content

    def _generate_with_externalAPI(self,
                       prompt: str,
                       system_message: str = "You are an AI assistant that responds to user requests.") -> str:
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

            return response.choices[0].text.strip()

        except Exception as e:
            error_msg = f"externalAPI API error: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) 