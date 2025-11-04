"""
Ollama LLM Integration Module
Handles interaction with Ollama Cloud API
"""
from typing import Optional, Dict, List
import requests
from src.config.settings import Config
from src.config.logger import app_logger

class OllamaLLM:
    """Ollama Cloud LLM client"""
    
    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        model: str = None
    ):
        """
        Initialize Ollama client
        
        Args:
            base_url: Ollama API base URL
            api_key: API key (if required)
            model: Model name to use
        """
        self.base_url = (base_url or Config.OLLAMA_BASE_URL).rstrip('/')
        self.api_key = api_key or Config.OLLAMA_API_KEY
        self.model = model or Config.OLLAMA_MODEL
        
        app_logger.info(f"Initialized Ollama client: {self.base_url}, model: {self.model}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """
        Generate a response from Ollama
        
        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        try:
            # Prepare the request
            url = f"{self.base_url}/api/generate"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            # Add additional parameters
            payload["options"].update(kwargs)
            
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Make the request
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get("response", "")
            
            app_logger.info(f"Generated response: {len(generated_text)} characters")
            
            return generated_text
            
        except requests.exceptions.RequestException as e:
            app_logger.error(f"Ollama API request failed: {e}")
            raise
        except Exception as e:
            app_logger.error(f"Error generating response: {e}")
            raise
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """
        Chat completion with conversation history
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        try:
            url = f"{self.base_url}/api/chat"
            
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            payload["options"].update(kwargs)
            
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get("message", {}).get("content", "")
            
            return generated_text
            
        except requests.exceptions.RequestException as e:
            app_logger.error(f"Ollama chat API request failed: {e}")
            raise
        except Exception as e:
            app_logger.error(f"Error in chat: {e}")
            raise

# Singleton instance
_llm_instance = None

def get_ollama_llm() -> OllamaLLM:
    """
    Get or create singleton Ollama LLM instance
    
    Returns:
        OllamaLLM instance
    """
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = OllamaLLM()
    return _llm_instance
