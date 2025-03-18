import json
import os
import requests

from typing import Dict, List, Optional, Any, Type, Union
from abc import ABC, abstractmethod
from pydantic import ValidationError

from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

from langchain_core.language_models.llms import LLM
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_community.llms.together import Together
# from langchain_community.llms. import ChatCustomLCClient

from settings_manager import WWSettingsManager

# Configuration constants
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7


class LLMProviderBase(ABC):
    """Base class for all LLM providers."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.cached_models = None
        self.llm_instance = None
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the provider."""
        pass
    
    @property
    @abstractmethod
    def default_endpoint(self) -> str:
        """Return the default endpoint for the provider."""
        pass

    @property
    def model_list_key(self) -> str:
        """Return the key for the model name in the provider's json response."""
        return "models"

    @property
    def model_key(self) -> str:
        """Return the key for the model name in the provider's json response."""
        return "name"
    
    @property
    def model_requires_api_key(self) -> bool:
        """Return whether the provider requires an API key."""
        return False
    
    @property
    def use_reverse_sort(self) -> bool:
        """Return whether to reverse the output of the model list."""
        return False

    @abstractmethod
    def get_llm_instance(self, overrides) -> Union[LLM, BaseChatModel]:
        """Returns a configured LLM instance."""
        pass
    
    def _do_models_request(self, url: str, headers: Dict[str, str] = None) -> List[str]:
        """Send a request to the provider to fetch available models."""
        return requests.get(url, headers=headers)
        
    
    def get_available_models(self, do_refresh: bool = False) -> List[str]:
        """Returns a list of available models from the provider."""
        if do_refresh or self.cached_models is None:
            try:
                # In a real implementation, this would make an API call to fetch models
                url = self.get_base_url()
                if url[-1] != "/":
                    url += "/"
                url += "models"
                response = self._do_models_request(url)
                if response.status_code == 200:
                    models_data = response.json()
                    self.cached_models = [model[self.model_key] for model in models_data.get(self.model_list_key, [])]
                    self.cached_models.sort(reverse=self.use_reverse_sort)
                else:
                    self.cached_models = []
            except Exception as e:
                print(f"Error fetching {self.provider_name} models: {e}")
                self.cached_models = []
        return self.cached_models

    
    def get_current_model(self) -> str:
        """Returns the currently configured model name."""
        return self.config.get("model", "")

    def get_default_endpoint(self) -> str:
        """Returns the default endpoint for the provider."""
        return self.default_endpoint

    def get_base_url(self) -> str:
        base_url = self.config.get("endpoint") or "http://192.168.178.45:1234"
        if not base_url.endswith("/v1"):  # Ensure it includes /v1
         base_url += "/v1"
        return base_url

    
    def get_api_key(self) -> str:
        """Returns the API key for the provider."""
        return self.config.get("api_key", "")
    
    def get_timeout(self, overrides) -> int:
        """Returns the timeout setting for the provider."""
        return overrides.get("timeout", self.config.get("timeout", 30))
    
    def get_context_window(self) -> int:
        """Returns the context window size for the current model."""
        # This would ideally be retrieved dynamically based on the model
        # For the base implementation, we'll return a default value
        return 4096

 
    def get_model_endpoint(self, overrides=None) -> str:
        """Returns the model endpoint for the provider."""
        url = overrides and overrides.get("endpoint") or self.config.get("endpoint", self.get_base_url())
        return url.replace("/v1/chat/completions", "/v1/models")

    def test_connection(self, overrides = None) -> bool:
        """Test the connection to the provider."""
        overrides["max_tokens"] = 1 # Minimal request for testing
        llm = self.get_llm_instance(overrides)
        if not llm:
            return False

        prompt = PromptTemplate(input_variables=[], template="")
        chain = prompt | llm | StrOutputParser()

        return True




class LMStudioProvider(LLMProviderBase):
    """LMStudio provider implementation."""
    
    @property
    def provider_name(self) -> str:
        return "LMStudio"
    
    @property
    def default_endpoint(self) -> str:
        return "http://localhost:1234"
    
    def get_llm_instance(self, overrides) -> BaseChatModel:
        if not self.llm_instance:
            # LMStudio uses the OpenAI-compatible API
            self.llm_instance = ChatOpenAI(
                openai_api_key=self.get_api_key() or "not-needed",
                openai_api_base=self.get_base_url() or "http://localhost:1234/v1",
                model_name=self.get_current_model() or "gemma-3-27b-it",
                temperature=self.config.get("temperature", DEFAULT_TEMPERATURE),
                max_tokens=self.config.get("max_tokens", DEFAULT_MAX_TOKENS),
                request_timeout=self.get_timeout(overrides)
            )
        return self.llm_instance
    
    def get_available_models(self, do_refresh: bool = False) -> List[str]:
        if do_refresh or self.cached_models is None:
            try:
                base_url = self.get_base_url() or "http://localhost:1234/v1"
                response = requests.get(f"{base_url}/models")
                if response.status_code == 200:
                    models_data = response.json()
                    self.cached_models = [model["id"] for model in models_data.get("data", [])]
                else:
                    self.cached_models = ["local-model"]
            except Exception as e:
                print(f"Error fetching LMStudio models: {e}")
                self.cached_models = ["local-model"]
        return self.cached_models


class CustomProvider(LLMProviderBase):
    """Custom LLM provider implementation for local network tools."""
    
    @property
    def provider_name(self) -> str:
        return "Custom"
    
    @property
    def default_endpoint(self) -> str:
        return "http://localhost:1234"
    
    def get_api_key(self):
        return super().get_api_key() or "not-needed"
    
    def get_llm_instance(self, overrides) -> BaseChatModel:
        if not self.llm_instance:
            self.config["endpoint"] = overrides.get("endpoint", self.get_base_url())
            self.config["api_key"] = overrides.get("api_key", self.get_api_key())
            self.config["model"] = overrides.get("model", self.get_current_model())
            self.llm_instance = ChatOpenAI( # most custom models are OpenAI compatible
                base_url=self.get_base_url(),
                api_key=self.get_api_key(),
                model_name=self.get_current_model() or "custom-model",
                temperature=self.config.get("temperature", DEFAULT_TEMPERATURE),
                max_tokens=self.config.get("max_tokens", DEFAULT_MAX_TOKENS),
                request_timeout=self.get_timeout(overrides)
            )
        return self.llm_instance
    
    def get_available_models(self, do_refresh: bool = False) -> List[str]:
        if do_refresh or self.cached_models is None:
            try:
                response = requests.get(
                    f"{self.get_base_url()}/v1/models",
                    headers={"Authorization": f"Bearer {self.get_api_key()}"}
                )
                if response.status_code == 200:
                    models_data = response.json()
                    self.cached_models = [model["id"] for model in models_data.get("data", [])]
                else:
                    if do_refresh:
                        raise Exception(f"Error fetching Custom models: {response.text}")
                    self.cached_models = ["custom-model"]
            except Exception as e:
                print(f"Error fetching custom models: {e}")
                if do_refresh:
                    raise e
                self.cached_models = ["custom-model"]
        return self.cached_models


class WW_Aggregator:
    """Main aggregator class for managing LLM providers."""
    
    def __init__(self):
        self._provider_cache = {}
        self._settings = None
    
    def create_provider(self, provider_name: str, config: Dict[str, Any] = None) -> Optional[LLMProviderBase]:
        """Create a new provider instance."""
        provider_class = self._get_provider_class(provider_name)
        if not provider_class:
            return None
        
        return provider_class(config)
    
    def get_provider(self, provider_name: str) -> Optional[LLMProviderBase]:
        """Get a provider instance by name."""
        if provider_name not in self._provider_cache:
            config = self._get_provider_config(provider_name)
            if not config:
                return None
            
            provider = config.get("provider")
            provider_class = self._get_provider_class(provider)
            if not provider_class:
                return None
            
            self._provider_cache[provider_name] = provider_class(config)
        
        return self._provider_cache[provider_name]
    
    def get_active_llms(self) -> List[str]:
        """Returns a list of all configured and cached LLMs."""
        return list(self._provider_cache.keys())
    
    def _get_provider_config(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """Get the configuration for a provider from settings."""
        settings = WWSettingsManager.get_llm_configs()
        if not settings:
            return None
        
        return settings.get(provider_name)
    
    def _get_provider_class(self, provider_name: str) -> Optional[Type[LLMProviderBase]]:
        """Get the provider class based on the provider name."""
        provider_map = {
            cls().provider_name: cls
            for cls in LLMProviderBase.__subclasses__()
        }
        return provider_map.get(provider_name)

import threading
import queue

class LLMAPIAggregator:
    """Main class for the LLM API Aggregator."""
    
    def __init__(self):
        self.aggregator = WW_Aggregator()
        self.interrupt_flag = threading.Event()
    
    def get_llm_providers(self) -> List[str]:
        """Dynamically returns a list of supported LLM provider names."""
        return [cls().provider_name for cls in LLMProviderBase.__subclasses__()]
    
    def send_prompt_to_llm(
        self, 
        final_prompt: str, 
        overrides: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Send a prompt to the active LLM and return the generated text."""
        overrides = overrides or {}
        settings = WWSettingsManager.get_llm_configs()
        
        # Determine which provider to use
        provider_name = overrides.get("provider") or WWSettingsManager.get_active_llm_name()
        if provider_name == "Local": # need to rename this to Default everywhere
            provider_name = WWSettingsManager.get_active_llm_name()
            overrides = {}
        if not provider_name:
            raise ValueError("No active LLM provider specified")
        
        # Get the provider instance
        provider = self.aggregator.get_provider(provider_name)
        if not provider:
            raise ValueError(f"Provider '{provider_name}' not found or not configured")
        
        # Get the LLM instance
        llm = provider.get_llm_instance(overrides)
        
        # Create messages format if conversation history is provided
        if conversation_history:
            from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
            
            messages = []
            for message in conversation_history:
                role = message.get("role", "").lower()
                content = message.get("content", "")
                
                if role == "system":
                    messages.append(SystemMessage(content=content))
                elif role == "user" or role == "human":
                    messages.append(HumanMessage(content=final_prompt))
                elif role == "assistant" or role == "ai":
                    messages.append(AIMessage(content=content))
            
        request_payload = {
                      "model": provider.get_current_model(),  # Ensure this matches LM Studio's model
                      "messages": messages,  # Ensure it's in OpenAI format
                      "temperature": provider.config.get("temperature", DEFAULT_TEMPERATURE),
                      "max_tokens": provider.config.get("max_tokens", DEFAULT_MAX_TOKENS),
                      }

        logging.debug(f"Sending request to LM Studio:\n{json.dumps(request_payload, indent=2)}")


            # Add the current prompt
        messages = [
                        {"role": "system", "content": "Always answer in rhymes. Today is Thursday"},
                        {"role": "user", "content": "What day is it today?"}
                       ]

            
            # Generate response
        response = llm.invoke(messages)
        return response.content
        else:
        # Simple prompt-based invocation
    return llm.invoke(final_prompt).content

    def stream_prompt_to_llm(
        self, 
        final_prompt: str, 
        overrides: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ):
        """Stream a prompt to the active LLM and yield the generated text."""
        overrides = overrides or {}
        settings = WWSettingsManager.get_llm_configs()
        
        # Determine which provider to use
        provider_name = overrides.get("provider") or WWSettingsManager.get_active_llm_name()
        if provider_name == "Local": # need to rename this to Default everywhere
            provider_name = WWSettingsManager.get_active_llm_name()
            overrides = {}
        if not provider_name:
            raise ValueError("No active LLM provider specified")
        
        # Get the provider instance
        provider = self.aggregator.get_provider(provider_name)
        if not provider:
            raise ValueError(f"Provider '{provider_name}' not found or not configured")
        
        # Get the LLM instance
        llm = provider.get_llm_instance(overrides)
        
        # Create messages format if conversation history is provided
        if conversation_history:
            from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
            
            messages = []
            for message in conversation_history:
                role = message.get("role", "").lower()
                content = message.get("content", "")
                
                if role == "system":
                    messages.append(SystemMessage(content=content))
                elif role == "user" or role == "human":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant" or role == "ai":
                    messages.append(AIMessage(content=content))
            
            # Add the current prompt
            messages.append(HumanMessage(content=final_prompt))
            
            # Generate response
            for chunk in llm.stream(messages):
                if self.interrupt_flag.is_set():
                    break
                yield chunk.content
        else:
            # Simple prompt-based invocation
            for chunk in llm.stream(final_prompt):
                if self.interrupt_flag.is_set():
                    break
                yield chunk.content

    def interrupt(self):
        """Interrupt the streaming process."""
        self.interrupt_flag.set()

WWApiAggregator = LLMAPIAggregator()


def main():
    """Example usage of the LLM API Aggregator."""
    aggregator = LLMAPIAggregator()
    
    overrides = {
        "api_key": "AIFakeKey123",
    }

    try:
        p = aggregator.aggregator.create_provider("Gemini")
        p.get_default_endpoint(overrides)
        p.get_base_url()
    except ValidationError as exc:
        print(exc.errors())
        #> 'missing'

    # Get list of supported providers
    providers = aggregator.get_llm_providers()
    print(f"Supported providers: {providers}")
    
    # Example prompt
    try:
        response = aggregator.send_prompt_to_llm("Hello, tell me a short story about a robot.")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
