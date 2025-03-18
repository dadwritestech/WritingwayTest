import json
import logging
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

from settings_manager import WWSettingsManager

# Configure logging
logging.basicConfig(level=logging.DEBUG)

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
        pass

    @property
    @abstractmethod
    def default_endpoint(self) -> str:
        pass

    def get_base_url(self) -> str:
        base_url = self.config.get("endpoint") or "http://localhost:1234"
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        return base_url

    def get_current_model(self) -> str:
        return self.config.get("model", "gemma-3-27b-it")

    def get_timeout(self, overrides) -> int:
        return overrides.get("timeout", self.config.get("timeout", 30))

    @abstractmethod
    def get_llm_instance(self, overrides) -> Union[LLM, BaseChatModel]:
        pass


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
            self.llm_instance = ChatOpenAI(
                openai_api_key="not-needed",  # Add this to bypass API key check
                openai_api_base=self.get_base_url(),
                model_name=self.get_current_model(),
                temperature=self.config.get("temperature", DEFAULT_TEMPERATURE),
                max_tokens=self.config.get("max_tokens", DEFAULT_MAX_TOKENS),
                request_timeout=180,  # Increased timeout to 180 seconds
            )
        return self.llm_instance


class WW_Aggregator:
    """Main aggregator class for managing LLM providers."""

    def __init__(self):
        self._provider_cache = {}

    def create_provider(
        self, provider_name: str, config: Dict[str, Any] = None
    ) -> Optional[LLMProviderBase]:
        provider_class = self._get_provider_class(provider_name)
        return provider_class(config) if provider_class else None

    def get_provider(self, provider_name: str) -> Optional[LLMProviderBase]:
        if provider_name not in self._provider_cache:
            config = WWSettingsManager.get_llm_configs().get(provider_name)
            if config:
                provider_class = self._get_provider_class(provider_name)
                if provider_class:
                    self._provider_cache[provider_name] = provider_class(config)
        return self._provider_cache.get(provider_name)

    def _get_provider_class(
        self, provider_name: str
    ) -> Optional[Type[LLMProviderBase]]:
        provider_map = {cls().provider_name: cls for cls in LLMProviderBase.__subclasses__()}
        return provider_map.get(provider_name)


class LLMAPIAggregator:
    """Main class for the LLM API Aggregator."""

    def __init__(self):
        self.aggregator = WW_Aggregator()

    def send_prompt_to_llm(
        self,
        final_prompt: str,
        overrides: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        overrides = overrides or {}
        provider_name = overrides.get("provider") or WWSettingsManager.get_active_llm_name()
        provider = self.aggregator.get_provider(provider_name)
        if not provider:
            raise ValueError(f"Provider '{provider_name}' not found or not configured")

        llm = provider.get_llm_instance(overrides)

        # Import message classes here to ensure they're available for later use
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        messages = []
        if conversation_history:
            for message in conversation_history:
                role = message["role"].lower()
                content = message["content"]
                if role == "system":
                    messages.append(SystemMessage(content=content))
                elif role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=final_prompt))

        # Convert messages into JSON format
        formatted_messages = [
            {"role": msg.type, "content": msg.content} for msg in messages
        ]

        request_payload = {
            "model": provider.get_current_model(),
            "messages": formatted_messages,
            "temperature": provider.config.get("temperature", DEFAULT_TEMPERATURE),
            "max_tokens": provider.config.get("max_tokens", DEFAULT_MAX_TOKENS),
        }

        logging.debug(
            f"Sending request to LM Studio:\n{json.dumps(request_payload, indent=2)}"
        )

        # Invoke model and process response
        response_message = llm.invoke(formatted_messages)

        if isinstance(response_message, AIMessage):
            response_text = response_message.content
        else:
            response_text = str(response_message)

        logging.debug(f"Processed response from LM Studio: {response_text}")
        return response_text


WWApiAggregator = LLMAPIAggregator()


def main():
    """Example usage of the LLM API Aggregator."""
    aggregator = LLMAPIAggregator()
    try:
        response = aggregator.send_prompt_to_llm(
            "Hello, tell me a short story about a robot."
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
