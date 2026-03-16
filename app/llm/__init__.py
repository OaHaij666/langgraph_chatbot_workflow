import os
from typing import Generator
from config import get_config
from app.llm.base import BaseLLMClient
from app.llm.providers import openai, anthropic, ollama, proxy, unsloth


class LLMClient(BaseLLMClient):
    """
    LLM 客户端，根据配置选择对应的 provider。
    """
    
    def __init__(self, provider: str = None):
        provider = provider or get_config("llm.provider", "openai")
        self._client = self._create_client(provider)
    
    def _create_client(self, provider: str) -> BaseLLMClient:
        if provider == "openai":
            return openai.OpenAIClient(
                api_key=get_config("llm.api_key") or os.getenv("OPENAI_API_KEY"),
                model=get_config("llm.model", "gpt-3.5-turbo")
            )
        elif provider == "anthropic":
            return anthropic.AnthropicClient(
                api_key=get_config("llm.api_key") or os.getenv("ANTHROPIC_API_KEY"),
                model=get_config("llm.model", "claude-3-haiku-20240307")
            )
        elif provider == "ollama":
            return ollama.OllamaClient(
                base_url=get_config("llm.base_url") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                model=get_config("llm.model", "llama2")
            )
        elif provider == "proxy":
            return proxy.ProxyClient(
                base_url=get_config("llm.base_url") or os.getenv("LLM_PROXY_URL"),
                api_key=get_config("llm.api_key") or os.getenv("LLM_PROXY_KEY"),
                model=get_config("llm.model")
            )
        elif provider == "unsloth":
            return unsloth.UnslothClient(
                model_path=get_config("llm.model_path", "./models")
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    
    def chat(self, prompt: str) -> str:
        return self._client.chat(prompt)
    
    def stream_chat(self, prompt: str) -> Generator[str, None, None]:
        return self._client.stream_chat(prompt)
