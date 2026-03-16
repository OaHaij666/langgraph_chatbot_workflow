from typing import Generator


class BaseLLMClient:
    """LLM 客户端基类"""
    
    def chat(self, prompt: str) -> str:
        raise NotImplementedError
    
    def stream_chat(self, prompt: str) -> Generator[str, None, None]:
        raise NotImplementedError
