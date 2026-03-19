from typing import Generator


class BaseLLMClient:
    """LLM 客户端基类"""
    
    def chat(self, prompt: str, multimodal_inputs: list[dict] | None = None) -> str:
        raise NotImplementedError
    
    def stream_chat(self, prompt: str, multimodal_inputs: list[dict] | None = None) -> Generator[str, None, None]:
        raise NotImplementedError
