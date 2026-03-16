import os
from typing import Generator
from app.llm.base import BaseLLMClient


class OllamaClient(BaseLLMClient):
    """Ollama 本地模型客户端"""
    
    def __init__(self, base_url: str = None, model: str = "llama2"):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model
    
    def chat(self, prompt: str) -> str:
        import requests
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False}
        )
        return response.json().get("response", "")
    
    def stream_chat(self, prompt: str) -> Generator[str, None, None]:
        import requests
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": True},
            stream=True
        )
        for line in response.iter_lines():
            if line:
                data = line.decode("utf-8")
                if "response" in data:
                    import json
                    chunk = json.loads(data)
                    if chunk.get("response"):
                        yield chunk["response"]
