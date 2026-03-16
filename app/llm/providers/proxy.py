import os
from typing import Generator
import requests
from app.llm.base import BaseLLMClient


class ProxyClient(BaseLLMClient):
    """代理客户端，支持自定义 API 端点"""
    
    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        model: str = None,
        headers: dict = None
    ):
        self.base_url = base_url or os.getenv("LLM_PROXY_URL")
        self.api_key = api_key or os.getenv("LLM_PROXY_KEY")
        self.model = model or os.getenv("LLM_PROXY_MODEL")
        self.headers = headers or {}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
    
    def _make_request(self, messages: list, stream: bool = False):
        """发起请求"""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream
        }
        response = requests.post(
            self.base_url,
            json=payload,
            headers=self.headers,
            stream=stream
        )
        return response
    
    def chat(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self._make_request(messages, stream=False)
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    def stream_chat(self, prompt: str) -> Generator[str, None, None]:
        messages = [{"role": "user", "content": prompt}]
        response = self._make_request(messages, stream=True)
        for line in response.iter_lines():
            if line:
                data = line.decode("utf-8")
                if data.startswith("data: "):
                    import json
                    chunk = json.loads(data[6:])
                    if "choices" in chunk and chunk["choices"]:
                        delta = chunk["choices"][0].get("delta", {})
                        if delta.get("content"):
                            yield delta["content"]
