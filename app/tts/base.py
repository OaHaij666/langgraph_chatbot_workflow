from typing import Generator


class BaseTTSClient:
    """TTS 客户端基类"""
    
    def speak(self, text: str) -> None:
        raise NotImplementedError
    
    def stream_speak(self, text: str):
        raise NotImplementedError
