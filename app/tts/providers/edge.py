import os
from typing import Generator
import asyncio
from app.tts.base import BaseTTSClient


class EdgeTTSClient(BaseTTSClient):
    """Edge TTS 客户端（免费、快速）"""
    
    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural"):
        self.voice = voice
    
    async def _speak_async(self, text: str) -> bytes:
        import edge_tts
        communicate = edge_tts.Communicate(text, self.voice)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return audio_data
    
    def speak(self, text: str) -> bytes:
        return asyncio.run(self._speak_async(text))
    
    def stream_speak(self, text: str):
        """流式返回音频字节"""
        async def gen():
            import edge_tts
            communicate = edge_tts.Communicate(text, self.voice)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
        
        return gen()
