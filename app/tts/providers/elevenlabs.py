import os
from typing import Generator
from app.tts.base import BaseTTSClient


class ElevenLabsClient(BaseTTSClient):
    """ElevenLabs TTS 客户端（高质量、可定制）"""
    
    def __init__(
        self,
        api_key: str = None,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model: str = "eleven_multilingual_v2"
    ):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = voice_id
        self.model = model
    
    def speak(self, text: str) -> bytes:
        import requests
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        data = {
            "text": text,
            "model_id": self.model
        }
        response = requests.post(url, json=data, headers=headers)
        return response.content
    
    def stream_speak(self, text: str):
        """流式返回音频字节"""
        import requests
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        data = {
            "text": text,
            "model_id": self.model
        }
        response = requests.post(url, json=data, headers=headers, stream=True)
        for chunk in response.iter_bytes(chunk_size=1024):
            yield chunk
