import os
from typing import Generator
from app.tts.base import BaseTTSClient


class OpenAITTSClient(BaseTTSClient):
    """OpenAI TTS 客户端"""
    
    def __init__(self, api_key: str = None, model: str = "tts-1", voice: str = "alloy"):
        import openai
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.voice = voice
    
    def speak(self, text: str) -> None:
        response = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text
        )
        response.stream_to_file("temp_output.mp3")
    
    def stream_speak(self, text: str):
        """流式返回音频字节"""
        response = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            stream=True
        )
        for chunk in response.iter_bytes():
            yield chunk
