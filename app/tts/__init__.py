import os
from typing import Generator
from config import get_config
from app.tts.base import BaseTTSClient
from app.tts.providers import openai, edge, elevenlabs, silero, piper, qwen3


class TTSClient(BaseTTSClient):
    """
    TTS 客户端，根据配置选择对应的 provider。
    """
    
    def __init__(self, provider: str = None):
        provider = provider or get_config("tts.provider", "edge")
        self._client = self._create_client(provider)
    
    def _create_client(self, provider: str) -> BaseTTSClient:
        if provider == "openai":
            return openai.OpenAITTSClient(
                api_key=get_config("tts.api_key") or os.getenv("OPENAI_API_KEY"),
                model=get_config("tts.model", "tts-1"),
                voice=get_config("tts.voice", "alloy")
            )
        elif provider == "edge":
            return edge.EdgeTTSClient(
                voice=get_config("tts.voice", "zh-CN-XiaoxiaoNeural")
            )
        elif provider == "elevenlabs":
            return elevenlabs.ElevenLabsClient(
                api_key=get_config("tts.api_key") or os.getenv("ELEVENLABS_API_KEY"),
                voice_id=get_config("tts.voice_id", "21m00Tcm4TlvDq8ikWAM"),
                model=get_config("tts.model", "eleven_multilingual_v2")
            )
        elif provider == "silero":
            return silero.SileroClient(
                language=get_config("tts.language", "ru"),
                speaker=get_config("tts.speaker", "aidar")
            )
        elif provider == "piper":
            return piper.PiperClient(
                model_path=get_config("tts.model_path", "./tts_models"),
                voice=get_config("tts.voice", "zh_CN-huayan-medium")
            )
        elif provider == "qwen3":
            return qwen3.Qwen3TTSClient(
                model_path=get_config("tts.model_path", "./tts_models"),
                voice=get_config("tts.voice", "emilia")
            )
        else:
            raise ValueError(f"Unknown TTS provider: {provider}")
    
    def speak(self, text: str):
        return self._client.speak(text)
    
    def stream_speak(self, text: str):
        return self._client.stream_speak(text)
