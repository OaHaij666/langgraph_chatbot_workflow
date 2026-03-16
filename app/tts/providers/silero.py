from typing import Generator
import torch
from app.tts.base import BaseTTSClient


class SileroClient(BaseTTSClient):
    """Silero TTS 客户端（本地运行、免费）"""
    
    def __init__(self, language: str = "ru", speaker: str = "aidar"):
        self.language = language
        self.speaker = speaker
        self._model = None
    
    def _load_model(self):
        if self._model is None:
            import silero_tts
            self._model, self._example_text = silero_tts(
                language=self.language,
                speaker=self.speaker
            )
    
    def speak(self, text: str) -> bytes:
        self._load_model()
        audio = self._model.apply_tts(text=text)
        import io
        import numpy as np
        wav_io = io.BytesIO()
        import scipy.io.wavfile as wavfile
        wavfile.write(wav_io, 24000, audio.numpy())
        return wav_io.getvalue()
    
    def stream_speak(self, text: str):
        """流式返回音频（需要累积一段文本）"""
        self._load_model()
        audio = self._model.apply_tts(text=text)
        import io
        import numpy as np
        wav_io = io.BytesIO()
        import scipy.io.wavfile as wavfile
        wavfile.write(wav_io, 24000, audio.numpy())
        yield wav_io.getvalue()
