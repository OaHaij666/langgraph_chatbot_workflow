from typing import Generator
from config import get_config
from app.tts.base import BaseTTSClient


class Qwen3TTSClient(BaseTTSClient):
    """
    Qwen3-TTS 客户端。
    阿里通义千问开源 TTS，0.6B 参数，量化后显存约 2GB，支持中文，流式输出。
    
    使用方式：
    1. pip install qwen-tts
    2. 下载模型：Qwen3-TTS-12Hz-0.6B-Base
    3. 配置 model_path 指向模型目录
    """
    
    def __init__(
        self,
        model_path: str = None,
        voice: str = "emilia"
    ):
        self.model_path = model_path or get_config("tts.model_path", "./tts_models")
        self.voice = voice
        self._model = None
    
    def _load_model(self):
        """加载模型（延迟加载）"""
        if self._model is None:
            from qwen_tts import TTSModel
            self._model = TTSModel(
                model_path=self.model_path,
                device="cuda"
            )
    
    def speak(self, text: str) -> bytes:
        """同步生成语音"""
        self._load_model()
        audio = self._model.generate(text, voice=self.voice)
        
        import io
        import scipy.io.wavfile as wavfile
        wav_io = io.BytesIO()
        wavfile.write(wav_io, 24000, audio.numpy())
        return wav_io.getvalue()
    
    def stream_speak(self, text: str) -> Generator[bytes, None, None]:
        """
        流式生成语音。
        Qwen3-TTS 支持流式输出，可实时返回音频 chunks。
        """
        self._load_model()
        
        for chunk in self._model.stream_generate(text, voice=self.voice):
            yield chunk.numpy().tobytes()
