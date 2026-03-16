import os
import subprocess
from typing import Generator
from config import get_config
from app.tts.base import BaseTTSClient


class PiperClient(BaseTTSClient):
    """
    Piper TTS 客户端。
    轻量级本地 TTS，基于 ONNX，显存占用低（约 2GB），支持中文。
    
    使用方式：
    1. 安装：pip install piper-tts
    2. 下载中文模型：zh_CN-huayan-medium.onnx
    3. 配置 model_path 指向模型文件
    """
    
    def __init__(
        self,
        model_path: str = None,
        voice: str = "zh_CN-huayan-medium"
    ):
        self.model_path = model_path or get_config("tts.model_path", "./tts_models")
        self.voice = voice
        
        self._model_path = os.path.join(self.model_path, f"{self.voice}.onnx")
        self._onnx_path = os.path.join(self.model_path, f"{self.voice}.onnx.json")
    
    def _ensure_model(self):
        """检查模型文件是否存在"""
        if not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"Piper 模型未找到: {self._model_path}\n"
                f"请下载中文模型并放入 tts_models 目录\n"
                f"模型下载: https://github.com/rhasspy/piper"
            )
    
    def speak(self, text: str) -> bytes:
        """同步生成语音"""
        self._ensure_model()
        
        result = subprocess.run(
            ["piper", "--model", self._model_path, "--output-file", "-"],
            input=text.encode("utf-8"),
            capture_output=True
        )
        
        return result.stdout
    
    def stream_speak(self, text: str) -> Generator[bytes, None, None]:
        """
        流式生成语音。
        Piper 本身不支持真正的流式输出，这里返回完整音频。
        如需流式播放，可配合音频流播放器使用。
        """
        self._ensure_model()
        
        result = subprocess.run(
            ["piper", "--model", self._model_path, "--output-file", "-"],
            input=text.encode("utf-8"),
            capture_output=True
        )
        
        yield result.stdout
