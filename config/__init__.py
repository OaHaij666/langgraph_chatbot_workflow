import os
import json
from pathlib import Path


CONFIG_DIR = Path(__file__).parent
CONFIG_FILE = CONFIG_DIR / "config.json"


class Config:
    """配置管理类"""
    
    def __init__(self):
        self._config = self._load_config()
    
    def _load_config(self) -> dict:
        """加载配置文件"""
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return self._default_config()
    
    def _default_config(self) -> dict:
        """默认配置"""
        return {
            "llm": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "api_key": "",
                "base_url": "",
                "base_model_path": "./models/Qwen3___5-4B",
                "adapter_path": "./models/qwen3.5-4b-lora-default",
                "adapters": {
                    "default": {
                        "id": "default",
                        "name": "general-assistant",
                        "path": "./models/qwen3.5-4b-lora-default",
                        "params": {
                            "lora_scale": 0.6
                        }
                    }
                },
                "max_resident_adapters": 4,
                "switch_history_size": 30,
                "enable_multimodal": False
            },
            "rag": {
                "top_k": 3
            },
            "tts": {
                "provider": "default"
            },
            "intent": {
                "keywords": {
                    "task": ["帮我", "帮我做", "请帮我", "执行", "完成", "处理"],
                    "chat": ["你好", "在吗", "聊聊", "聊天", "闲聊"]
                },
                "confidence": {
                    "keyword": 0.95,
                    "bert": 0.8,
                    "llm": 0.6,
                    "fallback": 0.5
                },
                "confirm_threshold": 0.7
            }
        }
    
    def get(self, key: str, default=None):
        """获取配置，支持点号分隔的键，如 'llm.provider'"""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default
    
    def set(self, key: str, value) -> None:
        """设置配置"""
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def save(self) -> None:
        """保存配置到文件"""
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=4, ensure_ascii=False)


_config = Config()


def get_config(key: str = None, default=None):
    """获取配置"""
    if key is None:
        return _config._config
    return _config.get(key, default)


def set_config(key: str, value) -> None:
    """设置配置"""
    _config.set(key, value)


def save_config() -> None:
    """保存配置"""
    _config.save()
