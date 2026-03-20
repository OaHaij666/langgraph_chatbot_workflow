"""
Intent 类型定义子模块

本模块定义意图识别相关的枚举和数据结构。
"""
from app.intent.types.enums import IntentCategory, RecognizerType
from app.intent.types.result import IntentResult

__all__ = [
    "IntentCategory",
    "RecognizerType", 
    "IntentResult",
]
