"""
意图识别枚举定义
"""
from enum import Enum


class IntentCategory(Enum):
    """
    意图类别枚举
    
    Values:
        TASK: 任务类意图，用户需要执行特定任务
        CHAT: 聊天类意图，用户想进行日常对话
        OTHER: 其他类意图，无法明确分类
    """
    TASK = "任务"
    CHAT = "聊天"
    OTHER = "其他"


class RecognizerType(Enum):
    """
    识别器类型枚举
    
    Values:
        KEYWORD: 关键词匹配识别器
        BERT: BERT 模型分类识别器
        LLM: LLM 兜底识别器
        FALLBACK: 默认兜底
    """
    KEYWORD = "keyword"
    BERT = "bert"
    LLM = "llm"
    FALLBACK = "fallback"
