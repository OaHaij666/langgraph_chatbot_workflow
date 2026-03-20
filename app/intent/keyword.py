"""
关键词匹配意图识别器

基于预定义关键词列表进行意图匹配，速度快、置信度高。
关键词配置位于 config/config.json 的 intent.keywords 字段。
"""
from typing import Optional
from config import get_config
from app.intent.types import IntentResult, IntentCategory, RecognizerType


def _get_keywords_config() -> dict[str, list[str]]:
    """
    从配置文件获取关键词配置
    
    Returns:
        关键词配置字典，格式为 {"task": [...], "chat": [...]}
    """
    return get_config("intent.keywords", {
        "task": ["帮我", "帮我做", "请帮我", "执行", "完成", "处理"],
        "chat": ["你好", "在吗", "聊聊", "聊天", "闲聊"],
    })


def _get_confidence(recognizer: str) -> float:
    """
    从配置文件获取指定识别器的置信度
    
    Args:
        recognizer: 识别器类型名称
    
    Returns:
        置信度值
    """
    return get_config(f"intent.confidence.{recognizer}", 0.95)


def recognize_by_keyword(text: str) -> Optional[IntentResult]:
    """
    基于关键词的意图识别
    
    按关键词长度降序排列，优先匹配更长的关键词。
    
    Args:
        text: 用户输入文本
    
    Returns:
        IntentResult: 识别到的意图结果，未命中返回 None
    """
    keywords_config = _get_keywords_config()
    confidence = _get_confidence("keyword")
    
    category_mapping = {
        "task": IntentCategory.TASK,
        "chat": IntentCategory.CHAT,
    }
    
    sorted_items = sorted(
        keywords_config.items(),
        key=lambda x: max(len(kw) for kw in x[1]) if x[1] else 0,
        reverse=True
    )
    
    for key, keywords in sorted_items:
        category = category_mapping.get(key)
        if category is None:
            continue
        
        for keyword in keywords:
            if keyword in text:
                return IntentResult(
                    category=category,
                    confidence=confidence,
                    recognizer=RecognizerType.KEYWORD,
                    raw_text=text,
                )
    
    return None
