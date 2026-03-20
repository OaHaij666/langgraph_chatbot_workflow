"""
意图识别主模块

本模块实现三层意图识别策略:
1. 关键词匹配: 快速、高置信度
2. BERT 分类: 中等置信度（待实现）
3. LLM 兜底: 低置信度或默认

使用方式:
    from app.intent.recognize import recognize
    
    result = recognize("帮我查天气")
    print(result.category)      # IntentCategory.TASK
    print(result.confidence)    # 0.95
    print(result.recognizer)    # RecognizerType.KEYWORD
"""
from typing import Optional
from app.intent.types import IntentResult, IntentCategory, RecognizerType
from app.intent.keyword import recognize_by_keyword
from app.intent.bert import recognize_by_bert
from app.intent.llm import recognize_by_llm


def recognize(text: str) -> IntentResult:
    """
    意图识别主函数
    
    按优先级调用三层识别器:
    1. 关键词匹配 > BERT > LLM 兜底
    
    Args:
        text: 用户输入文本
    
    Returns:
        IntentResult: 意图识别结果，包含类别、置信度、识别器类型等
    """
    # 第一层：关键词匹配（高置信度）
    result = recognize_by_keyword(text)
    if result is not None:
        return result

    # 第二层：BERT 分类（中等置信度）
    result = recognize_by_bert(text)
    if result is not None:
        return result

    # 第三层：LLM 兜底（低置信度或默认）
    return recognize_by_llm(text)


def recognize_with_fallback(text: str, fallback: IntentCategory = IntentCategory.CHAT) -> IntentResult:
    """
    带指定兜底类别的意图识别
    
    Args:
        text: 用户输入文本
        fallback: 兜底时的默认类别
    
    Returns:
        IntentResult: 意图识别结果
    """
    result = recognize(text)
    if result.category == IntentCategory.OTHER:
        return IntentResult(
            category=fallback,
            confidence=0.5,
            recognizer=RecognizerType.FALLBACK,
            raw_text=text,
        )
    return result
