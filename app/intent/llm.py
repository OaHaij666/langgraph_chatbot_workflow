"""
LLM 兜底意图识别器

当关键词和 BERT 都无法识别时，使用 LLM 进行意图分类。
"""
from config import get_config
from app.intent.types import IntentResult, IntentCategory, RecognizerType


def _get_confidence(recognizer: str) -> float:
    """从配置文件获取指定识别器的置信度"""
    return get_config(f"intent.confidence.{recognizer}", 0.5)


def recognize_by_llm(text: str) -> IntentResult:
    """
    基于 LLM 的意图识别兜底方案
    
    Args:
        text: 用户输入文本
    
    Returns:
        IntentResult: 识别到的意图结果
    """
    # TODO: 实现 LLM 意图分类
    # 示例实现：
    # from app.llm import LLMClient
    # llm = LLMClient()
    # prompt = f"判断以下用户输入的意图类别（任务/聊天/其他）：{text}"
    # response = llm.chat(prompt)
    # category = parse_category(response)
    
    # 当前默认返回聊天意图
    return IntentResult(
        category=IntentCategory.CHAT,
        confidence=_get_confidence("fallback"),
        recognizer=RecognizerType.FALLBACK,
        raw_text=text,
    )
