from app.intent.keyword import recognize_by_keyword
from app.intent.bert import recognize_by_bert
from app.intent.llm import recognize_by_llm


# 意图类型：
# - 任务: 用户需要执行特定任务（如"帮我查天气"）
# - 聊天: 用户想日常闲聊
# - 其他: 以上都不是，走默认分支


def recognize(text: str) -> str:
    """
    意图识别主函数，按优先级调用三层识别器。
    优先级：关键词匹配 > BERT > LLM 兜底
    
    Args:
        text: 用户输入文本
    
    Returns:
        意图类型：任务 / 聊天 / 其他
    """
    # 第一层：关键词匹配
    intent = recognize_by_keyword(text)
    if intent:
        return intent

    # 第二层：BERT 分类
    intent = recognize_by_bert(text)
    if intent:
        return intent

    # 第三层：LLM 兜底
    return recognize_by_llm(text)
