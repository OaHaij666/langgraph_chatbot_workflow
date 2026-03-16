INTENT_KEYWORD_MAP = {
    "任务": ["帮我", "帮我做", "请帮我"],
    "聊天": ["你好", "在吗", "聊聊"],
}


def recognize_by_keyword(text: str) -> str | None:
    """
    基于关键词的意图识别。
    
    Args:
        text: 用户输入文本
    
    Returns:
        识别到的意图类型，未命中返回 None
    """
    # 按关键词长度降序排列，优先匹配更长的关键词
    sorted_items = sorted(INTENT_KEYWORD_MAP.items(), key=lambda x: max(len(kw) for kw in x[1]), reverse=True)
    
    for intent, keywords in sorted_items:
        for kw in keywords:
            if kw in text:
                return intent
    return None
