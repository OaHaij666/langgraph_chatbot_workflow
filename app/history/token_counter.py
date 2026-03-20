"""
Token 计数工具

提供基于当前激活模型的 token 计数功能。
"""
from typing import Optional


def count_tokens(text: str) -> int:
    """
    计算文本的 token 数量
    
    使用当前激活的 tokenizer 进行计数。
    如果 tokenizer 未初始化，返回估算值（每 4 个字符约 1 个 token）。
    
    Args:
        text: 要计算的文本
    
    Returns:
        token 数量
    """
    if not text:
        return 0
    
    try:
        from app.llm.qlora_manager import QLoRASwitchManager
        
        manager = QLoRASwitchManager.get_instance()
        _, tokenizer, _ = manager.get_active_bundle()
        
        if tokenizer is None:
            return _estimate_tokens(text)
        
        actual_tokenizer = _get_actual_tokenizer(tokenizer)
        
        if hasattr(actual_tokenizer, 'encode'):
            return len(actual_tokenizer.encode(text, add_special_tokens=False))
        elif hasattr(actual_tokenizer, 'tokenize'):
            return len(actual_tokenizer.tokenize(text))
        else:
            return _estimate_tokens(text)
            
    except Exception:
        return _estimate_tokens(text)


def count_messages_tokens(messages: list[dict]) -> int:
    """
    计算消息列表的总 token 数量
    
    Args:
        messages: 消息列表，每条消息包含 role 和 content
    
    Returns:
        总 token 数量
    """
    total = 0
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content", "")
            role = msg.get("role", "")
            total += count_tokens(content)
            total += count_tokens(role)
            total += 4
    return total


def _get_actual_tokenizer(tokenizer):
    """
    获取实际的 tokenizer 对象
    
    某些 tokenizer 包装器（如 Unsloth 的）会有 .tokenizer 属性。
    """
    if hasattr(tokenizer, 'tokenizer'):
        return tokenizer.tokenizer
    return tokenizer


def _estimate_tokens(text: str) -> int:
    """
    估算 token 数量
    
    对于中文：约 1.5 个字符 = 1 token
    对于英文：约 4 个字符 = 1 token
    混合文本：取中间值，约 2.5 个字符 = 1 token
    """
    if not text:
        return 0
    
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    
    chinese_tokens = chinese_chars / 1.5
    other_tokens = other_chars / 4
    
    return int(chinese_tokens + other_tokens) + 1
