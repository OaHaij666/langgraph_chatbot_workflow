"""
意图识别结果数据类
"""
from dataclasses import dataclass
from typing import Optional
from app.intent.types.enums import IntentCategory, RecognizerType


@dataclass
class IntentResult:
    """
    意图识别结果
    
    Attributes:
        category: 意图类别
        confidence: 置信度 (0.0 - 1.0)
        recognizer: 识别器类型
        raw_text: 原始输入文本
    """
    category: IntentCategory
    confidence: float
    recognizer: RecognizerType
    raw_text: Optional[str] = None
    
    def __post_init__(self):
        """验证置信度范围"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"置信度必须在 0.0-1.0 之间，当前值: {self.confidence}")
    
    @property
    def branch_name(self) -> str:
        """获取对应的分支名称"""
        return self.category.value
    
    def should_confirm(self) -> bool:
        """
        是否需要用户确认
        
        当置信度低于 0.7 且不是关键词匹配时，建议用户确认意图
        
        Returns:
            是否需要确认
        """
        return self.confidence < 0.7 and self.recognizer != RecognizerType.KEYWORD
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "category": self.category.value,
            "confidence": self.confidence,
            "recognizer": self.recognizer.value,
            "raw_text": self.raw_text,
        }
