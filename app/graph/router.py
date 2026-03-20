"""
意图路由模块

根据意图识别结果分派到对应的处理分支。
"""
from app.intent.recognize import recognize
from app.intent.types import IntentCategory
from app.branches.task import run_task_branch
from app.branches.chat import run_chat_branch
from app.branches.default import run_default_branch


def process(text: str):
    """
    主入口函数，负责分派任务到对应的分支。
    
    Args:
        text: 用户输入的文本
    
    Returns:
        流式响应生成器
    """
    # 意图识别
    intent_result = recognize(text)

    # 根据意图类别分派到对应分支
    if intent_result.category == IntentCategory.TASK:
        return run_task_branch(text)
    elif intent_result.category == IntentCategory.CHAT:
        return run_chat_branch(text)
    else:
        return run_default_branch(text)
