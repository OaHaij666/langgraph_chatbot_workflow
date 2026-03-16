from app.intent.recognize import recognize
from app.branches.task import run_task_branch
from app.branches.chat import run_chat_branch
from app.branches.default import run_default_branch
from app.history import add_history


def process(text: str):
    """
    主入口函数，负责分派任务到对应的分支。
    
    Args:
        text: 用户输入的文本
    
    Returns:
        流式响应生成器
    """
    # 意图识别
    intent = recognize(text)

    # 根据意图分派到对应分支
    if intent == "任务":
        return run_task_branch(text)
    elif intent == "聊天":
        return run_chat_branch(text)
    else:
        return run_default_branch(text)
