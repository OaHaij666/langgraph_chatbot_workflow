from typing import Generator
from app.llm import LLMClient


def run_task_branch(text: str) -> Generator[str, None, None]:
    """
    任务分支：处理需要执行特定任务的用户请求。
    
    Args:
        text: 用户输入文本
    
    Returns:
        任务执行结果
    """
    llm = LLMClient()
    prompt = (
        "你是一个任务助手。请将用户请求拆解为可执行步骤，"
        "如果无法执行具体动作，给出清晰可操作的下一步。\n\n"
        f"用户请求：{text}"
    )
    try:
        yield llm.chat(prompt)
    except Exception as e:
        yield f"任务分支处理失败：{str(e)}"
