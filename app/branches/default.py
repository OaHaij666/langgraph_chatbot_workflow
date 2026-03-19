from typing import Generator
from app.llm import LLMClient


def run_default_branch(text: str) -> Generator[str, None, None]:
    llm = LLMClient()
    prompt = (
        "你是一个通用AI助手。请基于用户输入给出清晰、简洁、友好的回答。\n\n"
        f"用户输入：{text}"
    )
    try:
        yield llm.chat(prompt)
    except Exception as e:
        yield f"默认分支处理失败：{str(e)}"
