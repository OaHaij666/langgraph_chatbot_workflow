from pathlib import Path
from typing import Dict, Optional, Iterable


PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


class PromptManager:
    _cache: Dict[str, str] = {}
    
    @classmethod
    def get(cls, name: str) -> str:
        if name not in cls._cache:
            cls._cache[name] = cls._load(name)
        return cls._cache[name]
    
    @classmethod
    def _load(cls, name: str) -> str:
        prompt_file = PROMPTS_DIR / f"{name}.txt"
        
        if prompt_file.exists():
            return prompt_file.read_text(encoding="utf-8").strip()
        
        return cls._default_prompt(name)
    
    @classmethod
    def _default_prompt(cls, name: str) -> str:
        return f"你是一个{name}助手。"
    
    @classmethod
    def reload(cls, name: Optional[str] = None) -> None:
        if name:
            cls._cache.pop(name, None)
        else:
            cls._cache.clear()
    
    @classmethod
    def list_available(cls) -> list:
        if not PROMPTS_DIR.exists():
            return []
        return [f.stem for f in PROMPTS_DIR.glob("*.txt")]
    
    @classmethod
    def get_for_agent(cls, agent_id: str) -> str:
        prompt_name = f"agent_{agent_id}"
        if prompt_name in cls.list_available():
            return cls.get(prompt_name)
        return cls.get("default")
    
    @classmethod
    def get_for_branch(cls, branch: str) -> str:
        if branch in cls.list_available():
            return cls.get(branch)
        return cls.get("default")


def get_prompt(name: str) -> str:
    return PromptManager.get(name)


def get_prompt_for_agent(agent_id: str) -> str:
    return PromptManager.get_for_agent(agent_id)


def get_prompt_for_branch(branch: str) -> str:
    return PromptManager.get_for_branch(branch)


def get_system_prompt(agent_id: Optional[str], branch: str = "chat") -> str:
    if agent_id:
        prompt_name = f"agent_{agent_id}"
        if prompt_name in PromptManager.list_available():
            return PromptManager.get(prompt_name)
    return PromptManager.get_for_branch(branch)


def build_chat_prompt(
    clean_text: str,
    rag_context: list[str],
    history_entries: Iterable,
    system_prompt: str,
) -> str:
    parts = [f"[system prompt]\n{system_prompt}"]

    if rag_context:
        rag_content = "\n".join(rag_context)
        parts.append(f"[rag prompt]\n{rag_content}")

    history_lines = []
    for entry in history_entries or []:
        history_lines.append(entry.to_display_format())
    if history_lines:
        parts.append(f"[history prompt]\n" + "\n".join(history_lines))

    parts.append(f"[user prompt]\n{clean_text}")
    return "\n\n".join(parts)


def reload_prompts(name: Optional[str] = None) -> None:
    PromptManager.reload(name)


def list_prompts() -> list:
    return PromptManager.list_available()
