"""
LLM 模块统一入口

本模块提供 LLM 客户端的统一创建和管理接口，支持多种 Provider:
- openai: OpenAI API
- anthropic: Anthropic Claude API
- ollama: 本地 Ollama 服务
- proxy: 自定义代理服务
- unsloth: 本地 Unsloth 模型（支持 QLoRA 动态切换）

核心功能:
1. LLMClient: 统一的 LLM 客户端封装，根据 provider 配置自动选择实现
2. resolve_adapter: 根据 agent_id 解析对应的 QLoRA adapter 配置
3. preload_unsloth_adapters: 启动时预加载所有 adapter
4. get_unsloth_runtime_status: 获取 Unsloth 运行态信息

与 QLoRA 动态切换的关系:
- 当 provider=unsloth 时，LLMClient 会创建 UnslothClient
- UnslothClient 通过 QLoRASwitchManager 获取模型
- resolve_adapter 用于将 agent_id 映射到 adapter 配置

参考文档: QLORA_DYNAMIC_SWITCH_TECH.md
"""
import os
from typing import Generator, Optional
from config import get_config
from app.llm.base import BaseLLMClient
from app.llm.qlora_manager import QLoRASwitchManager
from app.llm.providers import openai, anthropic, ollama, proxy, unsloth


def _normalize_adapter_entry(adapter_id: str, raw) -> dict:
    """
    标准化 adapter 配置条目

    支持两种配置格式:
    1. 字符串格式: "adapter_id": "./path/to/adapter"
    2. 字典格式: {"id": "xxx", "path": "./xxx", "params": {...}, "lora_scale": 0.6}

    Args:
        adapter_id: adapter 标识符
        raw: 原始配置值

    Returns:
        标准化的 adapter 条目: {"id", "name", "path", "params"}
    """
    if isinstance(raw, str):
        return {"id": adapter_id, "name": adapter_id, "path": raw, "params": {}}
    if isinstance(raw, dict):
        params = raw.get("params", {})
        lora_scale = raw.get("lora_scale")
        if lora_scale is not None:
            params = {**params, "lora_scale": lora_scale}
        return {
            "id": raw.get("id", adapter_id),
            "name": raw.get("name", adapter_id),
            "path": raw.get("path") or raw.get("adapter_path"),
            "params": params,
        }
    return {"id": adapter_id, "name": adapter_id, "path": None, "params": {}}


def get_adapter_registry() -> dict[str, dict]:
    """
    获取 adapter 注册表

    从配置文件读取 llm.adapters 或 llm.adapter_map，构建完整的 adapter 注册表。
    这是 resolve_adapter 的底层实现，也用于遍历所有 adapter（如预加载）。

    配置格式示例:
        "llm.adapters": {
            "default": {"path": "./models/adapter-default", "lora_scale": 0.6},
            "coding_agent": {"path": "./models/adapter-coding", "lora_scale": 0.7}
        }

    Returns:
        adapter 注册表: {adapter_id: {"id", "name", "path", "params"}}
    """
    adapters = get_config("llm.adapters", None)
    if not adapters:
        adapters = get_config("llm.adapter_map", None)

    registry: dict[str, dict] = {}
    if isinstance(adapters, dict):
        for adapter_id, raw in adapters.items():
            entry = _normalize_adapter_entry(adapter_id, raw)
            if entry["path"]:
                registry[entry["id"]] = entry
    elif isinstance(adapters, list):
        for idx, raw in enumerate(adapters):
            adapter_id = raw.get("id") if isinstance(raw, dict) else f"adapter_{idx}"
            entry = _normalize_adapter_entry(adapter_id, raw)
            if entry["path"]:
                registry[entry["id"]] = entry

    default_path = get_config("llm.adapter_path")
    if default_path and "default" not in registry:
        registry["default"] = _normalize_adapter_entry("default", default_path)
    return registry


def resolve_adapter(agent_id: Optional[str] = None) -> dict:
    """
    根据 agent_id 解析 adapter 配置

    解析优先级:
    1. 如果 agent_id 在注册表中，返回对应配置
    2. 返回 default adapter
    3. 返回注册表中的第一个 adapter
    4. 抛出异常

    这是 LangGraph chat 分支等模块将 agent_id 映射到 adapter 的核心函数。

    Args:
        agent_id: 目标 agent ID（如 "coding_agent", "translator" 等）

    Returns:
        adapter 配置条目: {"id", "name", "path", "params"}

    Raises:
        ValueError: 没有可用的 adapter 配置
    """
    registry = get_adapter_registry()
    if agent_id and agent_id in registry:
        return registry[agent_id]
    if "default" in registry:
        return registry["default"]
    if registry:
        first_key = next(iter(registry.keys()))
        return registry[first_key]
    raise ValueError("未配置可用的 QLoRA adapter，请检查 llm.adapters 或 llm.adapter_path")


def resolve_adapter_path(agent_id: Optional[str] = None) -> Optional[str]:
    """
    根据 agent_id 解析 adapter 路径

    这是 resolve_adapter 的便捷封装，仅返回 path 字段。

    Args:
        agent_id: 目标 agent ID

    Returns:
        adapter 路径，如果无法解析则返回 llm.adapter_path 配置
    """
    try:
        return resolve_adapter(agent_id).get("path")
    except ValueError:
        return get_config("llm.adapter_path")


class LLMClient(BaseLLMClient):
    """
    LLM 客户端统一封装

    根据配置自动选择对应的 Provider 实现，对外提供统一的接口。
    支持 chat() 和 stream_chat() 方法。

    当 provider=unsloth 时，会创建 UnslothClient 并支持 adapter 切换。

    使用示例:
        # 使用默认配置
        client = LLMClient()
        response = client.chat("你好")

        # 指定 adapter
        client = LLMClient(provider="unsloth", adapter_id="coding_agent")
        response = client.chat("写一个函数")
    """

    def __init__(self, provider: str = None, adapter_path: Optional[str] = None, adapter_id: Optional[str] = None):
        """
        初始化 LLM 客户端

        Args:
            provider: LLM 提供者（openai/anthropic/ollama/proxy/unsloth）
            adapter_path: adapter 路径（仅 unsloth provider 使用）
            adapter_id: adapter ID（仅 unsloth provider 使用）
        """
        provider = provider or get_config("llm.provider", "openai")
        self._client = self._create_client(provider, adapter_path, adapter_id)

    def _create_client(self, provider: str, adapter_path: Optional[str], adapter_id: Optional[str]) -> BaseLLMClient:
        """
        创建具体的 Provider 客户端

        根据配置创建对应的客户端实例:
        - openai: OpenAIClient
        - anthropic: AnthropicClient
        - ollama: OllamaClient
        - proxy: ProxyClient
        - unsloth: UnslothClient（支持 QLoRA 动态切换）

        Args:
            provider: 提供者名称
            adapter_path: adapter 路径
            adapter_id: adapter ID

        Returns:
            具体的客户端实例

        Raises:
            ValueError: 未知的 provider
        """
        if provider == "openai":
            return openai.OpenAIClient(
                api_key=get_config("llm.api_key") or os.getenv("OPENAI_API_KEY"),
                model=get_config("llm.model", "gpt-3.5-turbo")
            )
        elif provider == "anthropic":
            return anthropic.AnthropicClient(
                api_key=get_config("llm.api_key") or os.getenv("ANTHROPIC_API_KEY"),
                model=get_config("llm.model", "claude-3-haiku-20240307")
            )
        elif provider == "ollama":
            return ollama.OllamaClient(
                base_url=get_config("llm.base_url") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                model=get_config("llm.model", "llama2")
            )
        elif provider == "proxy":
            return proxy.ProxyClient(
                base_url=get_config("llm.base_url") or os.getenv("LLM_PROXY_URL"),
                api_key=get_config("llm.api_key") or os.getenv("LLM_PROXY_KEY"),
                model=get_config("llm.model")
            )
        elif provider == "unsloth":
            return unsloth.UnslothClient(
                model_path=get_config("llm.base_model_path", get_config("llm.model_path", "./models")),
                adapter_path=adapter_path,
                adapter_id=adapter_id,
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    def chat(self, prompt: str, multimodal_inputs: list[dict] | None = None) -> str:
        """
        同步聊天接口

        Args:
            prompt: 用户输入
            multimodal_inputs: 多模态输入（部分 provider 支持）

        Returns:
            模型回复
        """
        try:
            return self._client.chat(prompt, multimodal_inputs=multimodal_inputs)
        except TypeError:
            return self._client.chat(prompt)

    def stream_chat(self, prompt: str, multimodal_inputs: list[dict] | None = None) -> Generator[str, None, None]:
        """
        流式聊天接口

        Args:
            prompt: 用户输入
            multimodal_inputs: 多模态输入（部分 provider 支持）

        Yields:
            生成的文本块
        """
        try:
            return self._client.stream_chat(prompt, multimodal_inputs=multimodal_inputs)
        except TypeError:
            return self._client.stream_chat(prompt)


def get_llm_client(
    adapter_path: Optional[str] = None,
    provider: Optional[str] = None,
    adapter_id: Optional[str] = None,
) -> LLMClient:
    """
    获取 LLM 客户端实例

    这是创建 LLMClient 的推荐方式，参数会从配置中自动填充。

    Args:
        adapter_path: adapter 路径（仅 unsloth）
        provider: 提供者名称
        adapter_id: adapter ID（仅 unsloth）

    Returns:
        LLMClient 实例
    """
    return LLMClient(provider=provider, adapter_path=adapter_path, adapter_id=adapter_id)


def preload_unsloth_adapters() -> list[str]:
    """
    预加载所有 Unsloth adapter

    在 FastAPI 启动时调用，确保所有 adapter 已加载到缓存，
    减少首次请求的延迟。

    Returns:
        已加载的 adapter ID 列表
    """
    manager = QLoRASwitchManager.get_instance()
    return manager.preload_all_adapters()


def get_unsloth_runtime_status() -> dict:
    """
    获取 Unsloth 运行态状态

    用于 /llm/runtime 监控接口，返回:
    - base_model_loaded: 底模是否已加载
    - hot_switch_supported: 是否支持热切换
    - active_adapter_id: 当前激活的 adapter
    - loaded_hot_swap_adapters: 已加载的热切换 adapter
    - loaded_fallback_adapters: 已加载的 fallback adapter
    - switch_history: 切换历史
    - memory: 显存状态

    Returns:
        运行态状态字典
    """
    manager = QLoRASwitchManager.get_instance()
    return manager.get_runtime_status()
