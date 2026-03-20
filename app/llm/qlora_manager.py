"""
QLoRA 动态切换管理器

本模块实现了 QLoRA 多适配器的预加载、底模常驻、动态切换、运行态监控等核心功能。

设计目标:
- 启动阶段预加载所有可能用到的 QLoRA，降低首次切换时延
- 底模（如 Qwen3.5 4B）一次加载后常驻显存，不重复加载
- adapter 与底模解耦，支持请求级快速切换
- 支持热切换能力探测与降级策略（fallback）
- 内存监控与缓存上限控制，避免多 adapter 叠加导致 OOM

两种切换模式:
1. 热切换模式（hot swap）: 底模常驻，仅加载 adapter 参数，通过 set_adapter 切换
2. Fallback 模式: 当热切换不可用时，完整加载 model+adapter 组合，缓存到 _full_model_cache

使用方式:
    manager = QLoRASwitchManager.get_instance()  # 单例模式
    manager.activate_adapter(adapter_id="coding_agent")  # 切换到指定 adapter
    model, tokenizer, processor = manager.get_active_bundle()  # 获取当前激活的模型

参考文档: QLORA_DYNAMIC_SWITCH_TECH.md
"""
from __future__ import annotations

import time
import torch
from typing import Optional
from config import get_config


class QLoRASwitchManager:
    """
    QLoRA 动态切换管理器（单例模式）

    负责管理底模和多个 QLoRA adapter 的加载、切换和监控。

    核心属性:
        _base_model: 底模实例，启动后常驻显存
        _base_tokenizer: 底模对应的 tokenizer
        _base_processor: 多模态处理器（可选）
        _active_model: 当前激活的模型（可能是底模或 fallback 模型）
        _adapter_index: 已加载的 hot swap adapter 索引 {adapter_id: entry}
        _hot_switch_supported: 是否支持热切换（通过 hasattr 检测）
        _full_model_cache: fallback 模式下的完整模型缓存 {adapter_id: (model, tokenizer, processor)}
        _switch_history: 切换历史记录，用于性能监控

    并发安全提示:
        本类使用单例模式，activate_adapter() 和 get_active_bundle() 之间
        存在竞态条件。如果多个请求并发调用 activate_adapter()，
        可能导致请求 A 切换到 adapter_1 后，请求 B 切换到 adapter_2，
        导致请求 A 使用了 adapter_2 的模型。

        解决方案:
        1. 客户端层面串行化请求（当前项目采用此方案）
        2. 或在 activate_adapter() 和 get_active_bundle() 之间加锁保证原子性
    """
    _instance: "QLoRASwitchManager" | None = None

    def __init__(self):
        self._base_model = None
        self._base_tokenizer = None
        self._base_processor = None
        self._active_model = None
        self._active_tokenizer = None
        self._active_processor = None
        self._active_adapter_id = None
        self._adapter_index: dict[str, dict] = {}
        self._hot_switch_supported = False
        self._full_model_cache: dict[str, tuple] = {}
        self._switch_history: list[dict] = []
        self._base_loaded = False

    @classmethod
    def get_instance(cls) -> "QLoRASwitchManager":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _normalize_adapter_entry(self, adapter_id: str, raw) -> dict:
        """
        标准化 adapter 配置条目

        支持两种配置格式:
        1. 字符串格式: "adapter_id": "./path/to/adapter"
        2. 字典格式: {"id": "xxx", "path": "./xxx", "params": {...}}

        Args:
            adapter_id: adapter 标识符
            raw: 原始配置值（字符串或字典）

        Returns:
            标准化的 adapter 条目: {"id", "name", "path", "params"}
        """
        if isinstance(raw, str):
            return {"id": adapter_id, "name": adapter_id, "path": raw, "params": {}}
        if isinstance(raw, dict):
            path = raw.get("path") or raw.get("adapter_path")
            params = raw.get("params", {})
            lora_scale = raw.get("lora_scale")
            if lora_scale is not None:
                params = {**params, "lora_scale": lora_scale}
            return {
                "id": raw.get("id", adapter_id),
                "name": raw.get("name", adapter_id),
                "path": path,
                "params": params,
            }
        return {"id": adapter_id, "name": adapter_id, "path": None, "params": {}}

    def _get_adapter_registry(self) -> dict[str, dict]:
        """
        获取 adapter 注册表

        从配置文件读取 llm.adapters 或 llm.adapter_map，构建完整的 adapter 注册表。
        如果没有配置 default adapter，会使用 llm.adapter_path 作为兜底。

        Returns:
            adapter 注册表: {adapter_id: {"id", "name", "path", "params"}}
        """
        adapters = get_config("llm.adapters", None)
        if not adapters:
            adapters = get_config("llm.adapter_map", None)
        registry: dict[str, dict] = {}
        if isinstance(adapters, dict):
            for adapter_id, raw in adapters.items():
                entry = self._normalize_adapter_entry(adapter_id, raw)
                if entry["path"]:
                    registry[adapter_id] = entry
        elif isinstance(adapters, list):
            for idx, raw in enumerate(adapters):
                adapter_id = raw.get("id") if isinstance(raw, dict) else f"adapter_{idx}"
                entry = self._normalize_adapter_entry(adapter_id, raw)
                if entry["path"]:
                    registry[entry["id"]] = entry
        default_path = get_config("llm.adapter_path")
        if default_path and "default" not in registry:
            registry["default"] = self._normalize_adapter_entry("default", default_path)
        return registry

    def resolve_adapter(self, adapter_id: Optional[str], adapter_path: Optional[str] = None) -> dict:
        """
        解析 adapter 配置

        优先级:
        1. 如果提供了 adapter_path，直接使用该路径
        2. 如果提供了 adapter_id 且在注册表中，返回对应配置
        3. 返回 default adapter
        4. 抛出异常

        Args:
            adapter_id: adapter 标识符
            adapter_path: adapter 路径（优先级最高）

        Returns:
            adapter 配置条目

        Raises:
            ValueError: 没有可用的 adapter 配置
        """
        registry = self._get_adapter_registry()
        if adapter_path:
            return {"id": adapter_id or "runtime", "name": adapter_id or "runtime", "path": adapter_path, "params": {}}
        if adapter_id and adapter_id in registry:
            return registry[adapter_id]
        if "default" in registry:
            return registry["default"]
        raise ValueError("未配置可用的 QLoRA adapter，请检查 llm.adapters 或 llm.adapter_path")

    def _load_base_model(self):
        """
        加载底模（常驻显存）

        底模只加载一次，后续通过 _base_loaded 标记避免重复加载。
        加载完成后会检测是否支持热切换（通过 hasattr 检查 load_adapter/set_adapter 方法）。

        配置项:
        - llm.base_model_path: 底模路径
        - llm.max_seq_length: 最大序列长度
        - llm.load_in_4bit: 是否使用 4bit 量化
        - llm.enable_multimodal: 是否启用多模态支持
        """
        if self._base_loaded:
            return
        from unsloth import FastLanguageModel

        base_model_path = get_config("llm.base_model_path", get_config("llm.model_path", "./models/Qwen3___5-4B"))
        max_seq_length = get_config("llm.max_seq_length", 8192)
        load_in_4bit = get_config("llm.load_in_4bit", True)
        self._base_model, self._base_tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_path,
            max_seq_length=max_seq_length,
            dtype=None if load_in_4bit else torch.float16,
            load_in_4bit=load_in_4bit,
            trust_remote_code=True,
            local_files_only=True,
        )
        FastLanguageModel.for_inference(self._base_model)
        if get_config("llm.enable_multimodal", False):
            try:
                from transformers import AutoProcessor
                self._base_processor = AutoProcessor.from_pretrained(
                    base_model_path,
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except Exception:
                self._base_processor = None
        self._active_model = self._base_model
        self._active_tokenizer = self._base_tokenizer
        self._active_processor = self._base_processor
        self._base_loaded = True
        self._hot_switch_supported = hasattr(self._base_model, "load_adapter") and hasattr(self._base_model, "set_adapter")

    def _record_switch(self, adapter_id: str, elapsed_ms: float):
        """
        记录切换历史

        用于性能监控和问题排查，保留最近 N 条记录（由 llm.switch_history_size 配置）。

        Args:
            adapter_id: 切换到的 adapter ID
            elapsed_ms: 切换耗时（毫秒）
        """
        self._switch_history.append(
            {"adapter_id": adapter_id, "elapsed_ms": round(elapsed_ms, 2), "timestamp": time.time()}
        )
        max_records = get_config("llm.switch_history_size", 30)
        if len(self._switch_history) > max_records:
            self._switch_history = self._switch_history[-max_records:]

    def _set_lora_scale(self, model, adapter_entry: dict):
        """
        设置 LoRA scale 参数

        通过 adapter.params.lora_scale 或全局配置 llm.lora_scale 设置缩放系数。

        Args:
            model: 目标模型
            adapter_entry: adapter 配置条目
        """
        params = adapter_entry.get("params", {})
        lora_scale = params.get("lora_scale", get_config("llm.lora_scale", 0.6))
        if hasattr(model, "set_scale"):
            try:
                model.set_scale("default", lora_scale)
            except Exception:
                pass

    def _activate_hot_swap_adapter(self, adapter_entry: dict):
        """
        热切换模式激活 adapter

        流程:
        1. 如果 adapter 未加载，调用 load_adapter 加载到内存
        2. 调用 set_adapter 切换到目标 adapter
        3. 设置 lora_scale
        4. 更新 _active_* 属性

        这种模式下底模保持不变，仅切换 adapter 参数，延迟低。

        Args:
            adapter_entry: adapter 配置条目
        """
        adapter_id = adapter_entry["id"]
        adapter_path = adapter_entry["path"]
        if adapter_id not in self._adapter_index:
            try:
                self._base_model.load_adapter(adapter_path, adapter_name=adapter_id)
            except TypeError:
                self._base_model.load_adapter(adapter_path, adapter_id)
            self._adapter_index[adapter_id] = adapter_entry
        self._base_model.set_adapter(adapter_id)
        self._set_lora_scale(self._base_model, adapter_entry)
        self._active_model = self._base_model
        self._active_tokenizer = self._base_tokenizer
        self._active_processor = self._base_processor
        self._active_adapter_id = adapter_id

    def _activate_fallback_adapter(self, adapter_entry: dict):
        """
        Fallback 模式激活 adapter

        当热切换不可用或失败时，使用此模式完整加载 model+adapter 组合。
        加载后的模型会缓存到 _full_model_cache，避免重复加载。

        注意: 这种模式下每个 adapter 会占用独立的显存，需要通过 max_resident_adapters 控制上限。

        Args:
            adapter_entry: adapter 配置条目
        """
        adapter_id = adapter_entry["id"]
        adapter_path = adapter_entry["path"]
        if adapter_id not in self._full_model_cache:
            from unsloth import FastLanguageModel

            max_seq_length = get_config("llm.max_seq_length", 8192)
            load_in_4bit = get_config("llm.load_in_4bit", True)
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=adapter_path,
                max_seq_length=max_seq_length,
                dtype=None if load_in_4bit else torch.float16,
                load_in_4bit=load_in_4bit,
                trust_remote_code=True,
                local_files_only=True,
            )
            FastLanguageModel.for_inference(model)
            self._set_lora_scale(model, adapter_entry)
            processor = None
            if get_config("llm.enable_multimodal", False):
                try:
                    from transformers import AutoProcessor
                    processor = AutoProcessor.from_pretrained(
                        adapter_path,
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                except Exception:
                    processor = None
            self._full_model_cache[adapter_id] = (model, tokenizer, processor)
        self._active_model, self._active_tokenizer, self._active_processor = self._full_model_cache[adapter_id]
        self._active_adapter_id = adapter_id
        self._enforce_fallback_cache_limit()

    def _enforce_fallback_cache_limit(self):
        """
        强制执行 fallback 缓存上限

        当 fallback 模式下加载的模型数量超过 llm.max_resident_adapters 时，
        移除非活跃的模型并清理显存。

        这是 OOM 保护机制的一部分。

        并发安全提示:
            本方法直接移除缓存中的模型，不检查是否有请求正在使用。
            如果请求 A 持有 adapter_1 的模型引用，而请求 B 触发缓存清理，
            adapter_1 可能被移除，导致请求 A 访问已释放的内存而崩溃。

            当前项目通过客户端串行化请求避免此问题。
            如果后续需要支持并发，需要实现引用计数机制：
            1. get_active_bundle() 时增加引用计数
            2. 请求完成后减少引用计数
            3. 本方法只移除引用计数为 0 的模型
        """
        max_resident = get_config("llm.max_resident_adapters", 4)
        if max_resident <= 0:
            return
        if len(self._full_model_cache) <= max_resident:
            return
        removable = [k for k in self._full_model_cache.keys() if k != self._active_adapter_id]
        while len(self._full_model_cache) > max_resident and removable:
            to_remove = removable.pop(0)
            self._full_model_cache.pop(to_remove, None)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def activate_adapter(self, adapter_id: Optional[str] = None, adapter_path: Optional[str] = None) -> dict:
        """
        激活指定的 adapter（核心切换接口）

        切换流程:
        1. 确保底模已加载
        2. 解析 adapter 配置
        3. 优先尝试热切换模式
        4. 热切换失败则降级到 fallback 模式
        5. 记录切换耗时

        Args:
            adapter_id: 目标 adapter ID
            adapter_path: adapter 路径（优先级高于 adapter_id）

        Returns:
            切换结果，包含:
            - adapter_id: 当前激活的 adapter ID
            - adapter_path: adapter 路径
            - adapter_params: adapter 级别的生成参数
            - switch_latency_ms: 切换耗时（毫秒）
            - hot_switch_supported: 是否使用了热切换
            - memory: 显存状态报告
        """
        self._load_base_model()
        entry = self.resolve_adapter(adapter_id=adapter_id, adapter_path=adapter_path)
        started = time.perf_counter()
        if self._hot_switch_supported:
            try:
                self._activate_hot_swap_adapter(entry)
            except Exception:
                self._hot_switch_supported = False
                self._activate_fallback_adapter(entry)
        else:
            self._activate_fallback_adapter(entry)
        elapsed_ms = (time.perf_counter() - started) * 1000
        self._record_switch(entry["id"], elapsed_ms)
        return {
            "adapter_id": self._active_adapter_id,
            "adapter_path": entry["path"],
            "adapter_params": entry.get("params", {}),
            "switch_latency_ms": round(elapsed_ms, 2),
            "hot_switch_supported": self._hot_switch_supported,
            "memory": self.get_memory_report(),
        }

    def preload_all_adapters(self) -> list[str]:
        """
        预加载所有注册的 adapter

        在启动阶段调用，确保常用 adapter 已进入缓存，
        减少首次请求时的切换延迟。

        Returns:
            已加载的 adapter ID 列表
        """
        self._load_base_model()
        registry = self._get_adapter_registry()
        loaded = []
        for adapter_id in registry:
            self.activate_adapter(adapter_id=adapter_id)
            loaded.append(adapter_id)
        return loaded

    def get_active_bundle(self):
        """
        获取当前激活的模型组件

        Returns:
            (model, tokenizer, processor) 三元组
        """
        self._load_base_model()
        return self._active_model, self._active_tokenizer, self._active_processor

    def get_memory_report(self) -> dict:
        """
        获取显存使用报告

        Returns:
            包含 CUDA 显存统计和缓存状态的字典
        """
        report = {"cuda_available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            report.update(
                {
                    "allocated_mb": round(torch.cuda.memory_allocated() / (1024 ** 2), 2),
                    "reserved_mb": round(torch.cuda.memory_reserved() / (1024 ** 2), 2),
                    "max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
                }
            )
        report["resident_hot_swap_adapters"] = len(self._adapter_index)
        report["resident_fallback_adapters"] = len(self._full_model_cache)
        return report

    def get_runtime_status(self) -> dict:
        """
        获取运行态状态（用于监控接口）

        Returns:
            完整的运行态信息，包括:
            - base_model_loaded: 底模是否已加载
            - hot_switch_supported: 是否支持热切换
            - active_adapter_id: 当前激活的 adapter
            - loaded_hot_swap_adapters: 已加载的热切换 adapter 列表
            - loaded_fallback_adapters: 已加载的 fallback adapter 列表
            - switch_history: 最近 10 条切换记录
            - memory: 显存状态
        """
        return {
            "base_model_loaded": self._base_loaded,
            "hot_switch_supported": self._hot_switch_supported,
            "active_adapter_id": self._active_adapter_id,
            "loaded_hot_swap_adapters": list(self._adapter_index.keys()),
            "loaded_fallback_adapters": list(self._full_model_cache.keys()),
            "switch_history": self._switch_history[-10:],
            "memory": self.get_memory_report(),
        }
