"""
Unsloth LLM Provider

本模块实现了基于 Unsloth 框架的本地 LLM 客户端，支持:
- QLoRA adapter 动态切换（通过 QLoRASwitchManager）
- 文本生成（同步/流式）
- 多模态输入（图像）

核心流程:
1. 客户端初始化时获取 QLoRASwitchManager 单例
2. 每次调用前通过 _activate() 激活目标 adapter
3. 使用激活后的模型进行推理

与 QLoRASwitchManager 的关系:
- UnslothClient 不直接加载模型，而是通过 manager 获取
- manager 负责底模常驻和 adapter 切换
- UnslothClient 负责构建输入和执行推理

参考文档: QLORA_DYNAMIC_SWITCH_TECH.md
"""
from typing import Generator
import base64
import io
import requests
from config import get_config
from app.llm.base import BaseLLMClient
from app.llm.qlora_manager import QLoRASwitchManager
from transformers import TextStreamer


class TimedTextStreamer(TextStreamer):
    """
    带计时功能的文本流式输出器

    继承自 transformers.TextStreamer，用于流式输出模型生成的文本。
    同时记录首 token 时间，用于性能监控。

    注意: 会自动过滤特殊标记如 <|im_end|>
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_token_time = None
        self.start_time = None

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """处理生成的文本块，过滤特殊标记"""
        text = text.replace("<|im_end|>", "").replace("</s>", "").strip()
        if text:
            super().on_finalized_text(text, stream_end)


class UnslothClient(BaseLLMClient):
    """
    Unsloth 本地 LLM 客户端

    通过 QLoRASwitchManager 获取模型实例，支持 QLoRA adapter 动态切换。

    属性:
        model_path: 底模路径（仅用于记录，实际加载由 manager 负责）
        adapter_path: 目标 adapter 路径
        adapter_id: 目标 adapter ID
        _manager: QLoRASwitchManager 单例
        _model/_tokenizer/_processor: 当前激活的模型组件
        _adapter_params: adapter 级别的生成参数（优先级高于全局配置）

    使用示例:
        client = UnslothClient(adapter_id="coding_agent")
        response = client.chat("你好")  # 自动激活 coding_agent adapter
    """

    def __init__(self, model_path: str = None, adapter_path: str | None = None, adapter_id: str | None = None):
        """
        初始化 Unsloth 客户端

        Args:
            model_path: 底模路径（可选，从配置读取）
            adapter_path: adapter 路径（优先级高于 adapter_id）
            adapter_id: adapter ID（在注册表中查找）
        """
        self.model_path = model_path or get_config("llm.base_model_path", get_config("llm.model_path", "./models"))
        self.adapter_path = adapter_path
        self.adapter_id = adapter_id
        self._manager = QLoRASwitchManager.get_instance()
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._adapter_params: dict = {}

    def _activate(self):
        """
        激活目标 adapter

        调用 QLoRASwitchManager.activate_adapter() 切换到目标 adapter，
        然后获取当前激活的模型组件。

        这是每次推理前的必要步骤，确保使用正确的 adapter。

        并发安全提示:
            由于 QLoRASwitchManager 是单例，activate_adapter() 和 get_active_bundle()
            之间可能被其他请求插入。当前项目通过客户端串行化请求来避免此问题。
            如果后续需要支持并发，请参考 QLoRASwitchManager 类文档中的解决方案。
        """
        result = self._manager.activate_adapter(adapter_id=self.adapter_id, adapter_path=self.adapter_path)
        self._adapter_params = result.get("adapter_params", {})
        self._model, self._tokenizer, self._processor = self._manager.get_active_bundle()

    def _get_text_tokenizer(self):
        """
        获取文本 tokenizer

        某些 tokenizer 包装器（如 Unsloth 的）会有 .tokenizer 属性，
        需要解包才能访问底层的方法。
        """
        if hasattr(self._tokenizer, 'tokenizer'):
            return self._tokenizer.tokenizer
        return self._tokenizer

    def _load_image(self, image_input: dict):
        """
        加载图像输入

        支持三种图像来源:
        1. path: 本地文件路径
        2. url: 网络 URL
        3. base64: Base64 编码（支持 data:image/xxx;base64, 前缀）

        Args:
            image_input: {"source": "path|url|base64", "value": "..."}

        Returns:
            PIL.Image 对象（RGB 模式）

        Raises:
            ValueError: 缺少 value 或不支持的 source 类型
        """
        source = image_input.get("source", "path")
        value = image_input.get("value", "")
        if not value:
            raise ValueError("多模态输入缺少 value")
        try:
            from PIL import Image
        except Exception as e:
            raise ValueError(f"多模态输入需要 Pillow 支持: {str(e)}")

        if source == "path":
            return Image.open(value).convert("RGB")
        if source == "url":
            response = requests.get(value, timeout=20)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        if source == "base64":
            base64_data = value.split(",", 1)[1] if "," in value else value
            decoded = base64.b64decode(base64_data)
            return Image.open(io.BytesIO(decoded)).convert("RGB")
        raise ValueError(f"不支持的多模态输入 source: {source}")

    def _build_inputs(self, prompt: str, multimodal_inputs: list[dict] | None):
        """
        构建模型输入

        根据是否有多模态输入，选择不同的处理路径:
        1. 纯文本: 使用 tokenizer 直接编码
        2. 多模态: 使用 processor 构建输入（包含图像）

        Args:
            prompt: 文本提示
            multimodal_inputs: 多模态输入列表 [{"type": "image", "source": "...", "value": "..."}]

        Returns:
            (inputs, text_tokenizer) 元组
            - inputs: 可直接传入 model.generate() 的输入张量
            - text_tokenizer: 用于解码输出的 tokenizer

        Raises:
            ValueError: 多模态输入但未启用多模态支持，或 processor 不可用
        """
        multimodal_enabled = get_config("llm.enable_multimodal", False)
        usable_inputs = multimodal_inputs or []
        usable_images = [item for item in usable_inputs if isinstance(item, dict) and item.get("type", "image") == "image"]
        text_tokenizer = self._get_text_tokenizer()

        if not usable_images:
            return text_tokenizer(text=prompt, return_tensors="pt").to("cuda"), text_tokenizer

        if not multimodal_enabled:
            raise ValueError("当前未启用 Unsloth 多模态输入，请在 config.llm.enable_multimodal 中开启")

        if self._processor is None:
            raise ValueError("当前模型未加载到多模态处理器，请确认模型与 processor 可用")

        images = [self._load_image(item) for item in usable_images]
        content = [{"type": "text", "text": prompt}]
        content.extend({"type": "image"} for _ in images)
        messages = [{"role": "user", "content": content}]
        chat_text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[chat_text],
            images=images,
            return_tensors="pt",
        )
        return inputs.to("cuda"), text_tokenizer

    def chat(self, prompt: str, multimodal_inputs: list[dict] | None = None) -> str:
        """
        同步聊天接口

        流程:
        1. 激活目标 adapter
        2. 构建输入（支持多模态）
        3. 执行模型生成
        4. 解码输出并清理特殊标记

        Args:
            prompt: 用户输入文本
            multimodal_inputs: 多模态输入列表（可选）

        Returns:
            模型生成的回复文本
        """
        self._activate()
        inputs, text_tokenizer = self._build_inputs(prompt, multimodal_inputs)

        max_new_tokens = get_config("llm.max_new_tokens", 500)
        temperature = self._adapter_params.get("temperature", get_config("llm.temperature", 0.7))
        top_p = self._adapter_params.get("top_p", get_config("llm.top_p", 0.9))

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            use_cache=True,
            eos_token_id=text_tokenizer.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
        )

        result = text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = result.replace("<|im_end|>", "").replace("</s>", "").strip()
        result = result.replace("<｜end▁of▁thinking｜>", "").replace("<｜begin▁of▁thinking｜>", "").strip()
        return result.split("assistant\n")[-1].strip()

    def stream_chat(self, prompt: str, multimodal_inputs: list[dict] | None = None) -> Generator[str, None, None]:
        """
        流式聊天接口

        使用 TextStreamer 实现流式输出，通过独立线程执行生成，
        主线程逐块 yield 输出。

        流程:
        1. 激活目标 adapter
        2. 构建输入
        3. 创建 TimedTextStreamer
        4. 启动生成线程
        5. 逐块 yield 输出

        Args:
            prompt: 用户输入文本
            multimodal_inputs: 多模态输入列表（可选）

        Yields:
            生成的文本块
        """
        self._activate()
        inputs, text_tokenizer = self._build_inputs(prompt, multimodal_inputs)

        from threading import Thread

        max_new_tokens = get_config("llm.max_new_tokens", 500)
        temperature = self._adapter_params.get("temperature", get_config("llm.temperature", 0.7))
        top_p = self._adapter_params.get("top_p", get_config("llm.top_p", 0.9))

        streamer = TimedTextStreamer(text_tokenizer, skip_prompt=True)

        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "use_cache": True,
            "eos_token_id": text_tokenizer.eos_token_id,
            "pad_token_id": text_tokenizer.pad_token_id,
        }

        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        for chunk in streamer:
            chunk = chunk.replace("<|im_end|>", "").replace("</s>", "").strip()
            chunk = chunk.replace("<｜end▁of▁thinking｜>", "").replace("<｜begin▁of▁thinking｜>", "").strip()
            if chunk:
                yield chunk

        thread.join()
