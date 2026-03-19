from typing import Generator
import base64
import io
import torch
import requests
from config import get_config
from app.llm.base import BaseLLMClient
from transformers import TextStreamer


class TimedTextStreamer(TextStreamer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_token_time = None
        self.start_time = None

    def on_finalized_text(self, text: str, stream_end: bool = False):
        text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        if text:
            super().on_finalized_text(text, stream_end)


class UnslothClient(BaseLLMClient):
    def __init__(self, model_path: str = None):
        self.model_path = model_path or get_config("llm.model_path", "./models")
        self._model = None
        self._tokenizer = None
        self._processor = None

    def _load_model(self):
        if self._model is None:
            from unsloth import FastLanguageModel

            adapter_path = get_config("llm.adapter_path")
            max_seq_length = get_config("llm.max_seq_length", 8192)
            load_in_4bit = get_config("llm.load_in_4bit", True)

            self._model, self._tokenizer = FastLanguageModel.from_pretrained(
                model_name=adapter_path,
                max_seq_length=max_seq_length,
                dtype=None if load_in_4bit else torch.float16,
                load_in_4bit=load_in_4bit,
                trust_remote_code=True,
                local_files_only=True,
            )

            torch.cuda.empty_cache()
            FastLanguageModel.for_inference(self._model)

            lora_scale = get_config("llm.lora_scale", 0.6)
            if hasattr(self._model, 'set_scale'):
                self._model.set_scale('default', lora_scale)

            if get_config("llm.enable_multimodal", False):
                try:
                    from transformers import AutoProcessor
                    self._processor = AutoProcessor.from_pretrained(
                        adapter_path,
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                except Exception:
                    self._processor = None

    def _get_text_tokenizer(self):
        if hasattr(self._tokenizer, 'tokenizer'):
            return self._tokenizer.tokenizer
        return self._tokenizer

    def _load_image(self, image_input: dict):
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
        self._load_model()
        inputs, text_tokenizer = self._build_inputs(prompt, multimodal_inputs)

        max_new_tokens = get_config("llm.max_new_tokens", 500)
        temperature = get_config("llm.temperature", 0.7)
        top_p = get_config("llm.top_p", 0.9)

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
        result = result.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        result = result.replace("<think>", "").replace("</think>", "").strip()
        return result.split("assistant\n")[-1].strip()

    def stream_chat(self, prompt: str, multimodal_inputs: list[dict] | None = None) -> Generator[str, None, None]:
        self._load_model()
        inputs, text_tokenizer = self._build_inputs(prompt, multimodal_inputs)

        from threading import Thread

        max_new_tokens = get_config("llm.max_new_tokens", 500)
        temperature = get_config("llm.temperature", 0.7)
        top_p = get_config("llm.top_p", 0.9)

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
            chunk = chunk.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
            chunk = chunk.replace("<think>", "").replace("</think>", "").strip()
            if chunk:
                yield chunk

        thread.join()
