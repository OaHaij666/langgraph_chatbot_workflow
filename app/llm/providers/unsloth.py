from typing import Generator
from config import get_config
from app.llm.base import BaseLLMClient


class UnslothClient(BaseLLMClient):
    """
    Unsloth 本地模型客户端。
    用户只需将模型文件放入指定文件夹，系统会自动使用 Unsloth 加速推理。
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or get_config("llm.model_path", "./models")
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """加载模型（延迟加载）"""
        if self._model is None:
            from unsloth import FastLanguageModel
            self._model, self._tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=2048,
                load_in_4bit=True
            )
            FastLanguageModel.for_inference(self._model)
    
    def chat(self, prompt: str) -> str:
        """同步调用 Unsloth 模型"""
        self._load_model()
        
        messages = [{"role": "user", "content": prompt}]
        from unsloth import chat_template
        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self._tokenizer(text, return_tensors="pt").to("cuda")
        outputs = self._model.generate(**inputs, max_new_tokens=512, use_cache=True)
        result = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return result.split("assistant\n")[-1].strip()
    
    def stream_chat(self, prompt: str) -> Generator[str, None, None]:
        """流式调用 Unsloth 模型"""
        self._load_model()
        
        messages = [{"role": "user", "content": prompt}]
        from unsloth import chat_template
        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self._tokenizer(text, return_tensors="pt").to("cuda")
        
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True)
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": 512,
            "use_cache": True
        }
        
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for chunk in streamer:
            yield chunk
        
        thread.join()
