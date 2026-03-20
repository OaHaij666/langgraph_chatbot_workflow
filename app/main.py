"""
FastAPI 应用入口

本模块是 AI Chat System 的 HTTP API 入口，提供:
- POST /chat: 聊天接口（支持同步和流式）
- GET /llm/runtime: LLM 运行态监控接口
- GET /: 健康检查

与 QLoRA 动态切换的关系:
- 启动时调用 preload_unsloth_adapters() 预加载所有 adapter
- 每次请求根据 target_agent_id/persona 参数选择 adapter
- 通过 _resolve_request_client() 动态创建对应 adapter 的客户端

请求流程:
1. 请求到达 /chat
2. _resolve_request_client() 解析 target_agent_id
3. resolve_adapter() 获取 adapter 配置
4. get_llm_client() 创建 UnslothClient（自动激活 adapter）
5. 执行推理并返回结果

参考文档: QLORA_DYNAMIC_SWITCH_TECH.md
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
import json
import uvicorn
import sys

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("Python:", sys.version, file=sys.stderr)
print("Starting import config...", file=sys.stderr)
from config import get_config
print("Config imported", file=sys.stderr)
from app.llm import get_llm_client, preload_unsloth_adapters, resolve_adapter, get_unsloth_runtime_status
print("LLM client imported", file=sys.stderr)

app = FastAPI(title="AI Chat System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_client = None


@app.on_event("startup")
async def startup_event():
    """
    应用启动事件处理

    根据 provider 配置执行不同的初始化:
    - unsloth: 预加载所有 adapter，创建默认客户端
    - ollama: 检查服务可用性
    - 其他: 直接创建客户端
    """
    global llm_client
    provider = get_config("llm.provider", "openai")
    if provider == "unsloth":
        print(f"Loading {provider} model...")
        preload_unsloth_adapters()
        default_adapter = resolve_adapter()
        llm_client = get_llm_client(
            adapter_id=default_adapter["id"],
            adapter_path=default_adapter["path"],
            provider="unsloth",
        )
        print(f"Model loaded successfully")
    else:
        llm_client = get_llm_client()
        if provider == "ollama":
            try:
                import requests
                base_url = getattr(llm_client._client, "base_url", "http://localhost:11434")
                requests.get(f"{base_url}/api/tags", timeout=3)
            except Exception:
                pass


class ChatRequest(BaseModel):
    """
    聊天请求模型

    属性:
        message: 用户消息内容
        stream: 是否使用流式输出
        output_type: 输出类型（保留字段）
        persona: 目标角色/agent ID（用于 adapter 选择）
        target_agent_id: 目标 agent ID（优先级高于 persona）
        multimodal_inputs: 多模态输入列表（图像等）
    """
    message: str
    stream: bool = False
    output_type: str = "text"
    persona: Optional[str] = None
    target_agent_id: Optional[str] = None
    multimodal_inputs: list[dict] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """
    聊天响应模型

    属性:
        message: 模型回复内容
        intent: 意图标识
    """
    message: str
    intent: str


@app.get("/")
async def root():
    """健康检查接口"""
    return {"status": "ok", "message": "AI Chat System is running"}


@app.get("/llm/runtime")
async def llm_runtime():
    """
    LLM 运行态监控接口

    返回 Unsloth provider 的运行态信息:
    - base_model_loaded: 底模是否已加载
    - hot_switch_supported: 是否支持热切换
    - active_adapter_id: 当前激活的 adapter
    - loaded_hot_swap_adapters: 已加载的热切换 adapter 列表
    - loaded_fallback_adapters: 已加载的 fallback adapter 列表
    - switch_history: 切换历史记录
    - memory: 显存使用情况

    对于非 unsloth provider，返回 null。
    """
    provider = get_config("llm.provider", "openai")
    if provider != "unsloth":
        return {"provider": provider, "runtime": None}
    return {"provider": provider, "runtime": get_unsloth_runtime_status()}


def _resolve_request_client(request: ChatRequest):
    """
    根据请求参数解析 LLM 客户端

    这是 QLoRA 动态切换的关键函数:
    1. 从请求中提取 target_agent_id 或 persona
    2. 调用 resolve_adapter() 获取 adapter 配置
    3. 创建指定 adapter 的 UnslothClient

    对于非 unsloth provider，直接返回全局客户端。

    Args:
        request: 聊天请求

    Returns:
        LLM 客户端实例
    """
    provider = get_config("llm.provider", "openai")
    if provider != "unsloth":
        return llm_client
    target_agent_id = request.target_agent_id or request.persona or "default"
    adapter = resolve_adapter(target_agent_id)
    return get_llm_client(
        provider="unsloth",
        adapter_id=adapter["id"],
        adapter_path=adapter["path"],
    )


def build_streaming_response(request: ChatRequest):
    """
    构建流式响应

    使用 Server-Sent Events (SSE) 格式返回流式输出。
    每个文本块以 "data: {json}\n\n" 格式发送。
    结束时发送 "[DONE]" 标记。

    Args:
        request: 聊天请求

    Returns:
        StreamingResponse 实例
    """
    async def generate():
        request_client = _resolve_request_client(request)
        try:
            for chunk in request_client.stream_chat(
                request.message,
                multimodal_inputs=request.multimodal_inputs,
            ):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield f"data: {json.dumps({'content': '[DONE]'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    聊天接口

    支持同步和流式两种模式:
    - stream=False: 同步返回完整回复
    - stream=True: 流式返回 SSE 格式

    请求示例:
        {
            "message": "你好",
            "target_agent_id": "coding_agent",
            "stream": false
        }

    Args:
        request: 聊天请求

    Returns:
        ChatResponse 或 StreamingResponse

    Raises:
        HTTPException: 模型未加载或推理错误
    """
    if llm_client is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        request_client = _resolve_request_client(request)
        if request.stream:
            return build_streaming_response(request)
        message = request_client.chat(request.message, multimodal_inputs=request.multimodal_inputs)
        return ChatResponse(message=message, intent="chat")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = get_config("server.port", 721)
    uvicorn.run(app, host="0.0.0.0", port=port)
