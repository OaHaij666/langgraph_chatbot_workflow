"""
LangGraph Chat 分支

本模块实现了基于 LangGraph 的聊天处理流程，支持:
- RAG 检索增强生成
- 多 agent 动态切换（通过 QLoRA adapter）
- 流式 TTS 输出
- 会话历史管理

与 QLoRA 动态切换的关系:
- 根据 @mention 或 agent_id 解析目标 agent
- 调用 resolve_adapter() 获取 adapter 配置
- 在 stream_llm 节点创建指定 adapter 的 LLMClient

流程图:
    retrieve_rag → build_prompt → stream_llm → stream_tts → save_history → END

参考文档: QLORA_DYNAMIC_SWITCH_TECH.md
"""
from typing import TypedDict, Generator, Optional
from langgraph.graph import StateGraph, END
from app.rag import RagRetriever
from app.llm import LLMClient, resolve_adapter
from app.tts import TTSClient
from app.history import (
    get_recent_messages,
    add_user_message,
    add_agent_message,
    set_session,
    get_session,
    process_message,
)
from app.prompt import build_chat_prompt, get_system_prompt


class ChatState(TypedDict):
    """
    Chat 分支的状态定义

    状态在各个节点间传递和更新:
    - text: 原始用户输入（可能包含 @mention）
    - clean_text: 清理后的文本（去除 @mention）
    - receiver_id: @mention 的目标接收者
    - session_id: 会话 ID
    - user_id: 用户 ID
    - agent_id: 默认 agent ID
    - target_agent_id: 实际目标 agent ID（用于 adapter 选择）
    - adapter_id: QLoRA adapter ID
    - adapter_path: QLoRA adapter 路径
    - history: 会话历史
    - rag_context: RAG 检索结果
    - prompt: 构建好的提示词
    - llm_stream: LLM 流式输出生成器
    - response: 完整响应文本
    """
    text: str
    clean_text: str
    receiver_id: str
    session_id: str
    user_id: Optional[str]
    agent_id: str
    target_agent_id: str
    adapter_id: str
    adapter_path: Optional[str]
    history: list
    rag_context: list[str]
    prompt: str
    llm_stream: Generator
    response: str


async def retrieve_rag(state: ChatState) -> ChatState:
    """
    RAG 检索节点

    使用 RagRetriever 从知识库检索相关内容，
    结果存入 state["rag_context"] 用于后续提示词构建。

    Args:
        state: 当前状态

    Returns:
        更新后的状态（包含 rag_context）
    """
    retriever = RagRetriever()
    state["rag_context"] = retriever.retrieve(state["clean_text"], top_k=3)
    return state


async def build_prompt(state: ChatState) -> ChatState:
    """
    提示词构建节点

    组装完整的提示词:
    1. 获取最近的历史消息
    2. 获取目标 agent 的系统提示
    3. 调用 build_chat_prompt 组装提示词

    Args:
        state: 当前状态

    Returns:
        更新后的状态（包含 prompt）
    """
    history_entries = await get_recent_messages(10, state.get("session_id"))

    system_prompt = get_system_prompt(state.get("target_agent_id"), branch="chat")
    state["prompt"] = build_chat_prompt(
        clean_text=state["clean_text"],
        rag_context=state["rag_context"],
        history_entries=history_entries,
        system_prompt=system_prompt,
    )
    return state


async def stream_llm(state: ChatState) -> ChatState:
    """
    LLM 流式生成节点

    这是 QLoRA 动态切换的关键节点:
    1. 使用 state 中的 adapter_id 和 adapter_path 创建 LLMClient
    2. LLMClient 内部会激活对应的 adapter
    3. 执行流式生成

    Args:
        state: 当前状态

    Returns:
        更新后的状态（包含 llm_stream）
    """
    llm = LLMClient(adapter_id=state.get("adapter_id"), adapter_path=state.get("adapter_path"))
    state["llm_stream"] = llm.stream_chat(state["prompt"])
    return state


async def stream_tts(state: ChatState) -> ChatState:
    """
    TTS 流式输出节点

    将 LLM 的流式输出同时:
    1. 发送到 TTS 服务进行语音合成
    2. 拼接成完整响应

    Args:
        state: 当前状态

    Returns:
        更新后的状态（包含 response）
    """
    tts = TTSClient()
    
    full_response = ""
    for chunk in state["llm_stream"]:
        full_response += chunk
        tts.stream_speak(chunk)
    
    state["response"] = full_response
    return state


async def save_history(state: ChatState) -> ChatState:
    """
    历史保存节点

    将用户消息和 agent 回复保存到会话历史中。

    Args:
        state: 当前状态

    Returns:
        更新后的状态
    """
    await add_user_message(
        content=state["text"],
        user_id=state.get("user_id"),
        session_id=state.get("session_id"),
    )
    
    await add_agent_message(
        agent_id=state.get("target_agent_id") or state.get("agent_id", "assistant"),
        content=state.get("response", ""),
        receiver_id=state.get("user_id", "user"),
        session_id=state.get("session_id"),
    )
    
    return state


def create_chat_graph() -> StateGraph:
    """
    创建 Chat 分支的 LangGraph 图

    节点顺序:
        retrieve_rag → build_prompt → stream_llm → stream_tts → save_history → END

    Returns:
        配置好的 StateGraph 实例
    """
    graph = StateGraph(ChatState)
    
    graph.add_node("retrieve_rag", retrieve_rag)
    graph.add_node("build_prompt", build_prompt)
    graph.add_node("stream_llm", stream_llm)
    graph.add_node("stream_tts", stream_tts)
    graph.add_node("save_history", save_history)
    
    graph.set_entry_point("retrieve_rag")
    graph.add_edge("retrieve_rag", "build_prompt")
    graph.add_edge("build_prompt", "stream_llm")
    graph.add_edge("stream_llm", "stream_tts")
    graph.add_edge("stream_tts", "save_history")
    graph.add_edge("save_history", END)
    
    return graph


chat_graph = create_chat_graph().compile()


async def run_chat_branch(
    text: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    agent_id: str = "assistant",
) -> str:
    """
    运行 Chat 分支

    这是 Chat 分支的主入口，负责:
    1. 解析用户输入（提取 @mention）
    2. 确定目标 agent 和 adapter
    3. 初始化状态
    4. 执行 LangGraph 流程

    QLoRA adapter 选择逻辑:
    - 如果用户输入包含 @mention（如 "@coding_agent 你好"），
      则 target_agent_id 为被 @ 的 agent
    - 否则使用默认 agent_id
    - 通过 resolve_adapter() 将 agent_id 映射到 adapter 配置

    Args:
        text: 用户输入文本（可能包含 @mention）
        session_id: 会话 ID
        user_id: 用户 ID
        agent_id: 默认 agent ID

    Returns:
        agent 的完整回复文本
    """
    if session_id:
        set_session(session_id)
    
    parsed, filter_result = process_message(text)
    
    target_agent_id = parsed.receiver_id if parsed.receiver_id and parsed.receiver_id != "@all" else agent_id
    adapter = resolve_adapter(target_agent_id)

    state: ChatState = {
        "text": text,
        "clean_text": parsed.clean_content,
        "receiver_id": parsed.receiver_id,
        "session_id": session_id or get_session(),
        "user_id": user_id,
        "agent_id": agent_id,
        "target_agent_id": target_agent_id,
        "adapter_id": adapter["id"],
        "adapter_path": adapter["path"],
        "history": [],
        "rag_context": [],
        "prompt": "",
        "llm_stream": None,
        "response": "",
    }
    
    full_response = ""
    async for chunk in chat_graph.astream(state):
        if "stream_tts" in chunk:
            full_response = chunk["stream_tts"].get("response", "")
    
    return full_response
