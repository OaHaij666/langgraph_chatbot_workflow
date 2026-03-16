from typing import TypedDict, Generator
from langgraph.graph import StateGraph, END
from app.rag import RagRetriever
from app.llm import LLMClient
from app.tts import TTSClient
from app.history import get_history, get_recent_history


class ChatState(TypedDict):
    text: str
    history: list
    rag_context: list[str]
    prompt: str
    llm_stream: Generator


def retrieve_rag(state: ChatState) -> ChatState:
    """RAG 检索节点"""
    retriever = RagRetriever()
    state["rag_context"] = retriever.retrieve(state["text"], top_k=3)
    return state


def build_prompt(state: ChatState) -> ChatState:
    """构建提示词节点"""
    history_text = get_recent_history(5)
    
    parts = []
    parts.append("系统提示：你是一个友好的AI助手。")
    
    if state["rag_context"]:
        parts.append("相关上下文：\n" + "\n".join(state["rag_context"]))
    
    if history_text:
        parts.append("对话历史：\n" + "\n".join(
            f"用户：{h['user']}\n助手：{h['assistant']}" for h in history_text
        ))
    
    parts.append(f"用户：{state['text']}")
    
    state["prompt"] = "\n\n".join(parts)
    return state


def stream_llm(state: ChatState) -> ChatState:
    """LLM 流式调用节点"""
    llm = LLMClient()
    state["llm_stream"] = llm.stream_chat(state["prompt"])
    return state


def stream_tts(state: ChatState) -> Generator[str, None, None]:
    """TTS 流式输出节点"""
    tts = TTSClient()
    
    full_response = ""
    for chunk in state["llm_stream"]:
        full_response += chunk
        tts.stream_speak(chunk)
    
    return full_response


def create_chat_graph() -> StateGraph:
    """创建聊天分支的 LangGraph"""
    graph = StateGraph(ChatState)
    
    graph.add_node("retrieve_rag", retrieve_rag)
    graph.add_node("build_prompt", build_prompt)
    graph.add_node("stream_llm", stream_llm)
    graph.add_node("stream_tts", stream_tts)
    
    graph.set_entry_point("retrieve_rag")
    graph.add_edge("retrieve_rag", "build_prompt")
    graph.add_edge("build_prompt", "stream_llm")
    graph.add_edge("stream_llm", "stream_tts")
    graph.add_edge("stream_tts", END)
    
    return graph


chat_graph = create_chat_graph().compile()


def run_chat_branch(text: str) -> Generator[str, None, None]:
    """
    聊天分支入口。
    
    流程：
    1. RAG 检索
    2. 构建提示词
    3. LLM 流式输出
    4. TTS 流式播放
    
    Args:
        text: 用户输入文本
    
    Yields:
        流式返回的文本片段
    """
    state: ChatState = {
        "text": text,
        "history": get_history(),
        "rag_context": [],
        "prompt": "",
        "llm_stream": None
    }
    
    for chunk in chat_graph.stream(state):
        if "stream_tts" in chunk:
            yield chunk["stream_tts"]
