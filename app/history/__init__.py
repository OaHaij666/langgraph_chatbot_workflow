from typing import List, Optional
from .models import MessageEntry, HistoryEntry
from .base import BaseHistoryStore
from .mysql_store import MySQLHistoryStore
from .utils import (
    ParsedMessage,
    FilterResult,
    parse_mentions,
    check_keywords,
    process_message,
    HistoryCleaner,
)
from .token_counter import count_tokens


_store: Optional[MySQLHistoryStore] = None
_current_session_id: str = "default"


def get_history_store() -> MySQLHistoryStore:
    global _store
    if _store is None:
        _store = MySQLHistoryStore()
    return _store


def set_session(session_id: str) -> None:
    global _current_session_id
    _current_session_id = session_id


def get_session() -> str:
    return _current_session_id


async def add_message(
    sender_id: str,
    content: str,
    receiver_id: str = "@all",
    role: str = "user",
    session_id: Optional[str] = None,
    token_count: int = 0,
    metadata: Optional[dict] = None,
) -> MessageEntry:
    store = get_history_store()
    actual_token_count = token_count if token_count > 0 else count_tokens(content)
    entry = MessageEntry(
        session_id=session_id or _current_session_id,
        sender_id=sender_id,
        receiver_id=receiver_id,
        role=role,
        content=content,
        token_count=actual_token_count,
        metadata=metadata or {},
    )
    return await store.add_message(entry)


async def add_user_message(
    content: str,
    receiver_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    token_count: int = 0,
    metadata: Optional[dict] = None,
    skip_filter: bool = False,
) -> Optional[MessageEntry]:
    from config import get_config
    
    parsed, filter_result = process_message(content)
    
    if not skip_filter and not filter_result.should_save:
        return None
    
    store = get_history_store()
    final_receiver = receiver_id or parsed.receiver_id
    final_content = parsed.clean_content
    actual_token_count = token_count if token_count > 0 else count_tokens(final_content)
    
    return await store.add_user_message(
        session_id=session_id or _current_session_id,
        content=final_content,
        receiver_id=final_receiver,
        user_id=user_id,
        token_count=actual_token_count,
        metadata=metadata,
    )


async def add_agent_message(
    agent_id: str,
    content: str,
    receiver_id: str = "@all",
    session_id: Optional[str] = None,
    token_count: int = 0,
    metadata: Optional[dict] = None,
) -> MessageEntry:
    store = get_history_store()
    actual_token_count = token_count if token_count > 0 else count_tokens(content)
    return await store.add_agent_message(
        session_id=session_id or _current_session_id,
        agent_id=agent_id,
        content=content,
        receiver_id=receiver_id,
        token_count=actual_token_count,
        metadata=metadata,
    )


async def get_messages(session_id: Optional[str] = None, limit: int = 100) -> List[MessageEntry]:
    store = get_history_store()
    return await store.get_session_messages(session_id or _current_session_id, limit)


async def get_recent_messages(n: int = 10, session_id: Optional[str] = None) -> List[MessageEntry]:
    store = get_history_store()
    return await store.get_recent_messages(session_id or _current_session_id, n)


async def get_messages_by_sender(
    sender_id: str,
    session_id: Optional[str] = None,
    limit: int = 100,
) -> List[MessageEntry]:
    store = get_history_store()
    return await store.get_messages_by_sender(sender_id, session_id or _current_session_id, limit)


async def get_messages_by_receiver(
    receiver_id: str,
    session_id: Optional[str] = None,
    limit: int = 100,
) -> List[MessageEntry]:
    store = get_history_store()
    return await store.get_messages_by_receiver(receiver_id, session_id or _current_session_id, limit)


async def get_conversation_between(
    sender_id: str,
    receiver_id: str,
    session_id: Optional[str] = None,
    limit: int = 100,
) -> List[MessageEntry]:
    store = get_history_store()
    return await store.get_conversation_between(sender_id, receiver_id, session_id or _current_session_id, limit)


async def clear_messages(session_id: Optional[str] = None) -> int:
    store = get_history_store()
    return await store.delete_session(session_id or _current_session_id)


async def get_stats(session_id: Optional[str] = None) -> dict:
    store = get_history_store()
    sid = session_id or _current_session_id
    return {
        "session_id": sid,
        "message_count": await store.get_session_message_count(sid),
        "token_count": await store.get_session_token_count(sid),
    }


async def start_history_cleaner() -> None:
    from config import get_config
    interval_hours = get_config("history.cleanup_interval_hours", 24)
    await HistoryCleaner.start(interval_hours)


async def stop_history_cleaner() -> None:
    await HistoryCleaner.stop()


async def cleanup_old_messages(days: Optional[int] = None) -> int:
    return await HistoryCleaner.cleanup_now(days)


__all__ = [
    "MessageEntry",
    "HistoryEntry",
    "BaseHistoryStore",
    "MySQLHistoryStore",
    "ParsedMessage",
    "FilterResult",
    "parse_mentions",
    "check_keywords",
    "process_message",
    "HistoryCleaner",
    "get_history_store",
    "set_session",
    "get_session",
    "add_message",
    "add_user_message",
    "add_agent_message",
    "get_messages",
    "get_recent_messages",
    "get_messages_by_sender",
    "get_messages_by_receiver",
    "get_conversation_between",
    "clear_messages",
    "get_stats",
    "start_history_cleaner",
    "stop_history_cleaner",
    "cleanup_old_messages",
    "count_tokens",
]
