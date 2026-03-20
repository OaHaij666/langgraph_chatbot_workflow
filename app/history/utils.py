import re
import asyncio
from typing import List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from config import get_config


@dataclass
class ParsedMessage:
    clean_content: str
    receiver_id: str
    mentions: List[str]


@dataclass
class FilterResult:
    should_save: bool
    matched_keywords: List[str]


DEFAULT_BLOCKED_KEYWORDS = [
    "密码",
    "password",
    "token",
    "secret",
    "api_key",
    "apikey",
    "private_key",
    "私钥",
    "银行卡",
    "身份证",
]


def parse_mentions(text: str) -> ParsedMessage:
    mention_pattern = r'@(\w+)'
    mentions = re.findall(mention_pattern, text)
    
    clean_content = re.sub(mention_pattern, '', text).strip()
    clean_content = re.sub(r'\s+', ' ', clean_content)
    
    receiver_id = mentions[0] if mentions else "@all"
    
    return ParsedMessage(
        clean_content=clean_content,
        receiver_id=receiver_id,
        mentions=mentions,
    )


def check_keywords(
    text: str, 
    blocked_keywords: Optional[List[str]] = None
) -> FilterResult:
    keywords = blocked_keywords or DEFAULT_BLOCKED_KEYWORDS
    keywords = keywords or []
    
    matched = []
    text_lower = text.lower()
    
    for keyword in keywords:
        if keyword.lower() in text_lower:
            matched.append(keyword)
    
    return FilterResult(
        should_save=len(matched) == 0,
        matched_keywords=matched,
    )


def process_message(
    text: str,
    blocked_keywords: Optional[List[str]] = None,
) -> Tuple[ParsedMessage, FilterResult]:
    parsed = parse_mentions(text)
    filter_result = check_keywords(parsed.clean_content, blocked_keywords)
    return parsed, filter_result


class HistoryCleaner:
    _task: Optional[asyncio.Task] = None
    _running: bool = False
    
    @classmethod
    async def start(cls, interval_hours: int = 24) -> None:
        if cls._running:
            return
        
        cls._running = True
        cls._task = asyncio.create_task(cls._cleanup_loop(interval_hours))
    
    @classmethod
    async def stop(cls) -> None:
        cls._running = False
        if cls._task:
            cls._task.cancel()
            try:
                await cls._task
            except asyncio.CancelledError:
                pass
            cls._task = None
    
    @classmethod
    async def _cleanup_loop(cls, interval_hours: int) -> None:
        from .postgres_store import PostgresHistoryStore
        
        store = PostgresHistoryStore()
        cleanup_days = get_config("history.cleanup_days", 7)
        
        while cls._running:
            try:
                deleted = await store.delete_old_messages(cleanup_days)
                if deleted > 0:
                    print(f"[HistoryCleaner] Cleaned {deleted} old messages")
            except Exception as e:
                print(f"[HistoryCleaner] Error: {e}")
            
            await asyncio.sleep(interval_hours * 3600)
    
    @classmethod
    async def cleanup_now(cls, days: Optional[int] = None) -> int:
        from .postgres_store import PostgresHistoryStore
        
        store = PostgresHistoryStore()
        cleanup_days = days or get_config("history.cleanup_days", 7)
        return await store.delete_old_messages(cleanup_days)
