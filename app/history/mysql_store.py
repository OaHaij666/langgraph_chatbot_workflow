"""
MySQL 历史记录存储实现

设计说明:
========

本模块实现了基于 MySQL 的聊天历史记录存储，支持多 Agent 群聊场景。

核心功能:
--------
1. 消息存储: 支持用户和 Agent 消息的持久化存储
2. 会话查询: 按 session_id 获取完整会话历史
3. 发送者/接收者查询: 支持按发送者或接收者筛选消息
4. 对话查询: 获取两个实体之间的对话记录
5. 自动清理: 支持按时间清理过期消息

多 Agent 群聊支持:
----------------
- receiver_id = '@all': 群发消息，所有 Agent 可见
- receiver_id = 特定ID: 私聊消息，仅指定接收者可见
- metadata 字段: 可存储 @mentioned_users 列表

使用示例:
--------
    store = MySQLHistoryStore()
    
    # 添加用户消息
    await store.add_user_message(
        session_id="session_123",
        content="@agent1 你好",
        receiver_id="@all",
        user_id="user_001"
    )
    
    # 添加 Agent 回复
    await store.add_agent_message(
        session_id="session_123",
        agent_id="agent1",
        content="你好！有什么可以帮助你的？",
        receiver_id="user_001"
    )
    
    # 获取会话历史
    messages = await store.get_session_messages("session_123")
"""

import json
from typing import List, Optional
from datetime import datetime, timedelta
from app.db import get_db_pool
from .base import BaseHistoryStore
from .models import MessageEntry


class MySQLHistoryStore(BaseHistoryStore):
    
    async def add_message(self, entry: MessageEntry) -> MessageEntry:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO chat_messages 
                    (session_id, sender_id, receiver_id, role, content, token_count, metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    entry.session_id,
                    entry.sender_id,
                    entry.receiver_id,
                    entry.role,
                    entry.content,
                    entry.token_count,
                    json.dumps(entry.metadata) if entry.metadata else None,
                    entry.created_at or datetime.now(),
                )
                entry.id = cur.lastrowid
                return entry
    
    async def get_session_messages(
        self, 
        session_id: str, 
        limit: int = 100
    ) -> List[MessageEntry]:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    """
                    SELECT * FROM chat_messages 
                    WHERE session_id = %s 
                    ORDER BY created_at ASC 
                    LIMIT %s
                    """,
                    (session_id, limit),
                )
                rows = await cur.fetchall()
                return [MessageEntry.from_db_row(dict(row)) for row in rows]
    
    async def get_recent_messages(
        self, 
        session_id: str, 
        n: int = 10
    ) -> List[MessageEntry]:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    """
                    SELECT * FROM chat_messages 
                    WHERE session_id = %s 
                    ORDER BY created_at DESC 
                    LIMIT %s
                    """,
                    (session_id, n),
                )
                rows = await cur.fetchall()
                return [MessageEntry.from_db_row(dict(row)) for row in reversed(rows)]
    
    async def get_messages_by_sender(
        self, 
        sender_id: str, 
        session_id: Optional[str] = None,
        limit: int = 100
    ) -> List[MessageEntry]:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                if session_id:
                    await cur.execute(
                        """
                        SELECT * FROM chat_messages 
                        WHERE sender_id = %s AND session_id = %s
                        ORDER BY created_at DESC 
                        LIMIT %s
                        """,
                        (sender_id, session_id, limit),
                    )
                else:
                    await cur.execute(
                        """
                        SELECT * FROM chat_messages 
                        WHERE sender_id = %s
                        ORDER BY created_at DESC 
                        LIMIT %s
                        """,
                        (sender_id, limit),
                    )
                rows = await cur.fetchall()
                return [MessageEntry.from_db_row(dict(row)) for row in rows]
    
    async def get_messages_by_receiver(
        self, 
        receiver_id: str, 
        session_id: Optional[str] = None,
        limit: int = 100
    ) -> List[MessageEntry]:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                if session_id:
                    await cur.execute(
                        """
                        SELECT * FROM chat_messages 
                        WHERE receiver_id = %s AND session_id = %s
                        ORDER BY created_at DESC 
                        LIMIT %s
                        """,
                        (receiver_id, session_id, limit),
                    )
                else:
                    await cur.execute(
                        """
                        SELECT * FROM chat_messages 
                        WHERE receiver_id = %s
                        ORDER BY created_at DESC 
                        LIMIT %s
                        """,
                        (receiver_id, limit),
                    )
                rows = await cur.fetchall()
                return [MessageEntry.from_db_row(dict(row)) for row in rows]
    
    async def get_conversation_between(
        self,
        sender_id: str,
        receiver_id: str,
        session_id: Optional[str] = None,
        limit: int = 100
    ) -> List[MessageEntry]:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                if session_id:
                    await cur.execute(
                        """
                        SELECT * FROM chat_messages 
                        WHERE ((sender_id = %s AND receiver_id = %s) OR (sender_id = %s AND receiver_id = %s))
                        AND session_id = %s
                        ORDER BY created_at ASC 
                        LIMIT %s
                        """,
                        (sender_id, receiver_id, receiver_id, sender_id, session_id, limit),
                    )
                else:
                    await cur.execute(
                        """
                        SELECT * FROM chat_messages 
                        WHERE (sender_id = %s AND receiver_id = %s) OR (sender_id = %s AND receiver_id = %s)
                        ORDER BY created_at ASC 
                        LIMIT %s
                        """,
                        (sender_id, receiver_id, receiver_id, sender_id, limit),
                    )
                rows = await cur.fetchall()
                return [MessageEntry.from_db_row(dict(row)) for row in rows]
    
    async def delete_session(self, session_id: str) -> int:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM chat_messages WHERE session_id = %s",
                    (session_id,),
                )
                return cur.rowcount
    
    async def delete_old_messages(self, days: int = 7) -> int:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                cutoff = datetime.now() - timedelta(days=days)
                await cur.execute(
                    "DELETE FROM chat_messages WHERE created_at < %s",
                    (cutoff,),
                )
                return cur.rowcount
    
    async def get_session_token_count(self, session_id: str) -> int:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT COALESCE(SUM(token_count), 0) 
                    FROM chat_messages 
                    WHERE session_id = %s
                    """,
                    (session_id,),
                )
                result = await cur.fetchone()
                return result[0] if result else 0
    
    async def get_session_message_count(self, session_id: str) -> int:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT COUNT(*) FROM chat_messages WHERE session_id = %s",
                    (session_id,),
                )
                result = await cur.fetchone()
                return result[0] if result else 0
    
    async def add_user_message(
        self,
        session_id: str,
        content: str,
        receiver_id: str = "@all",
        user_id: Optional[str] = None,
        token_count: int = 0,
        metadata: Optional[dict] = None,
    ) -> MessageEntry:
        entry = MessageEntry(
            session_id=session_id,
            sender_id=user_id or "user",
            receiver_id=receiver_id,
            role="user",
            content=content,
            token_count=token_count,
            metadata=metadata or {},
        )
        return await self.add_message(entry)
    
    async def add_agent_message(
        self,
        session_id: str,
        agent_id: str,
        content: str,
        receiver_id: str = "@all",
        token_count: int = 0,
        metadata: Optional[dict] = None,
    ) -> MessageEntry:
        entry = MessageEntry(
            session_id=session_id,
            sender_id=agent_id,
            receiver_id=receiver_id,
            role="agent",
            content=content,
            token_count=token_count,
            metadata=metadata or {},
        )
        return await self.add_message(entry)
