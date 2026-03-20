"""
MySQL 数据库管理模块

数据库设计说明:
================

表结构: chat_messages
--------------------
用于存储多 Agent 群聊系统的消息记录。

字段说明:
- id: 主键，自增整数
- session_id: 会话标识符，用于区分不同的聊天会话
- sender_id: 消息发送者标识（用户ID或Agent ID）
- receiver_id: 消息接收者标识（支持 @all 表示群发，或特定用户/Agent ID）
- role: 消息角色（user/agent/system）
- content: 消息内容
- token_count: Token 计数，用于上下文长度控制
- metadata: 元数据（JSON格式），存储额外信息如 @mention 列表、附件等
- created_at: 消息创建时间

索引设计:
- idx_session_created: 按 session_id 和 created_at 查询（获取会话消息列表）
- idx_sender_created: 按 sender_id 和 created_at 查询（获取某发送者的消息）
- idx_receiver_created: 按 receiver_id 和 created_at 查询（获取某接收者的消息）

多 Agent 群聊设计:
-----------------
1. 群聊模式: receiver_id = '@all' 表示消息对所有人可见
2. 私聊模式: receiver_id = 特定用户/Agent ID，表示定向消息
3. @mention: metadata 中可存储 mentioned_users 列表

历史记录清理:
------------
- cleanup_days 配置项控制自动清理天数
- 可通过 delete_old_messages 方法手动清理

敏感词过滤:
----------
- blocked_keywords 配置项定义敏感词列表
- 消息入库前应进行关键词检测
"""

import aiomysql
from typing import Optional
from config import get_config


class DatabasePool:
    _pool: Optional[aiomysql.Pool] = None
    
    @classmethod
    async def get_pool(cls) -> aiomysql.Pool:
        if cls._pool is None:
            cls._pool = await cls._create_pool()
        return cls._pool
    
    @classmethod
    async def _create_pool(cls) -> aiomysql.Pool:
        db_config = get_config("database")
        return await aiomysql.create_pool(
            host=db_config.get("host", "localhost"),
            port=db_config.get("port", 3306),
            db=db_config.get("database", "feifeina"),
            user=db_config.get("user", "root"),
            password=db_config.get("password", ""),
            minsize=db_config.get("pool_size", 5),
            maxsize=db_config.get("pool_size", 5) + db_config.get("max_overflow", 10),
            autocommit=True,
            charset="utf8mb4",
        )
    
    @classmethod
    async def close(cls) -> None:
        if cls._pool is not None:
            cls._pool.close()
            await cls._pool.wait_closed()
            cls._pool = None
    
    @classmethod
    async def init_tables(cls) -> None:
        pool = await cls.get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        session_id VARCHAR(64) NOT NULL COMMENT '会话标识符',
                        sender_id VARCHAR(64) NOT NULL COMMENT '消息发送者ID',
                        receiver_id VARCHAR(64) NOT NULL COMMENT '消息接收者ID，@all表示群发',
                        role VARCHAR(16) NOT NULL COMMENT '消息角色: user/agent/system',
                        content TEXT NOT NULL COMMENT '消息内容',
                        token_count INT DEFAULT 0 COMMENT 'Token计数',
                        metadata JSON COMMENT '元数据JSON',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
                        INDEX idx_session_created (session_id, created_at),
                        INDEX idx_sender_created (sender_id, created_at),
                        INDEX idx_receiver_created (receiver_id, created_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                    COMMENT='多Agent群聊消息记录表'
                """)


async def get_db_pool() -> aiomysql.Pool:
    return await DatabasePool.get_pool()


async def init_database() -> None:
    await DatabasePool.init_tables()


async def close_database() -> None:
    await DatabasePool.close()
