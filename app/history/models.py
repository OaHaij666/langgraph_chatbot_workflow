from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
import uuid


@dataclass
class MessageEntry:
    id: Optional[int] = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender_id: str = "user"
    receiver_id: str = "@all"
    role: str = "user"
    content: str = ""
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    
    @classmethod
    def from_db_row(cls, row: dict) -> "MessageEntry":
        return cls(
            id=row["id"],
            session_id=row["session_id"],
            sender_id=row["sender_id"],
            receiver_id=row["receiver_id"],
            role=row["role"],
            content=row["content"],
            token_count=row["token_count"],
            metadata=row["metadata"] or {},
            created_at=row["created_at"],
        )
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "role": self.role,
            "content": self.content,
            "token_count": self.token_count,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
    
    def to_display_format(self) -> str:
        if self.receiver_id == "@all":
            return f"{self.sender_id}: {self.content}"
        return f"{self.sender_id} → {self.receiver_id}: {self.content}"


HistoryEntry = MessageEntry
