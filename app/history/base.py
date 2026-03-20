from abc import ABC, abstractmethod
from typing import List, Optional
from .models import MessageEntry


class BaseHistoryStore(ABC):
    
    @abstractmethod
    async def add_message(self, entry: MessageEntry) -> MessageEntry:
        pass
    
    @abstractmethod
    async def get_session_messages(
        self, 
        session_id: str, 
        limit: int = 100
    ) -> List[MessageEntry]:
        pass
    
    @abstractmethod
    async def get_recent_messages(
        self, 
        session_id: str, 
        n: int = 10
    ) -> List[MessageEntry]:
        pass
    
    @abstractmethod
    async def get_messages_by_sender(
        self, 
        sender_id: str, 
        session_id: Optional[str] = None,
        limit: int = 100
    ) -> List[MessageEntry]:
        pass
    
    @abstractmethod
    async def get_messages_by_receiver(
        self, 
        receiver_id: str, 
        session_id: Optional[str] = None,
        limit: int = 100
    ) -> List[MessageEntry]:
        pass
    
    @abstractmethod
    async def get_conversation_between(
        self,
        sender_id: str,
        receiver_id: str,
        session_id: Optional[str] = None,
        limit: int = 100
    ) -> List[MessageEntry]:
        pass
    
    @abstractmethod
    async def delete_session(self, session_id: str) -> int:
        pass
    
    @abstractmethod
    async def delete_old_messages(self, days: int = 7) -> int:
        pass
    
    @abstractmethod
    async def get_session_token_count(self, session_id: str) -> int:
        pass
    
    @abstractmethod
    async def get_session_message_count(self, session_id: str) -> int:
        pass
