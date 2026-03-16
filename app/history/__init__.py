class HistoryManager:
    """
    历史上下文管理器，提供对话历史的存取接口。
    """
    
    def __init__(self):
        self._history: list = []
    
    def add(self, user_text: str, assistant_text: str) -> None:
        """
        插入一条对话记录。
        
        Args:
            user_text: 用户输入
            assistant_text: 助手回复
        """
        self._history.append({
            "user": user_text,
            "assistant": assistant_text
        })
    
    def get_all(self) -> list:
        """
        获取所有历史记录。
        
        Returns:
            对话历史列表
        """
        return self._history.copy()
    
    def get_recent(self, n: int = 10) -> list:
        """
        获取最近 n 条历史记录。
        
        Args:
            n: 返回最近 n 条
        
        Returns:
            最近 n 条对话记录
        """
        return self._history[-n:]
    
    def clear(self) -> None:
        """清空历史记录"""
        self._history.clear()


_history_manager = HistoryManager()


def add_history(user_text: str, assistant_text: str) -> None:
    """插入对话历史"""
    _history_manager.add(user_text, assistant_text)


def get_history() -> list:
    """获取所有历史"""
    return _history_manager.get_all()


def get_recent_history(n: int = 10) -> list:
    """获取最近 n 条历史"""
    return _history_manager.get_recent(n)
