# core/abstractions.py
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List
import threading

class BaseMemory(ABC):
    """所有记忆组件的抽象基类（生产级）"""
    def __init__(self, user_id: str, storage: Optional[Any] = None):
        self.user_id = user_id
        self.storage = storage
        self.lock = threading.Lock()  # 线程安全锁

    @abstractmethod
    def add(self, key: str, value: Any, **kwargs) -> None:
        """添加记忆数据"""
        pass

    @abstractmethod
    def get(self, key: Optional[str] = None, **kwargs) -> Any:
        """获取记忆数据"""
        pass

    @abstractmethod
    def clear(self, key: Optional[str] = None) -> None:
        """清空记忆数据"""
        pass

    @abstractmethod
    def _save_to_storage(self) -> None:
        """持久化到存储后端（内部方法）"""
        pass

    @abstractmethod
    def _load_from_storage(self) -> None:
        """从存储后端加载（内部方法）"""
        pass