# storage/base.py
from abc import ABC, abstractmethod
from typing import Any, Optional

class BaseStorage(ABC):
    """存储后端抽象接口（生产级）"""
    @abstractmethod
    def __init__(self, **kwargs):
        """初始化存储后端，接收配置参数"""
        pass

    @abstractmethod
    def save(self, user_id: str, memory_type: str, data: Any) -> None:
        """保存数据"""
        pass

    @abstractmethod
    def load(self, user_id: str, memory_type: str) -> Optional[Any]:
        """加载数据"""
        pass

    @abstractmethod
    def delete(self, user_id: str, memory_type: str) -> None:
        """删除数据"""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """存储健康检查"""
        pass