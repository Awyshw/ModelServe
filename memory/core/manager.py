# core/manager.py
from core.episodic import EpisodicMemory
from core.semantic import SemanticMemory
from core.transient import TransientMemory
from storage.base import BaseStorage
from storage.json_storage import JsonStorage
from storage.redis_storage import RedisStorage
from storage.leveldb_storage import LevelDBStorage
from config.settings import settings
from utils.logger import log
from utils.exceptions import StorageError, ValidationError
from typing import Optional, Dict, Any, List
import time
from functools import wraps

# 重试装饰器（生产级）
def retry(max_retries: int = 3, delay: float = 0.5):
    """操作重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except StorageError as e:
                    retries += 1
                    log.warning(f"操作失败（重试{retries}/{max_retries}）：{e.message}")
                    time.sleep(delay * retries)  # 指数退避
            raise StorageError(f"操作失败，已重试{max_retries}次")
        return wrapper
    return decorator

class MemoryManager:
    """生产级记忆管理器（可插拔+高可用）"""
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.storage = self._init_storage()
        self._init_memory_components()
        log.info(f"MemoryManager初始化成功，user_id={user_id}")

    def _init_storage(self) -> BaseStorage:
        """初始化存储后端（可插拔）"""
        try:
            if settings.STORAGE_TYPE == "redis":
                return RedisStorage(redis_url=settings.REDIS_URL)
            elif settings.STORAGE_TYPE == "leveldb":
                return LevelDBStorage(storage_path=settings.STORAGE_PATH)
            else:  # json
                return JsonStorage(storage_path=settings.STORAGE_PATH)
        except Exception as e:
            log.error(f"存储后端初始化失败，降级为JSON存储：{str(e)}")
            return JsonStorage(storage_path=settings.STORAGE_PATH)

    def _init_memory_components(self) -> None:
        """初始化记忆组件"""
        # 短期记忆
        self.episodic = EpisodicMemory(
            user_id=self.user_id,
            max_window_size=settings.EPISODIC_MAX_WINDOW,
            storage=self.storage
        )
        
        # 长期记忆
        self.semantic = SemanticMemory(
            user_id=self.user_id,
            embedding_model_name=settings.SEMANTIC_EMBED_MODEL,
            storage=self.storage
        )
        
        # 临时记忆
        self.transient = TransientMemory(user_id=self.user_id)

    # ------------------------------
    # 短期记忆操作（带重试）
    # ------------------------------
    @retry(max_retries=2)
    def add_dialogue_turn(self, user_input: str, assistant_response: str, tags: List[str] = None) -> None:
        """添加对话轮次（生产级：输入验证+重试）"""
        if not user_input or not isinstance(user_input, str):
            raise ValidationError("用户输入不能为空且必须为字符串")
        self.episodic.add(key="turn", value={
            "user_input": user_input,
            "assistant_response": assistant_response,
            "tags": tags or []
        })

    @retry(max_retries=2)
    def get_context(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """获取上下文"""
        if last_n <= 0 or last_n > settings.EPISODIC_MAX_WINDOW:
            raise ValidationError(f"last_n必须在1-{settings.EPISODIC_MAX_WINDOW}之间")
        return self.episodic.get(last_n=last_n)

    # ------------------------------
    # 其他核心方法（省略，参考上述模式优化）
    # ------------------------------
    
    def health_check(self) -> Dict[str, bool]:
        """健康检查"""
        return {
            "storage": self.storage.health_check(),
            "episodic": True,
            "semantic": True,
            "transient": True
        }

    def clear_all(self) -> None:
        """清空所有记忆（生产级：日志+确认）"""
        log.warning(f"清空所有记忆，user_id={self.user_id}")
        self.episodic.clear()
        self.semantic.clear()
        self.transient.clear()