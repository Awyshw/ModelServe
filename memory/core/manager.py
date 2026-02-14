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

# 重试装饰器（生产级：优化异常处理）
def retry(max_retries: int = 3, delay: float = 0.5):
    """操作重试装饰器（适配新的异常体系）"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except StorageError as e:
                    retries += 1
                    log.warning(f"操作失败（重试{retries}/{max_retries}）：{str(e)}")
                    time.sleep(delay * retries)  # 指数退避
            raise StorageError(f"操作失败，已重试{max_retries}次：{func.__name__}")
        return wrapper
    return decorator

class MemoryManager:
    """生产级记忆管理器（适配本地/OpenAI嵌入模式）"""
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.storage = self._init_storage()
        self._init_memory_components()
        log.info(
            f"MemoryManager初始化成功，user_id={user_id}，"
            f"嵌入模式={settings.EMBEDDING_MODE}，存储类型={settings.STORAGE_TYPE}"
        )

    def _init_storage(self) -> BaseStorage:
        """初始化存储后端（可插拔+降级策略）"""
        try:
            if settings.STORAGE_TYPE == "redis":
                return RedisStorage(redis_url=settings.REDIS_URL)
            elif settings.STORAGE_TYPE == "leveldb":
                return LevelDBStorage(storage_path=settings.STORAGE_PATH)
            else:  # 默认JSON存储
                return JsonStorage(storage_path=settings.STORAGE_PATH)
        except Exception as e:
            log.error(f"存储后端初始化失败，降级为JSON存储：{str(e)}")
            return JsonStorage(storage_path=settings.STORAGE_PATH)

    def _init_memory_components(self) -> None:
        """初始化记忆组件（适配新语义记忆模块）"""
        # 1. 短期记忆（无改动）
        self.episodic = EpisodicMemory(
            user_id=self.user_id,
            max_window_size=settings.EPISODIC_MAX_WINDOW,
            storage=self.storage
        )
        
        # 2. 长期记忆（核心修改：移除embedding_model_name参数）
        # 🌟 关键改动：新的SemanticMemory已通过配置自动选择本地/OpenAI嵌入，无需传模型名
        self.semantic = SemanticMemory(
            user_id=self.user_id,
            storage=self.storage  # 仅保留user_id和storage参数
        )
        
        # 3. 临时记忆（无改动）
        self.transient = TransientMemory(user_id=self.user_id)

    # ------------------------------
    # 短期记忆操作（带重试+输入验证）
    # ------------------------------
    @retry(max_retries=2)
    def add_dialogue_turn(self, user_input: str, assistant_response: str, tags: List[str] = None) -> None:
        """添加对话轮次"""
        if not isinstance(user_input, str) or len(user_input.strip()) == 0:
            raise ValidationError("用户输入不能为空且必须为非空字符串")
        if not isinstance(assistant_response, str) or len(assistant_response.strip()) == 0:
            raise ValidationError("助手回复不能为空且必须为非空字符串")
        
        self.episodic.add(key="turn", value={
            "user_input": user_input.strip(),
            "assistant_response": assistant_response.strip(),
            "tags": tags or []
        })
        log.debug(f"添加对话轮次成功，user_id={self.user_id}")

    @retry(max_retries=2)
    def get_context(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """获取最近N轮对话上下文"""
        if not isinstance(last_n, int) or last_n <= 0 or last_n > settings.EPISODIC_MAX_WINDOW:
            raise ValidationError(f"last_n必须为1-{settings.EPISODIC_MAX_WINDOW}之间的正整数")
        
        return self.episodic.get(last_n=last_n)

    # ------------------------------
    # 长期记忆操作（新增完整方法，适配新语义模块）
    # ------------------------------
    @retry(max_retries=2)
    def add_user_preference(self, key: str, value: str, confidence: float = 1.0) -> None:
        """添加/更新用户偏好"""
        if not isinstance(key, str) or len(key.strip()) == 0:
            raise ValidationError("偏好键不能为空且必须为非空字符串")
        if not isinstance(value, str) or len(value.strip()) == 0:
            raise ValidationError("偏好值不能为空且必须为非空字符串")
        if not isinstance(confidence, float) or not (0.0 <= confidence <= 1.0):
            raise ValidationError("置信度必须为0.0-1.0之间的浮点数")
        
        self.semantic.add(
            key=key.strip(),
            value=value.strip(),
            confidence=confidence
        )
        log.debug(f"添加用户偏好成功，user_id={self.user_id}，key={key.strip()}")

    @retry(max_retries=2)
    def retrieve_preferences(
        self,
        key: Optional[str] = None,
        query: Optional[str] = None,
        threshold: float = settings.SEMANTIC_SIMILARITY_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """检索用户偏好（精确查询/语义检索）"""
        # 验证参数：key和query不能同时传
        if key is not None and query is not None:
            raise ValidationError("只能选择精确查询(key)或语义检索(query)中的一种方式")
        
        # 验证阈值范围
        if not isinstance(threshold, float) or not (0.0 <= threshold <= 1.0):
            raise ValidationError("相似度阈值必须为0.0-1.0之间的浮点数")
        
        # 处理精确查询的key
        clean_key = key.strip() if isinstance(key, str) else None
        
        # 处理语义检索的query
        clean_query = query.strip() if isinstance(query, str) else None
        
        return self.semantic.get(
            key=clean_key,
            query=clean_query,
            threshold=threshold
        )

    @retry(max_retries=2)
    def delete_preference(self, key: str) -> None:
        """删除指定用户偏好"""
        if not isinstance(key, str) or len(key.strip()) == 0:
            raise ValidationError("偏好键不能为空且必须为非空字符串")
        
        self.semantic.clear(key=key.strip())
        log.debug(f"删除用户偏好成功，user_id={self.user_id}，key={key.strip()}")

    # ------------------------------
    # 临时记忆操作（新增完整方法）
    # ------------------------------
    def set_transient_data(self, key: str, value: Any) -> None:
        """设置临时数据（线程隔离）"""
        if not isinstance(key, str) or len(key.strip()) == 0:
            raise ValidationError("临时数据键不能为空且必须为非空字符串")
        
        self.transient.add(key=key.strip(), value=value)
        log.debug(f"设置临时数据成功，user_id={self.user_id}，key={key.strip()}")

    def get_transient_data(self, key: Optional[str] = None) -> Any:
        """获取临时数据（key=None返回全部）"""
        clean_key = key.strip() if isinstance(key, str) else None
        return self.transient.get(key=clean_key)

    def delete_transient_data(self, key: str) -> None:
        """删除指定临时数据"""
        if not isinstance(key, str) or len(key.strip()) == 0:
            raise ValidationError("临时数据键不能为空且必须为非空字符串")
        
        self.transient.clear(key=key.strip())
        log.debug(f"删除临时数据成功，user_id={self.user_id}，key={key.strip()}")

    # ------------------------------
    # 通用操作
    # ------------------------------
    def health_check(self) -> Dict[str, Any]:
        """健康检查（增强版：包含嵌入服务状态）"""
        try:
            # 检查嵌入服务可用性
            embed_health = True
            try:
                from core.embedding_service import get_embedding_service
                embed_service = get_embedding_service()
                # 测试嵌入生成
                embed_service.generate_embedding("health_check")
            except Exception as e:
                embed_health = False
                log.error(f"嵌入服务健康检查失败：{str(e)}")

            return {
                "storage": self.storage.health_check(),
                "episodic": True,
                "semantic": embed_health,
                "transient": True,
                "embedding_mode": settings.EMBEDDING_MODE,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            log.error(f"健康检查失败：{str(e)}")
            return {"status": "error", "message": str(e)}

    @retry(max_retries=2)
    def clear_all(self) -> None:
        """清空所有记忆（生产级：日志+原子操作）"""
        log.warning(f"执行清空所有记忆操作，user_id={self.user_id}")
        try:
            # 按顺序清空，确保原子性
            self.episodic.clear()
            self.semantic.clear()
            self.transient.clear()
            log.info(f"所有记忆清空成功，user_id={self.user_id}")
        except Exception as e:
            log.error(f"清空记忆失败，user_id={self.user_id}：{str(e)}")
            raise StorageError(f"清空记忆失败：{str(e)}")