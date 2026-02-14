# storage/redis_storage.py
from storage.base import BaseStorage
import redis
import json
from typing import Any, Optional
from utils.logger import log
from utils.exceptions import StorageError
from config.settings import settings

class RedisStorage(BaseStorage):
    """Redis存储后端（生产级：连接池+超时+重试）"""
    def __init__(self, redis_url: str = settings.REDIS_URL, **kwargs):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self._init_redis()
        log.info(f"RedisStorage初始化成功，连接地址：{redis_url}")

    def _init_redis(self) -> None:
        """初始化Redis连接（连接池+超时配置）"""
        try:
            # 生产级Redis配置
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=False,  # 二进制数据更高效
                socket_timeout=5,        # 连接超时5秒
                socket_connect_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30  # 健康检查间隔30秒
            )
            
            # 测试连接
            self.redis_client.ping()
            log.info("Redis连接测试成功")
            
        except Exception as e:
            log.error(f"Redis初始化失败：{str(e)}", exc_info=True)
            raise StorageError(f"Redis连接失败：{str(e)}")

    def _get_key(self, user_id: str, memory_type: str) -> bytes:
        """生成Redis键（二进制，提高性能）"""
        return f"openclaw:memory:{user_id}:{memory_type}".encode("utf-8")

    def save(self, user_id: str, memory_type: str, data: Any) -> None:
        """保存数据到Redis（JSON序列化）"""
        try:
            if not self.redis_client:
                raise StorageError("Redis客户端未初始化")
            
            key = self._get_key(user_id, memory_type)
            # 序列化数据（处理特殊类型）
            value = json.dumps(
                data,
                ensure_ascii=False,
                default=str
            ).encode("utf-8")
            
            # 设置过期时间（可选：短期记忆7天，长期记忆永久）
            expire_seconds = 60 * 60 * 24 * 7 if memory_type == "episodic" else None
            
            if expire_seconds:
                self.redis_client.setex(key, expire_seconds, value)
            else:
                self.redis_client.set(key, value)
            
            log.debug(f"Redis存储保存成功，user_id={user_id}，type={memory_type}，key={key.decode('utf-8')}")
            
        except Exception as e:
            log.error(f"Redis存储保存失败：{str(e)}", exc_info=True)
            raise StorageError(f"Redis数据保存失败：{str(e)}")

    def load(self, user_id: str, memory_type: str) -> Optional[Any]:
        """从Redis加载数据"""
        try:
            if not self.redis_client:
                raise StorageError("Redis客户端未初始化")
            
            key = self._get_key(user_id, memory_type)
            value = self.redis_client.get(key)
            
            if not value:
                log.debug(f"Redis键不存在，user_id={user_id}，type={memory_type}")
                return None
            
            # 反序列化
            data = json.loads(value.decode("utf-8"))
            log.debug(f"Redis存储加载成功，user_id={user_id}，type={memory_type}")
            return data
            
        except json.JSONDecodeError as e:
            log.error(f"Redis数据解析失败：{str(e)}", exc_info=True)
            raise StorageError(f"Redis数据解析失败：{str(e)}")
        except Exception as e:
            log.error(f"Redis存储加载失败：{str(e)}", exc_info=True)
            raise StorageError(f"Redis数据加载失败：{str(e)}")

    def delete(self, user_id: str, memory_type: str) -> None:
        """删除Redis中的数据"""
        try:
            if not self.redis_client:
                raise StorageError("Redis客户端未初始化")
            
            key = self._get_key(user_id, memory_type)
            deleted = self.redis_client.delete(key)
            
            if deleted:
                log.debug(f"Redis数据删除成功，user_id={user_id}，type={memory_type}")
            else:
                log.debug(f"Redis键不存在，无需删除，user_id={user_id}，type={memory_type}")
                
        except Exception as e:
            log.error(f"Redis存储删除失败：{str(e)}", exc_info=True)
            raise StorageError(f"Redis数据删除失败：{str(e)}")

    def health_check(self) -> bool:
        """Redis健康检查"""
        try:
            if not self.redis_client:
                return False
            return self.redis_client.ping()
        except Exception:
            return False

    def __del__(self):
        """析构函数：关闭Redis连接"""
        if self.redis_client:
            try:
                self.redis_client.close()
                log.info("Redis连接已关闭")
            except Exception as e:
                log.error(f"Redis连接关闭失败：{str(e)}")