# core/transient.py
from core.abstractions import BaseMemory
from typing import Dict, Any, Optional
import threading
from utils.logger import log
from utils.exceptions import ValidationError

class TransientMemory(BaseMemory):
    """临时记忆：线程局部存储（会话/请求级，生产级）"""
    # 线程局部存储：每个线程独立的缓存空间
    _thread_local = threading.local()

    def __init__(self, user_id: str, storage: Optional[Any] = None):
        # 临时记忆不持久化，storage参数仅为适配接口
        super().__init__(user_id=user_id, storage=None)
        
        # 初始化当前线程的用户缓存
        self._init_thread_cache()
        log.debug(f"TransientMemory初始化成功，user_id={user_id}，线程ID={threading.get_ident()}")

    def _init_thread_cache(self) -> None:
        """初始化当前线程的缓存结构"""
        if not hasattr(self._thread_local, "cache"):
            self._thread_local.cache = {}
        
        if self.user_id not in self._thread_local.cache:
            self._thread_local.cache[self.user_id] = {}

    def add(self, key: str, value: Any, **kwargs) -> None:
        """添加临时数据（线程隔离）"""
        try:
            if not isinstance(key, str) or len(key.strip()) == 0:
                raise ValidationError("临时数据键不能为空")
            
            self._init_thread_cache()  # 确保缓存结构存在
            with self.lock:
                self._thread_local.cache[self.user_id][key] = value
                log.debug(f"添加临时数据成功，user_id={self.user_id}，key={key}，线程ID={threading.get_ident()}")
                
        except Exception as e:
            log.error(f"添加临时数据失败：{str(e)}", exc_info=True)
            raise

    def get(self, key: Optional[str] = None, **kwargs) -> Any:
        """获取临时数据（key=None返回全部）"""
        try:
            self._init_thread_cache()
            with self.lock:
                user_cache = self._thread_local.cache.get(self.user_id, {})
                
                if key is None:
                    # 返回当前用户的所有临时数据（深拷贝避免外部修改）
                    result = dict(user_cache)
                    log.debug(f"获取所有临时数据成功，user_id={self.user_id}，共{len(result)}条")
                    return result
                else:
                    if key not in user_cache:
                        log.warning(f"临时数据键不存在，user_id={self.user_id}，key={key}")
                        return None
                    
                    result = user_cache[key]
                    log.debug(f"获取临时数据成功，user_id={self.user_id}，key={key}")
                    return result
                
        except Exception as e:
            log.error(f"获取临时数据失败：{str(e)}", exc_info=True)
            raise

    def clear(self, key: Optional[str] = None) -> None:
        """清空临时数据（key=None清空当前用户所有数据）"""
        try:
            self._init_thread_cache()
            with self.lock:
                user_cache = self._thread_local.cache.get(self.user_id, {})
                
                if key:
                    if key in user_cache:
                        del user_cache[key]
                        log.debug(f"删除临时数据成功，user_id={self.user_id}，key={key}")
                    else:
                        raise ValidationError(f"临时数据键不存在：{key}")
                else:
                    user_cache.clear()
                    log.warning(f"清空所有临时数据成功，user_id={self.user_id}，线程ID={threading.get_ident()}")
                    
        except Exception as e:
            log.error(f"清空临时数据失败：{str(e)}", exc_info=True)
            raise

    def _save_to_storage(self) -> None:
        """临时记忆不持久化，空实现"""
        pass

    def _load_from_storage(self) -> None:
        """临时记忆不加载，空实现"""
        pass