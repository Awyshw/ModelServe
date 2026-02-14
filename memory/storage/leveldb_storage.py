# storage/leveldb_storage.py
from memory.storage.base import BaseStorage
import plyvel
import json
from typing import Any, Optional
from utils.logger import log
from memory.utils.exceptions import StorageError

class LevelDBStorage(BaseStorage):
    """LevelDB存储后端（生产级）"""
    def __init__(self, storage_path: str = "./leveldb_data", **kwargs):
        self.storage_path = storage_path
        self.db = None
        self._init_db()

    def _init_db(self) -> None:
        """初始化LevelDB连接"""
        try:
            self.db = plyvel.DB(self.storage_path, create_if_missing=True)
            log.info(f"LevelDB存储初始化成功，路径：{self.storage_path}")
        except Exception as e:
            log.error(f"LevelDB初始化失败：{str(e)}")
            raise StorageError(f"存储初始化失败：{str(e)}")

    def _get_key(self, user_id: str, memory_type: str) -> bytes:
        """生成LevelDB键"""
        return f"memory:{user_id}:{memory_type}".encode("utf-8")

    def save(self, user_id: str, memory_type: str, data: Any) -> None:
        """保存数据（JSON序列化）"""
        try:
            key = self._get_key(user_id, memory_type)
            value = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.db.put(key, value)
            log.debug(f"LevelDB保存数据成功：user_id={user_id}, type={memory_type}")
        except Exception as e:
            log.error(f"LevelDB保存失败：{str(e)}")
            raise StorageError(f"数据保存失败：{str(e)}")

    def load(self, user_id: str, memory_type: str) -> Optional[Any]:
        """加载数据"""
        try:
            key = self._get_key(user_id, memory_type)
            value = self.db.get(key)
            if not value:
                return None
            return json.loads(value.decode("utf-8"))
        except Exception as e:
            log.error(f"LevelDB加载失败：{str(e)}")
            raise StorageError(f"数据加载失败：{str(e)}")

    def delete(self, user_id: str, memory_type: str) -> None:
        """删除数据"""
        try:
            key = self._get_key(user_id, memory_type)
            self.db.delete(key)
            log.debug(f"LevelDB删除数据成功：user_id={user_id}, type={memory_type}")
        except Exception as e:
            log.error(f"LevelDB删除失败：{str(e)}")
            raise StorageError(f"数据删除失败：{str(e)}")

    def health_check(self) -> bool:
        """健康检查"""
        try:
            test_key = b"health_check"
            self.db.put(test_key, b"ok")
            self.db.delete(test_key)
            return True
        except Exception:
            return False

    def __del__(self):
        """析构函数，关闭连接"""
        if self.db:
            self.db.close()
            log.info("LevelDB连接已关闭")