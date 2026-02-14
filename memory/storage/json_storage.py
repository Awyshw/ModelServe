# storage/json_storage.py
from storage.base import BaseStorage
import json
import os
import shutil
from typing import Any, Optional
from utils.logger import log
from utils.exceptions import StorageError

class JsonStorage(BaseStorage):
    """JSON文件存储后端（生产级，支持多用户/多类型）"""
    def __init__(self, storage_path: str = "./storage_data", **kwargs):
        self.storage_path = storage_path
        self._init_storage()
        log.info(f"JsonStorage初始化成功，存储路径：{self.storage_path}")

    def _init_storage(self) -> None:
        """初始化存储目录"""
        try:
            os.makedirs(self.storage_path, exist_ok=True)
        except Exception as e:
            log.error(f"JSON存储目录初始化失败：{str(e)}", exc_info=True)
            raise StorageError(f"存储目录创建失败：{str(e)}")

    def _get_user_dir(self, user_id: str) -> str:
        """获取用户专属目录"""
        user_dir = os.path.join(self.storage_path, user_id)
        os.makedirs(user_dir, exist_ok=True)
        return user_dir

    def _get_file_path(self, user_id: str, memory_type: str) -> str:
        """生成文件路径：storage_path/user_id/memory_type.json"""
        user_dir = self._get_user_dir(user_id)
        return os.path.join(user_dir, f"{memory_type}.json")

    def save(self, user_id: str, memory_type: str, data: Any) -> None:
        """保存数据到JSON文件（生产级：原子写入）"""
        try:
            file_path = self._get_file_path(user_id, memory_type)
            temp_file_path = f"{file_path}.tmp"
            
            # 1. 先写入临时文件（避免文件损坏）
            with open(temp_file_path, "w", encoding="utf-8") as f:
                json.dump(
                    data,
                    f,
                    ensure_ascii=False,
                    indent=2,
                    default=str  # 处理datetime等特殊类型
                )
            
            # 2. 原子替换目标文件
            shutil.move(temp_file_path, file_path)
            
            log.debug(f"JSON存储保存成功，user_id={user_id}，type={memory_type}，路径={file_path}")
            
        except Exception as e:
            log.error(f"JSON存储保存失败：{str(e)}", exc_info=True)
            # 清理临时文件
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise StorageError(f"JSON数据保存失败：{str(e)}")

    def load(self, user_id: str, memory_type: str) -> Optional[Any]:
        """从JSON文件加载数据"""
        try:
            file_path = self._get_file_path(user_id, memory_type)
            if not os.path.exists(file_path):
                log.debug(f"JSON文件不存在，user_id={user_id}，type={memory_type}")
                return None
            
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            log.debug(f"JSON存储加载成功，user_id={user_id}，type={memory_type}")
            return data
            
        except json.JSONDecodeError as e:
            log.error(f"JSON文件解析失败：{str(e)}，路径={file_path}", exc_info=True)
            raise StorageError(f"JSON数据解析失败：{str(e)}")
        except Exception as e:
            log.error(f"JSON存储加载失败：{str(e)}", exc_info=True)
            raise StorageError(f"JSON数据加载失败：{str(e)}")

    def delete(self, user_id: str, memory_type: str) -> None:
        """删除指定存储文件（清理空目录）"""
        try:
            file_path = self._get_file_path(user_id, memory_type)
            if os.path.exists(file_path):
                os.remove(file_path)
                log.debug(f"JSON文件删除成功，user_id={user_id}，type={memory_type}")
            
            # 清理空用户目录
            user_dir = self._get_user_dir(user_id)
            if os.path.exists(user_dir) and not os.listdir(user_dir):
                os.rmdir(user_dir)
                log.debug(f"空用户目录已清理，user_id={user_id}")
                
        except Exception as e:
            log.error(f"JSON存储删除失败：{str(e)}", exc_info=True)
            raise StorageError(f"JSON数据删除失败：{str(e)}")

    def health_check(self) -> bool:
        """健康检查：验证目录可读写"""
        try:
            # 创建测试文件
            test_file = os.path.join(self.storage_path, ".health_check")
            with open(test_file, "w") as f:
                f.write("ok")
            
            # 读取测试文件
            with open(test_file, "r") as f:
                if f.read() != "ok":
                    return False
            
            # 删除测试文件
            os.remove(test_file)
            return True
            
        except Exception as e:
            log.error(f"JSON存储健康检查失败：{str(e)}")
            return False