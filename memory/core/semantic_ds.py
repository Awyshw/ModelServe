# core/semantic.py
"""
语义记忆模块，实现用户偏好的长期记忆（纯DashVector存储方案）
"""
from core.abstractions import BaseMemory
from typing import Dict, List, Optional, Any
import numpy as np
import time
from utils.logger import log
from utils.exceptions import StorageError, ValidationError
from config.settings import settings
from core.embedding_service import get_embedding_service
from storage.dashvector_storage import DashVectorStorage
from functools import wraps

# 重试装饰器（DashVector API调用失败时重试）
def dashvector_retry(max_retries: int = 2, delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        log.error(f"DashVector调用失败，已重试{max_retries}次：{str(e)}")
                        raise StorageError(f"DashVector操作失败：{str(e)}")
                    log.warning(f"DashVector调用失败，重试{retries}/{max_retries}：{str(e)}")
                    time.sleep(delay * retries)  # 指数退避
            return None
        return wrapper
    return decorator

class DsSemanticMemory(BaseMemory):
    """长期记忆：用户偏好存储+语义检索（纯DashVector存储）"""
    def __init__(
        self, 
        user_id: str, 
        storage: Optional[Any] = None  # 兼容原有参数，实际不再使用
    ):
        super().__init__(user_id=user_id, storage=None)  # 强制置空Redis存储
        self.embed_service = get_embedding_service()
        
        # 初始化DashVector（核心存储）
        self.ds_storage = DashVectorStorage()
        
        # 向量维度（缓存，避免重复调用）
        self._vector_dim = self.embed_service.get_dimension()
        
        # DashVector配置
        self._collection_name = settings.DASHVECTOR_COLLECTION
        self._similarity_threshold = settings.SEMANTIC_SIMILARITY_THRESHOLD
        self._top_k = 20  # 检索返回TopK数量
        
        log.debug(
            f"SemanticMemory初始化成功（纯DashVector），user_id={user_id}，"
            f"嵌入模式：{settings.EMBEDDING_MODE}，"
            f"向量维度：{self._vector_dim}，"
            f"向量库：{self._collection_name}"
        )

    @dashvector_retry(max_retries=2)
    def add(self, key: str, value: str, **kwargs) -> None:
        """
        添加/更新用户偏好（直接存储到DashVector）
        :param key: 偏好键（用户维度唯一）
        :param value: 偏好值
        :param kwargs: confidence=float - 置信度(0-1)
        """
        try:
            # 1. 数据验证
            if not isinstance(key, str) or len(key.strip()) == 0:
                raise ValidationError("偏好键不能为空且必须为非空字符串")
            if not isinstance(value, str) or len(value.strip()) == 0:
                raise ValidationError("偏好值不能为空且必须为非空字符串")
            
            confidence = kwargs.get("confidence", 1.0)
            if not isinstance(confidence, float) or not (0.0 <= confidence <= 1.0):
                raise ValidationError("置信度必须是0.0-1.0之间的浮点数")
            
            # 2. 生成语义嵌入向量（key+value组合，提升检索准确性）
            embed_text = f"{key}:{value}"
            embedding = self.embed_service.generate_embedding(embed_text)
            
            # 3. 向量归一化（保证余弦相似度计算准确）
            embedding = embedding / np.linalg.norm(embedding)
            
            # 4. 构建DashVector存储的ID（用户ID+偏好Key，全局唯一）
            vec_id = f"{self.user_id}:{key}"
            
            # 5. 构建元数据（存储所有信息，无需额外存储）
            metadata = {
                "user_id": self.user_id,
                "key": key,
                "value": value,
                "confidence": confidence,
                "create_time": int(time.time()),
                "update_time": int(time.time())
            }
            
            # 6. 写入DashVector（upsert：存在则更新，不存在则插入）
            self.ds_storage.upsert_vector(
                user_id=self.user_id,
                pref_key=key,
                vector=embedding,
                metadata=metadata,
            )
            
            log.debug(
                f"添加偏好成功，user_id={self.user_id}，key={key}，"
                f"confidence={confidence}，vec_id={vec_id}"
            )
            
        except ValidationError as e:
            log.error(f"偏好数据验证失败：{str(e)}")
            raise
        except Exception as e:
            log.error(f"添加偏好到DashVector失败：{str(e)}", exc_info=True)
            raise StorageError(f"偏好存储失败：{str(e)}")

    @dashvector_retry(max_retries=2)
    def get(self, key: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        获取偏好（精确查询/语义检索，纯DashVector实现）
        :param key: 精确键名（None则语义检索）
        :param kwargs: 
            - query=str: 语义检索词
            - threshold=float: 相似度阈值（默认取配置）
            - top_k=int: 返回数量（默认20）
        :return: 偏好列表（带相似度/置信度）
        """
        try:
            # 1. 精确查询（按key查询）
            if key is not None:
                vec_id = f"{self.user_id}:{key}"
                # 从DashVector获取单条数据
                result = self.ds_storage.collection.fetch(vec_id)
                
                if result is None:
                    log.debug(f"精确查询无结果，user_id={self.user_id}，key={key}")
                    return []
                
                # 构造返回格式
                return [{
                    "key": result.metadata["key"],
                    "value": result.metadata["value"],
                    "confidence": result.metadata["confidence"],
                    "similarity": 1.0  # 精确查询相似度为1
                }]
            
            # 2. 语义检索（按query检索）
            query = kwargs.get("query")
            threshold = kwargs.get("threshold", self._similarity_threshold)
            top_k = kwargs.get("top_k", self._top_k)
            
            # 无查询词则返回该用户所有偏好（按置信度排序）
            if not query:
                return self._get_all_preferences()
            
            # 3. 执行语义检索
            # 3.1 生成查询向量
            query_embedding = self.embed_service.generate_embedding(query)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # 3.2 DashVector检索（过滤当前用户+相似度阈值）
            matches = self.ds_storage.search_vector(
                user_id=self.user_id,
                query_vector=query_embedding,
                top_k=top_k,
                threshold=threshold
            )
            # 按相似度降序排序
            matches.sort(key=lambda x: x["similarity"], reverse=True)
            
            log.debug(
                f"语义检索成功，user_id={self.user_id}，query={query}，"
                f"匹配{len(matches)}条，threshold={threshold}"
            )
            
            return matches
            
        except Exception as e:
            log.error(f"检索偏好失败：{str(e)}", exc_info=True)
            raise StorageError(f"偏好检索失败：{str(e)}")

    @dashvector_retry(max_retries=2)
    def clear(self, key: Optional[str] = None) -> None:
        """
        清空长期记忆（纯DashVector实现）
        :param key: 偏好键（None则清空该用户所有偏好）
        """
        try:
            if key is not None:
                # 删除单个偏好
                delete_result = self.ds_storage.delete_vector(self.user_id, key)
                
                if delete_result:
                    log.debug(f"删除偏好成功，user_id={self.user_id}，key={key}")
                else:
                    raise ValidationError(f"偏好键不存在：{key}")
            else:
                # 清空该用户所有偏好（先查询所有vec_id，再批量删除）
                all_vec_ids = self._get_all_vec_ids()
                if all_vec_ids:
                    # DashVector批量删除（支持列表入参）
                    self.ds_storage.collection.delete(all_vec_ids)
                    log.warning(f"清空所有偏好成功，user_id={self.user_id}，共删除{len(all_vec_ids)}条")
                else:
                    log.debug(f"无偏好数据可清空，user_id={self.user_id}")
                
        except ValidationError as e:
            log.error(f"清空偏好验证失败：{str(e)}")
            raise
        except Exception as e:
            log.error(f"清空偏好失败：{str(e)}", exc_info=True)
            raise StorageError(f"偏好删除失败：{str(e)}")

    def _get_all_preferences(self) -> List[Dict[str, Any]]:
        """获取该用户所有偏好（按置信度排序）"""
        try:
            # 1. 获取该用户所有向量ID
            all_vec_ids = self._get_all_vec_ids()
            if not all_vec_ids:
                return []
            
            # 2. 批量获取向量数据
            all_results = self.ds_storage.collection.fetch(all_vec_ids)
            
            # 3. 构造返回结果（按置信度降序）
            preferences = []
            for res in all_results:
                if res is not None:
                    preferences.append({
                        "key": res.metadata["key"],
                        "value": res.metadata["value"],
                        "confidence": res.metadata["confidence"],
                        "similarity": 1.0,  # 全量查询无相似度
                        "update_time": res.metadata["update_time"]
                    })
            
            # 按置信度排序
            preferences.sort(key=lambda x: x["confidence"], reverse=True)
            
            log.debug(f"获取所有偏好成功，user_id={self.user_id}，共{len(preferences)}条")
            return preferences
            
        except Exception as e:
            log.error(f"获取所有偏好失败：{str(e)}")
            raise StorageError(f"获取全量偏好失败：{str(e)}")

    def _get_all_vec_ids(self) -> List[str]:
        """获取该用户所有向量ID（用于批量删除）"""
        try:
            # DashVector暂时不支持直接按filter查询ID，需通过scan接口（分页）
            vec_ids = []
            
            # scan接口：按filter分页查询
            scan_result = self.ds_storage.collection.query(
                filter=f"user_id='{self.user_id}'",
                limit=100  # 最多100条
            )
            
            # 提取vec_id
            for item in scan_result.output:
                vec_ids.append(item.id)
            
            return vec_ids
            
        except Exception as e:
            log.error(f"获取用户所有vec_id失败：{str(e)}")
            raise StorageError(f"查询用户向量ID失败：{str(e)}")

    # 保留原有静态方法（兼容可能的外部调用）
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度（备用，实际已由DashVector计算）"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        
        similarity = float(np.dot(vec1, vec2) / (norm1 * norm2))
        return max(0.0, min(1.0, similarity))

    # 移除原有Redis相关方法（_save_to_storage/_load_from_storage）
    def _save_to_storage(self) -> None:
        """兼容原有抽象方法，实际无操作"""
        pass

    def _load_from_storage(self) -> None:
        """兼容原有抽象方法，实际无操作"""
        pass