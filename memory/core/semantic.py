# core/semantic.py
"""
语义记忆模块，实现用户偏好的长期记忆，这里需要添加向量数据库
TODO:向量数据库用于存储用户偏好的语义嵌入向量，以便进行语义检索 和推荐
"""
from core.abstractions import BaseMemory
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from utils.logger import log
from utils.exceptions import StorageError, ValidationError
from config.settings import settings
from core.embedding_service import get_embedding_service  # 新增

class SemanticMemory(BaseMemory):
    """长期记忆：用户偏好存储+语义检索（兼容本地/OpenAI嵌入）"""
    def __init__(
        self, 
        user_id: str, 
        storage: Optional[Any] = None
    ):
        super().__init__(user_id=user_id, storage=storage)
        
        # 使用工厂函数获取嵌入服务（自动适配本地/OpenAI）
        # TODO: API 的方式需要添加基于阿里云的调用方式
        self.embed_service = get_embedding_service()
        
        # 核心存储：{key: (value, confidence, embedding)}
        self.preferences: Dict[str, Tuple[str, float, np.ndarray]] = {}
        
        # 加载持久化数据
        if self.storage:
            self._load_from_storage()
        log.debug(
            f"SemanticMemory初始化成功，user_id={user_id}，"
            f"嵌入模式：{settings.EMBEDDING_MODE}，"
            f"向量维度：{self.embed_service.get_dimension()}"
        )

    def add(self, key: str, value: str, **kwargs) -> None:
        """
        添加/更新用户偏好（兼容本地/OpenAI嵌入）
        :param key: 偏好键
        :param value: 偏好值
        :param kwargs: confidence=float - 置信度(0-1)
        """
        try:
            # 数据验证
            if not isinstance(key, str) or len(key.strip()) == 0:
                raise ValidationError("偏好键不能为空")
            if not isinstance(value, str) or len(value.strip()) == 0:
                raise ValidationError("偏好值不能为空")
            
            confidence = kwargs.get("confidence", 1.0)
            if not isinstance(confidence, float) or not (0.0 <= confidence <= 1.0):
                raise ValidationError("置信度必须在0.0-1.0之间")
            
            # 生成语义嵌入向量（统一调用抽象接口）
            embed_text = f"{key}:{value}"
            embedding = self.embed_service.generate_embedding(embed_text)
            
            with self.lock:
                self.preferences[key] = (value, confidence, embedding)
                log.debug(f"添加偏好成功，user_id={self.user_id}，key={key}，confidence={confidence}")
            
            # 持久化
            if self.storage:
                self._save_to_storage()
                
        except Exception as e:
            log.error(f"添加偏好失败：{str(e)}", exc_info=True)
            raise

    def get(self, key: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        获取偏好（精确查询/语义检索，兼容两种嵌入模式）
        :param key: 精确键名（None则语义检索）
        :param kwargs: query=str - 检索词；threshold=float - 相似度阈值
        :return: 偏好列表（带相似度/置信度）
        """
        try:
            with self.lock:
                # 空数据直接返回
                if not self.preferences:
                    return []
                
                # 1. 精确查询
                if key and key in self.preferences:
                    value, confidence, _ = self.preferences[key]
                    result = [{
                        "key": key,
                        "value": value,
                        "confidence": confidence
                    }]
                    log.debug(f"精确查询偏好成功，user_id={self.user_id}，key={key}")
                    return result
                
                # 2. 语义检索
                query = kwargs.get("query")
                threshold = kwargs.get("threshold", settings.SEMANTIC_SIMILARITY_THRESHOLD)
                
                # 无查询词则返回所有偏好（按置信度排序）
                if not query:
                    sorted_items = sorted(
                        self.preferences.items(),
                        key=lambda x: x[1][1],
                        reverse=True
                    )
                    result = [{
                        "key": k,
                        "value": v[0],
                        "confidence": v[1]
                    } for k, v in sorted_items]
                    log.debug(f"返回所有偏好，user_id={self.user_id}，共{len(result)}条")
                    return result
                
                # 3. 执行语义检索（统一相似度计算）
                query_embed = self.embed_service.generate_embedding(query)
                matches = []
                
                for pref_key, (value, confidence, embed) in self.preferences.items():
                    similarity = self._cosine_similarity(query_embed, embed)
                    if similarity >= threshold:
                        matches.append({
                            "key": pref_key,
                            "value": value,
                            "confidence": confidence,
                            "similarity": round(similarity, 4)
                        })
                
                # 按相似度降序排序
                matches.sort(key=lambda x: x["similarity"], reverse=True)
                log.debug(f"语义检索成功，user_id={self.user_id}，匹配{len(matches)}条，查询词={query}")
                return matches
            
        except Exception as e:
            log.error(f"检索偏好失败：{str(e)}", exc_info=True)
            raise

    def clear(self, key: Optional[str] = None) -> None:
        """清空长期记忆（key=None清空全部）"""
        try:
            with self.lock:
                if key:
                    if key in self.preferences:
                        del self.preferences[key]
                        log.debug(f"删除偏好成功，user_id={self.user_id}，key={key}")
                    else:
                        raise ValidationError(f"偏好键不存在：{key}")
                else:
                    self.preferences.clear()
                    log.warning(f"清空所有偏好成功，user_id={self.user_id}")
            
            # 同步清理存储
            if self.storage:
                if key:
                    # 删除单个键需要重新保存全部数据
                    self._save_to_storage()
                else:
                    self.storage.delete(self.user_id, "semantic")
                    
        except Exception as e:
            log.error(f"清空偏好失败：{str(e)}", exc_info=True)
            raise

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度（兼容所有向量维度）"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        
        similarity = float(np.dot(vec1, vec2) / (norm1 * norm2))
        # 处理浮点精度问题（限制在0-1之间）
        return max(0.0, min(1.0, similarity))

    def _save_to_storage(self) -> None:
        """持久化偏好数据（numpy数组转列表）"""
        try:
            with self.lock:
                serialized = {
                    key: {
                        "value": value,
                        "confidence": confidence,
                        "embedding": embedding.tolist()
                    }
                    for key, (value, confidence, embedding) in self.preferences.items()
                }
                
                self.storage.save(self.user_id, "semantic", serialized)
            log.debug(f"长期记忆持久化成功，user_id={self.user_id}，共{len(serialized)}条偏好")
            
        except Exception as e:
            log.error(f"长期记忆持久化失败：{str(e)}", exc_info=True)
            raise StorageError(f"长期记忆存储失败：{str(e)}")

    def _load_from_storage(self) -> None:
        """从存储加载偏好数据"""
        try:
            serialized = self.storage.load(self.user_id, "semantic")
            if not serialized:
                log.debug(f"无持久化长期记忆数据，user_id={self.user_id}")
                return
            
            with self.lock:
                self.preferences = {
                    key: (
                        item["value"],
                        item["confidence"],
                        np.array(item["embedding"], dtype=np.float32)
                    )
                    for key, item in serialized.items()
                }
            log.debug(f"长期记忆加载成功，user_id={self.user_id}，加载{len(self.preferences)}条偏好")
            
        except Exception as e:
            log.error(f"长期记忆加载失败：{str(e)}", exc_info=True)
            raise StorageError(f"长期记忆加载失败：{str(e)}")