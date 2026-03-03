# storage/dashvector_storage.py
from typing import List, Dict, Any, Optional
import numpy as np
from dashvector import Client, DashVectorException, Doc
from config.settings import settings
from utils.logger import log
from utils.exceptions import StorageError

class DashVectorStorage:
    """DashVector 向量存储适配器（仅处理向量相关操作）"""
    def __init__(self):
        # 初始化 DashVector 客户端
        self.client = Client(
            api_key=settings.DASHVECTOR_API_KEY.get_secret_value() if settings.DASHVECTOR_API_KEY else None,
            endpoint=settings.DASHVECTOR_ENDPOINT
        )
        self.collection_name = settings.DASHVECTOR_COLLECTION  # 向量库名称
        self.dimension = settings.DASHVECTOR_DIMENSION # 向量维度 DASHVECTOR_DIMENSION=1024
        self._check_collection()  # 检查向量库是否存在

    def _check_collection(self):
        """检查向量库是否存在，不存在则创建"""
        try:
            collections = self.client.list()
            if self.collection_name not in collections.output:
                # 创建向量库
                log.info(f"创建 DashVector 向量库：{self.collection_name}")
                ret = self.client.create(
                    name=self.collection_name,
                    dimension=self.dimension,
                    metric="cosine",   # 距离度量方式，euclidean/dotproduct/cosine, 值为cosine时，dtype必须为float
                    dtype=float,  # 向量数据类型
                )
                if ret:
                    log.info(f"创建 DashVector 向量库成功：{self.collection_name}")
            self.collection = self.client.get(self.collection_name)
            if not self.collection:
                raise StorageError(f"DashVector 向量库获取失败：{self.collection_name}")
        except DashVectorException as e:
            log.error(f"DashVector 向量库检查失败：{str(e)}")
            raise StorageError(f"DashVector 初始化失败：{str(e)}")

    def upsert_vector(self, user_id: str, pref_key: str, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """
        插入/更新向量（唯一键：user_id:pref_key）
        :param user_id: 用户ID
        :param pref_key: 偏好Key
        :param vector: 向量数组
        """
        try:
            # 向量归一化（保证余弦相似度计算准确）
            # vector = vector / np.linalg.norm(vector)
            # 唯一ID：user_id:pref_key（避免不同用户Key冲突）
            vec_id = f"{user_id}:{pref_key}"
            
            # 插入/更新向量
            ret = self.collection.upsert(
                Doc(
                    id=vec_id,
                    vector=vector.tolist(),
                    fields=metadata
                )
            )
            if ret:
                log.debug(f"向量入库成功：{vec_id}")
            else:
                log.error(f"向量入库失败：{vec_id}")
                raise StorageError(f"DashVector upsert 失败")
        except DashVectorException as e:
            log.error(f"向量入库失败：{user_id}:{pref_key}，错误：{str(e)}")
            raise StorageError(f"DashVector upsert 失败：{str(e)}")

    def search_vector(self, user_id: str, query_vector: np.ndarray, top_k: int = 10, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        语义检索向量
        :param user_id: 用户ID（过滤仅当前用户的向量）
        :param query_vector: 查询向量
        :param top_k: 返回TopK结果
        :param threshold: 相似度阈值
        :return: 匹配的偏好Key列表（含相似度）
        """
        try:
            # 向量归一化
            # query_vector = query_vector / np.linalg.norm(query_vector)
            # 检索（过滤用户ID，避免跨用户检索）
            search_result = self.collection.query(
                vector=query_vector.tolist(),
                top_k=top_k,
                filter=f"user_id='{user_id}'",  # DashVector 过滤语法
                include_vector=False,
            )
            # 过滤阈值，返回结果
            matched = []
            for res in search_result.output:
                similarity = res.score  # 余弦相似度（DashVector 返回 0-1 之间）
                if similarity >= threshold:
                    matched.append({
                        "key": res.metadata["key"],
                        "value": res.metadata["value"],
                        "confidence": res.metadata["confidence"],
                        "similarity": round(similarity, 4),
                        "update_time": res.metadata["update_time"],
                        "vec_id": res.id
                    })
            log.debug(f"DashVector 检索结果：user_id={user_id}，匹配{len(matched)}条")
            return matched
        except DashVectorException as e:
            log.error(f"DashVector 检索失败：user_id={user_id}，错误：{str(e)}")
            raise StorageError(f"DashVector search 失败：{str(e)}")

    def delete_vector(self, user_id: str, pref_key: str):
        """删除指定向量"""
        try:
            vec_id = f"{user_id}:{pref_key}"
            ret = self.collection.delete(vec_id)
            log.debug(f"删除向量成功：{vec_id}")
            return ret
        except DashVectorException as e:
            log.error(f"删除向量失败：{user_id}:{pref_key}，错误：{str(e)}")
            raise StorageError(f"DashVector delete 失败：{str(e)}")

    def health_check(self) -> bool:
        """健康检查"""
        try:
            self.collection.stats()  # 获取向量库统计信息
            return True
        except Exception:
            return False