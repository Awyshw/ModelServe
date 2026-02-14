# core/embedding_service.py
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np
import time
import hashlib
from functools import lru_cache
from utils.logger import log
from utils.exceptions import StorageError, ValidationError
from config.settings import settings

# 缓存配置（生产级：小量缓存用LRU，大量建议用Redis）
CACHE_MAX_SIZE = 1000 if settings.EMBEDDING_CACHE_ENABLED else 0

class BaseEmbeddingService(ABC):
    """嵌入服务抽象接口（统一本地/OpenAI调用）"""
    @abstractmethod
    def generate_embedding(self, text: str) -> np.ndarray:
        """生成单段文本的嵌入向量"""
        pass
    
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """批量生成文本嵌入向量"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """获取嵌入向量维度"""
        pass

class LocalEmbeddingService(BaseEmbeddingService):
    """本地嵌入服务（基于sentence-transformers）"""
    def __init__(self, model_name: str = settings.LOCAL_EMBED_MODEL):
        self.model_name = model_name
        self._model = None
        self._dimension = None
        log.info(f"初始化本地嵌入服务，模型：{model_name}")

    @property
    def model(self):
        """懒加载本地模型"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                # 预计算向量维度
                self._dimension = len(self._model.encode("test"))
                log.info(f"本地模型加载成功，维度：{self._dimension}")
            except Exception as e:
                log.error(f"本地嵌入模型加载失败：{str(e)}", exc_info=True)
                raise StorageError(f"本地嵌入模型初始化失败：{str(e)}")
        return self._model

    @lru_cache(maxsize=CACHE_MAX_SIZE)
    def generate_embedding(self, text: str) -> np.ndarray:
        """生成嵌入向量（带LRU缓存）"""
        try:
            if not text or not isinstance(text, str):
                raise ValidationError("嵌入文本不能为空")
            
            # 清理文本（去除多余空格、换行）
            clean_text = text.strip().replace("\n", " ").replace("\r", " ")
            embedding = self.model.encode(
                clean_text,
                convert_to_numpy=True,
                normalize_embeddings=True  # 归一化，提高相似度计算准确性
            )
            log.debug(f"本地嵌入生成成功，文本长度：{len(clean_text)}，维度：{len(embedding)}")
            return embedding
        except Exception as e:
            log.error(f"本地嵌入生成失败：{str(e)}", exc_info=True)
            raise StorageError(f"本地嵌入生成失败：{str(e)}")

    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """批量生成嵌入向量（优化性能）"""
        try:
            if not texts or not isinstance(texts, list):
                raise ValidationError("批量嵌入文本不能为空")
            
            # 清理文本列表
            clean_texts = [
                t.strip().replace("\n", " ").replace("\r", " ") 
                for t in texts if isinstance(t, str) and t.strip()
            ]
            
            if not clean_texts:
                return []
            
            embeddings = self.model.encode(
                clean_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=32  # 批量大小优化
            )
            log.debug(f"本地批量嵌入生成成功，数量：{len(clean_texts)}")
            return [emb for emb in embeddings]
        except Exception as e:
            log.error(f"本地批量嵌入生成失败：{str(e)}", exc_info=True)
            raise StorageError(f"本地批量嵌入生成失败：{str(e)}")

    def get_dimension(self) -> int:
        """返回向量维度"""
        if self._dimension is None:
            # 预生成测试向量获取维度
            self.generate_embedding("test")
        return self._dimension

class OpenAIEmbeddingService(BaseEmbeddingService):
    """OpenAI API嵌入服务（生产级：重试+缓存+超时）"""
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY.get_secret_value() if settings.OPENAI_API_KEY else None
        self.model = settings.OPENAI_EMBED_MODEL
        self.api_base = settings.OPENAI_API_BASE or "https://api.openai.com/v1"
        self.timeout = settings.OPENAI_API_TIMEOUT
        self.max_retries = settings.OPENAI_MAX_RETRIES
        self._dimension = None
        
        # 初始化OpenAI客户端
        self._client = None
        self._init_client()
        log.info(f"初始化OpenAI嵌入服务，模型：{self.model}")

    def _init_client(self):
        """初始化OpenAI客户端"""
        try:
            import openai
            # 支持OpenAI SDK v1.x
            if hasattr(openai, "OpenAI"):
                self._client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base,
                    timeout=self.timeout
                )
            else:
                # 兼容旧版SDK
                openai.api_key = self.api_key
                if self.api_base:
                    openai.api_base = self.api_base
            log.info("OpenAI客户端初始化成功")
        except ImportError:
            raise StorageError("未安装openai SDK，请执行：pip install openai")
        except Exception as e:
            log.error(f"OpenAI客户端初始化失败：{str(e)}", exc_info=True)
            raise StorageError(f"OpenAI客户端初始化失败：{str(e)}")

    def _retry_decorator(self, func):
        """重试装饰器（指数退避）"""
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < self.max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    wait_time = 2 ** retries  # 指数退避：2,4,8秒
                    log.warning(
                        f"OpenAI API调用失败（重试{retries}/{self.max_retries}）：{str(e)}，"
                        f"等待{wait_time}秒后重试"
                    )
                    time.sleep(wait_time)
            raise StorageError(f"OpenAI API调用失败，已重试{self.max_retries}次")
        return wrapper

    @lru_cache(maxsize=CACHE_MAX_SIZE)
    @_retry_decorator
    def generate_embedding(self, text: str) -> np.ndarray:
        """生成OpenAI嵌入向量（带缓存+重试）"""
        try:
            if not text or not isinstance(text, str):
                raise ValidationError("嵌入文本不能为空")
            
            # OpenAI官方建议：文本长度限制
            clean_text = text.strip().replace("\n", " ")
            if len(clean_text) > 8191:
                log.warning(f"文本过长，截断至8191字符：{clean_text[:100]}...")
                clean_text = clean_text[:8191]
            
            # 调用OpenAI API
            if hasattr(self, "_client") and self._client:
                # OpenAI SDK v1.x
                response = self._client.embeddings.create(
                    input=clean_text,
                    model=self.model
                )
                embedding = np.array(response.data[0].embedding, dtype=np.float32)
            else:
                # 兼容旧版SDK
                import openai
                response = openai.Embedding.create(
                    input=clean_text,
                    model=self.model
                )
                embedding = np.array(response["data"][0]["embedding"], dtype=np.float32)
            
            # 缓存向量维度
            if self._dimension is None:
                self._dimension = len(embedding)
            
            log.debug(
                f"OpenAI嵌入生成成功，文本长度：{len(clean_text)}，"
                f"维度：{self._dimension}，模型：{self.model}"
            )
            return embedding
        except Exception as e:
            log.error(f"OpenAI嵌入生成失败：{str(e)}", exc_info=True)
            raise StorageError(f"OpenAI嵌入生成失败：{str(e)}")

    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """批量生成OpenAI嵌入向量"""
        try:
            if not texts or not isinstance(texts, list):
                raise ValidationError("批量嵌入文本不能为空")
            
            # 清理并过滤文本
            clean_texts = []
            original_indices = []
            for idx, text in enumerate(texts):
                if isinstance(text, str) and text.strip():
                    clean_text = text.strip().replace("\n", " ")
                    # 截断超长文本
                    if len(clean_text) > 8191:
                        clean_text = clean_text[:8191]
                        log.warning(f"批量文本{idx}过长，已截断")
                    clean_texts.append(clean_text)
                    original_indices.append(idx)
            
            if not clean_texts:
                return []
            
            # 批量调用API（OpenAI支持批量输入）
            if hasattr(self, "_client") and self._client:
                response = self._client.embeddings.create(
                    input=clean_texts,
                    model=self.model
                )
                embeddings = [np.array(item.embedding, dtype=np.float32) for item in response.data]
            else:
                import openai
                response = openai.Embedding.create(
                    input=clean_texts,
                    model=self.model
                )
                embeddings = [np.array(item["embedding"], dtype=np.float32) for item in response.data]
            
            # 缓存维度
            if self._dimension is None and embeddings:
                self._dimension = len(embeddings[0])
            
            # 还原原始顺序（处理空文本）
            result = [np.array([])] * len(texts)
            for idx, emb in zip(original_indices, embeddings):
                result[idx] = emb
            
            log.debug(f"OpenAI批量嵌入生成成功，有效数量：{len(clean_texts)}")
            return result
        except Exception as e:
            log.error(f"OpenAI批量嵌入生成失败：{str(e)}", exc_info=True)
            raise StorageError(f"OpenAI批量嵌入生成失败：{str(e)}")

    def get_dimension(self) -> int:
        """获取OpenAI嵌入向量维度"""
        if self._dimension is None:
            # 预生成测试向量获取维度
            self.generate_embedding("test")
        return self._dimension

def get_embedding_service() -> BaseEmbeddingService:
    """工厂函数：根据配置获取嵌入服务实例（单例）"""
    if getattr(get_embedding_service, "_instance", None) is None:
        if settings.EMBEDDING_MODE == "openai":
            get_embedding_service._instance = OpenAIEmbeddingService()
        else:
            get_embedding_service._instance = LocalEmbeddingService()
    return get_embedding_service._instance