from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional, Dict
from loguru import logger
import os
from embedding.config import settings

class ModelManager:
    _instance = None
    _models: Dict[str, SentenceTransformer] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_default_model()
        return cls._instance
    
    def _load_default_model(self):
        """加载默认模型"""
        try:
            default_model_name = settings.DEFAULT_MODEL
            self.load_model(default_model_name)
            logger.info(f"默认模型加载成功: {default_model_name}")
        except Exception as e:
            logger.error(f"加载默认模型失败: {e}")
            raise
    
    def _get_model_path(self, model_name: str) -> str:
        """获取模型路径（本地路径优先）"""
        # 检查是否在支持的模型列表中
        if model_name in settings.SUPPORTED_MODELS:
            model_path = settings.SUPPORTED_MODELS[model_name]
        else:
            model_path = model_name
        
        # 检查本地模型目录
        local_model_path = os.path.join(settings.MODEL_DIR, model_path)
        if os.path.exists(local_model_path) and os.path.isdir(local_model_path):
            return local_model_path
        return model_path  # 否则从 HuggingFace 下载
    
    def load_model(self, model_name: str) -> SentenceTransformer:
        """加载模型（单例模式，避免重复加载）"""
        if model_name in self._models:
            return self._models[model_name]
        
        logger.info(f"正在加载模型: {model_name}")
        try:
            model_path = self._get_model_path(model_name)
            model = SentenceTransformer(
                model_path,
                device="cuda" if os.environ.get("USE_CUDA", "False").lower() == "true" else "cpu"
            )
            
            # 缓存模型
            self._models[model_name] = model
            logger.info(f"模型加载成功: {model_name} (路径: {model_path})")
            return model
        except Exception as e:
            logger.error(f"加载模型 {model_name} 失败: {e}")
            raise
    
    def get_embedding(
        self,
        texts: List[str],
        model_name: Optional[str] = None,
        normalize: bool = settings.NORMALIZE_EMBEDDINGS,
        batch_size: int = settings.MAX_BATCH_SIZE
    ) -> np.ndarray:
        """生成文本嵌入向量"""
        # 选择模型
        model_name = model_name or settings.DEFAULT_MODEL
        model = self.load_model(model_name)
        
        # 批量处理文本
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        
        return embeddings
    
    def get_model_dimension(self, model_name: Optional[str] = None) -> int:
        """获取模型嵌入维度"""
        model_name = model_name or settings.DEFAULT_MODEL
        model = self.load_model(model_name)
        return model.get_sentence_embedding_dimension()
    
    def unload_model(self, model_name: str):
        """卸载模型释放内存"""
        if model_name in self._models:
            del self._models[model_name]
            logger.info(f"模型已卸载: {model_name}")
        else:
            logger.warning(f"模型未加载: {model_name}")

# 单例模型管理器
model_manager = ModelManager()