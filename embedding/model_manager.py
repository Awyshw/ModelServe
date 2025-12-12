from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional, Dict, Tuple
from loguru import logger
import os
from embedding.config import settings
import torch

class ModelManager:
    _instance = None
    _embedding_models: Dict[str, SentenceTransformer] = {}
    _rerank_models: Dict[str, SentenceTransformer] = {} 
    
    def __new__(cls):
        # 检查类是否已有实例
        if cls._instance is None:
            # 如果没有实例，则创建新实例
            cls._instance = super().__new__(cls)
            # 调用实例的_load_default_model方法加载默认模型
            cls._instance._load_default_model()
        # 返回类实例（可能是新创建的，也可能是已存在的）
        return cls._instance
    
    def _load_default_model(self):
        """加载默认模型"""
        try:
            # embedding 模型
            default_model_name = settings.DEFAULT_MODEL
            self.load_model(default_model_name)
            logger.info(f"默认模型加载成功: {default_model_name}")

            # rerank 模型
            default_rerank_model_name = settings.DEFAULT_RERANK_MODEL
            self.load_rerank_model(default_rerank_model_name)
            logger.info(f"默认 rerank 模型加载成功: {default_rerank_model_name}")
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

        if model_name in settings.SUPPORTED_RERANK_MODELS:
            model_path = settings.SUPPORTED_RERANK_MODELS[model_name]
        else:
            model_path = model_name
        
        # 检查本地模型目录
        local_model_path = os.path.join(settings.MODEL_DIR, model_path)
        print(f"local model path: {local_model_path}")
        if os.path.exists(local_model_path) and os.path.isdir(local_model_path):
            return local_model_path
        return model_path  # 否则从 HuggingFace 下载
    
    def load_model(self, model_name: str) -> SentenceTransformer:
        """加载模型（单例模式，避免重复加载）"""
        if model_name in self._embedding_models:
            return self._embedding_models[model_name]
        
        logger.info(f"正在加载模型: {model_name}")
        try:
            model_path = self._get_model_path(model_name)
            logger.info(f"模型路径： {model_path}")
            model = SentenceTransformer(
                model_path,
                device="cuda" if os.environ.get("USE_CUDA", "False").lower() == "true" else "cpu"
            )
            
            # 缓存模型
            self._embedding_models[model_name] = model
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
    
    def load_rerank_model(self, model_name: str) -> SentenceTransformer:
        """加载 Rerank 模型（Cross-Encoder 类型）"""
        if model_name in self._rerank_models:
            return self._rerank_models[model_name]
        
        # 处理模型名称映射
        if model_name in settings.SUPPORTED_RERANK_MODELS:
            model_name = settings.SUPPORTED_RERANK_MODELS[model_name]
        
        logger.info(f"正在加载 Rerank 模型: {model_name}")
        try:
            model_path = self._get_model_path(model_name)
            logger.info(f"模型路径： {model_path}")
            # Rerank 模型通常是 Cross-Encoder，使用 SentenceTransformer 加载
            model = SentenceTransformer(
                model_path,
                device="cuda" if os.environ.get("USE_CUDA", "False").lower() == "true" else "cpu"
            )
            
            self._rerank_models[model_name] = model
            logger.info(f"Rerank 模型加载成功: {model_name} (路径: {model_path})")
            return model
        except Exception as e:
            logger.error(f"加载 Rerank 模型 {model_name} 失败: {e}")
            raise
    
    def rerank(
        self,
        query: str,
        candidates: List[str],
        model_name: Optional[str] = None,
        return_scores: bool = settings.RETURN_SCORES,
        batch_size: int = settings.RERANK_BATCH_SIZE
    ) -> Tuple[List[str], List[float], List[int]]:
        """
        对候选文本进行重排序
        返回：排序后的文本列表、分数列表、原始索引列表
        """
        # 选择模型
        model_name = model_name or settings.DEFAULT_RERANK_MODEL
        model = self.load_rerank_model(model_name)

        # 验证输入
        if not query or not candidates:
            raise ValueError("query 和 candidates 不能为空")
        
        if len(candidates) > settings.MAX_RERANK_CANDIDATES:
            raise ValueError(f"候选文本数量不能超过 {settings.MAX_RERANK_CANDIDATES}")
        
        # 构建 (query, candidate) 对
        pairs = [(query, candidate) for candidate in candidates]
        
        # 计算分数（使用 cross-encoder 方式）
        logger.debug(f"Rerank 处理 {len(pairs)} 个候选文本")
        try:
            with torch.no_grad():
                scores = model.predict(
                    pairs,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
        except AttributeError:
            # 兼容某些特殊模型，使用 encode 后计算相似度
            logger.warning("当前模型不支持 predict 方法，使用余弦相似度 fallback")
            query_emb = model.encode([query], convert_to_tensor=True)
            candidate_embs = model.encode(candidates, convert_to_tensor=True)

            # 计算余弦相似度
            if model.device.type == "cuda":
                query_emb = query_emb.cuda()
                candidate_embs = candidate_embs.cuda()
            
            scores = torch.nn.functional.cosine_similarity(query_emb, candidate_embs).cpu().numpy()
        
        # 转换为 numpy 数组并展平
        scores = np.asarray(scores).flatten().tolist()

        # 按分数降序排序，保留原始索引
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        sorted_candidates = [candidates[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]
        original_indices = [i for i in sorted_indices]
        
        return sorted_candidates, sorted_scores, original_indices
    
    def unload_model(self, model_name: str, model_type: str = "embedding"):
        """卸载模型释放内存"""
        if model_type == "embedding":
            models_dict = self._embedding_models
        elif model_type == "rerank":
            models_dict = self._rerank_models
        else:
            logger.warning(f"不支持的模型类型: {model_type}")
            return
        
        if model_name in models_dict:
            del models_dict[model_name]
            logger.info(f"{model_type} 模型已卸载: {model_name}")
        else:
            logger.warning(f"{model_type} 模型未加载: {model_name}")
    
    def unload_all_models(self):
        """卸载所有模型"""
        self._embedding_models.clear()
        self._rerank_models.clear()
        logger.info("所有模型已卸载")

# 单例模型管理器
model_manager = ModelManager()

if __name__ == "__main__":
    # 测试 Rerank 功能
    query = "介绍人工智能的应用"
    candidates = [
        "机器学习是人工智能的一个分支",
        "医疗应用包括疾病诊断和药物研发",
        "人工智能在医疗领域有广泛应用",
        "编程语言包括 Python、Java 等",
        "人工智能可以用于自动驾驶汽车"
    ]
    
    try:
        sorted_docs, sorted_scores, original_indices = model_manager.rerank(
            query=query,
            candidates=candidates,
            model_name="bge-reranker-v2-m3"
        )
        
        print("Rerank 结果：")
        for i, (doc, score, orig_idx) in enumerate(zip(sorted_docs, sorted_scores, original_indices)):
            print(f"排名 {i+1}: 分数={score:.4f}, 原始索引={orig_idx}, 文本={doc}")
    except Exception as e:
        print(f"测试失败: {e}")