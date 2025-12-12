from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from loguru import logger
import time
import numpy as np
from embedding.config import settings
from embedding.model_manager import model_manager

# 创建 FastAPI 应用
app = FastAPI(
    title="Local Embedding API (OpenAI Compatible)",
    description="兼容 OpenAI Embedding API 的本地向量模型服务，支持 BGE、BCE 等Embedding 和 Reranker 模型",
    version="2.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# 数据模型（兼容 OpenAI API 格式）
# ------------------------------
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]  # 单个文本或文本列表
    model: Optional[str] = Field(default=settings.DEFAULT_MODEL, description="模型名称")
    encoding_format: Optional[str] = Field(default="float", description="编码格式: float 或 base64")
    user: Optional[str] = Field(default=None, description="用户标识")
    
    class Config:
        schema_extra = {
            "example": {
                "input": ["Hello world", "本地向量模型"],
                "model": "text-embedding-ada-002",  # 可映射到本地模型
                "encoding_format": "float"
            }
        }

class EmbeddingItem(BaseModel):
    object: str = "embedding"
    embedding: Union[List[float], str]  # 向量数据或 base64 编码
    index: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingItem]
    model: str
    usage: dict = Field(
        default_factory=lambda: {"prompt_tokens": 0, "total_tokens": 0},
        description="Token 使用统计（本地模型暂不精确统计）"
    )

# ------------------------------
# Rerank 数据模型（新增，遵循 OpenAI 风格）
# ------------------------------
class RerankRequest(BaseModel):
    query: str = Field(description="查询文本")
    documents: List[str] = Field(description="待排序的候选文本列表")
    model: Optional[str] = Field(default=settings.DEFAULT_RERANK_MODEL, description="Rerank 模型名称")
    return_scores: Optional[bool] = Field(default=settings.RETURN_SCORES, description="是否返回排序分数")
    return_documents: Optional[bool] = Field(default=True, description="是否返回排序后的文本")
    user: Optional[str] = Field(default=None, description="用户标识")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "介绍人工智能的应用",
                "documents": [
                    "机器学习是人工智能的一个分支",
                    "人工智能在医疗领域有广泛应用",
                    "编程语言包括 Python、Java 等",
                    "人工智能可以用于自动驾驶汽车"
                ],
                "model": "text-rerank-ada-001",  # 映射到本地 bce-reranker-base_v1
                "return_scores": True
            }
        }

class RerankItem(BaseModel):
    object: str = "rerank_result"
    index: int = Field(description="排序后的索引（从 0 开始）")
    original_index: int = Field(description="原始文档列表中的索引")
    document: Optional[str] = Field(default=None, description="候选文本内容")
    score: Optional[float] = Field(default=None, description="排序分数（越高排名越前）")

class RerankResponse(BaseModel):
    object: str = "list"
    data: List[RerankItem]
    model: str
    usage: dict = Field(
        default_factory=lambda: {"prompt_tokens": 0, "total_tokens": 0},
        description="Token 使用统计（本地模型暂不精确统计）"
    )


# ------------------------------
# 依赖项
# ------------------------------
def verify_api_key(authorization: Optional[str] = Header(None)):
    """验证 API Key（如果启用）"""
    if not settings.API_KEY_REQUIRED:
        return True
    
    if not authorization:
        raise HTTPException(status_code=401, detail="未提供 API Key")
    
    scheme, api_key = authorization.split() if authorization else (None, None)
    if scheme != "Bearer" or api_key not in settings.VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="无效的 API Key")
    
    return True

# ------------------------------
# API 端点（兼容 OpenAI /v1/embeddings）
# ------------------------------
@app.post("/v1/embeddings", response_model=EmbeddingResponse, dependencies=[Depends(verify_api_key)])
async def create_embedding(request: EmbeddingRequest):
    """生成文本嵌入向量（兼容 OpenAI API）"""
    start_time = time.time()
    
    try:
        # 处理输入：统一转为列表
        texts = [request.input] if isinstance(request.input, str) else request.input
        
        # 验证输入
        if not texts or len(texts) == 0:
            raise HTTPException(status_code=400, detail="输入文本不能为空")
        
        if len(texts) > 1000:  # 限制最大输入数量
            raise HTTPException(status_code=400, detail="单次请求最多支持 1000 个文本")
        
        # 映射到本地模型
        local_model_name = settings.SUPPORTED_MODELS.get(request.model, request.model)
        
        # 生成嵌入向量
        embeddings = model_manager.get_embedding(
            texts=texts,
            model_name=local_model_name
        )
        
        # 处理输出格式
        embedding_items = []
        for idx, emb in enumerate(embeddings):
            if request.encoding_format == "base64":
                # 转换为 base64 编码
                import base64
                emb_encoded = base64.b64encode(emb.astype(np.float32)).decode("utf-8")
                embedding_data = emb_encoded
            else:
                # 转换为 float 列表
                embedding_data = emb.tolist()
            
            embedding_items.append(
                EmbeddingItem(
                    object="embedding",
                    embedding=embedding_data,
                    index=idx
                )
            )
        
        # 构建响应
        response = EmbeddingResponse(
            data=embedding_items,
            model=request.model,
            usage={
                "prompt_tokens": sum(len(text.split()) for text in texts),  # 简单估算
                "total_tokens": sum(len(text.split()) for text in texts)
            }
        )
        
        logger.info(
            f"生成嵌入成功 - 模型: {local_model_name}, 文本数量: {len(texts)}, "
            f"耗时: {time.time() - start_time:.3f}s"
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"生成嵌入失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")

@app.post("/v1/rerank", response_model=RerankResponse, dependencies=[Depends(verify_api_key)])
async def create_rerank(request: RerankRequest):
    """文本重排序（兼容 OpenAI 风格 API）"""
    start_time = time.time()
    
    try:
        # 验证输入
        if not request.query:
            raise HTTPException(status_code=400, detail="query 不能为空")
        
        if not request.documents or len(request.documents) == 0:
            raise HTTPException(status_code=400, detail="documents 不能为空")
        
        if len(request.documents) > settings.MAX_RERANK_CANDIDATES:
            raise HTTPException(
                status_code=400,
                detail=f"documents 数量不能超过 {settings.MAX_RERANK_CANDIDATES} 个"
            )
        
        # 映射到本地模型
        local_model_name = settings.SUPPORTED_RERANK_MODELS.get(request.model, request.model)
        
        # 执行 rerank
        logger.debug(f"Rerank 请求 - 模型: {local_model_name}, query: {request.query}, 候选数: {len(request.documents)}")
        sorted_docs, sorted_scores, original_indices = model_manager.rerank(
            query=request.query,
            candidates=request.documents,
            model_name=local_model_name,
            return_scores=request.return_scores
        )
        
        # 构建响应数据
        rerank_items = []
        for rank_idx, (doc, score, orig_idx) in enumerate(zip(sorted_docs, sorted_scores, original_indices)):
            item = RerankItem(
                object="rerank_result",
                index=rank_idx,
                original_index=orig_idx,
                document=doc if request.return_documents else None,
                score=score if request.return_scores else None
            )
            rerank_items.append(item)
        
        # 构建响应
        response = RerankResponse(
            data=rerank_items,
            model=request.model,
            usage={
                "prompt_tokens": len(request.query.split()) + sum(len(doc.split()) for doc in request.documents),
                "total_tokens": len(request.query.split()) + sum(len(doc.split()) for doc in request.documents)
            }
        )
        
        logger.info(
            f"Rerank 完成 - 模型: {local_model_name}, 候选数: {len(request.documents)}, "
            f"耗时: {time.time() - start_time:.3f}s"
        )
        
        return response
    
    except HTTPException:
        raise
    except ValueError as ve:
        logger.error(f"Rerank 输入错误: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Rerank 失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")

# ------------------------------
# 辅助端点
# ------------------------------
@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "default_embedding_model": settings.DEFAULT_MODEL,
        "default_rerank_model": settings.DEFAULT_RERANK_MODEL,
        "supported_embedding_models": list(settings.SUPPORTED_MODELS.keys()),
        "supported_rerank_models": list(settings.SUPPORTED_RERANK_MODELS.keys()),
        "timestamp": time.time()
    }

@app.get("/models")
async def list_models():
    """列出支持的模型"""
    return {
        "models": [
            {
                "id": model_id,
                "local_path": settings.SUPPORTED_MODELS[model_id],
                "dimension": model_manager.get_model_dimension(model_id)
            }
            for model_id in settings.SUPPORTED_MODELS.keys()
        ],
        "rerank_models": [
            {
                "id": model_id,
                "local_path": settings.SUPPORTED_RERANK_MODELS[model_id],
                "description": f"Rerank 模型 - {model_id}"
            }
            for model_id in settings.SUPPORTED_RERANK_MODELS.keys()
        ]
    }

# ------------------------------
# 模型管理端点（新增，用于卸载模型）
# ------------------------------
@app.delete("/models/{model_name}")
async def unload_model(
    model_name: str,
    model_type: str = "embedding",
    _: bool = Depends(verify_api_key)
):
    """卸载指定模型"""
    try:
        model_manager.unload_model(model_name, model_type)
        return {"status": "success", "message": f"{model_type} 模型 {model_name} 已卸载"}
    except Exception as e:
        logger.error(f"卸载模型失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"卸载失败: {str(e)}")

# ------------------------------
# 启动服务
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"启动 Local Embedding API - 地址: {settings.HOST}:{settings.PORT}")
    logger.info(f"默认模型: {settings.DEFAULT_MODEL}")
    logger.info(f"默认 Rerank 模型: {settings.DEFAULT_RERANK_MODEL}")
    logger.info(f"支持的模型: {list(settings.SUPPORTED_MODELS.keys())}")
    logger.info(f"支持的 Rerank 模型: {list(settings.SUPPORTED_RERANK_MODELS.keys())}")
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower()
    )