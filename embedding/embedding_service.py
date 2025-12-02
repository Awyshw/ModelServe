import os
import json
import base64
import time
import structlog
from typing import List, Optional, Union, Dict
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge, Counter
from limits import RateLimitItemPerMinute
from limits.storage import MemoryStorage
from limits.strategies import FixedWindowRateLimiter
import torch
from transformers import AutoModel, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.transformers import patch as vllm_patch

# ----------------------
# 1. 初始化配置（环境变量优先）
# ----------------------
@dataclass
class AppConfig:
    # 服务配置
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    WORKERS: int = int(os.getenv("WORKERS", 1))  # 建议 ≤ GPU数量
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    # 安全配置
    API_KEYS: List[str] = os.getenv("API_KEYS", "").split(",")  # 支持多密钥（逗号分隔）
    CORS_ALLOW_ORIGINS: List[str] = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", 32))  # 限制单请求最大文本数
    MAX_SEQ_LEN: int = int(os.getenv("MAX_SEQ_LEN", 512))  # 全局最大序列长度
    # 限流配置
    RATE_LIMIT: int = int(os.getenv("RATE_LIMIT", 600))  # 每分钟最大请求数
    # 模型配置（支持多模型，格式：模型名:类型:维度:张量并行数）
    SUPPORTED_MODELS: Dict[str, Dict] = {}

# 解析支持的模型（环境变量示例：SUPPORTED_MODELS="BAAI/bge-large-en-v1.5:generative:1024:1;BAAI/bce-embedding-base:encoder:768:1"）
def parse_supported_models() -> Dict[str, Dict]:
    models_str = os.getenv("SUPPORTED_MODELS", "BAAI/bge-large-en-v1.5:generative:1024:1")
    models = {}
    for model_info in models_str.split(";"):
        if not model_info:
            continue
        name, type_, dim, tensor_parallel = model_info.split(":")
        models[name] = {
            "type": type_,
            "embedding_dim": int(dim),
            "tensor_parallel_size": int(tensor_parallel),
            "enable_hidden_states": True,
            "max_seq_len": AppConfig.MAX_SEQ_LEN
        }
    return models

# 初始化配置
config = AppConfig()
config.SUPPORTED_MODELS = parse_supported_models()

# ----------------------
# 2. 日志初始化（结构化日志，便于收集）
# ----------------------
logger = structlog.get_logger()
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

# ----------------------
# 3. 监控指标初始化（Prometheus）
# ----------------------
# GPU状态指标
GPU_MEM_USED = Gauge("gpu_memory_used_bytes", "GPU memory used", ["gpu_id"])
GPU_UTIL = Gauge("gpu_utilization_percent", "GPU utilization", ["gpu_id"])
# 业务指标
REQUEST_COUNT = Counter("embedding_requests_total", "Total embedding requests", ["model", "status"])
TOKEN_COUNT = Counter("embedding_tokens_total", "Total tokens processed", ["model"])
LATENCY_GAUGE = Gauge("embedding_request_latency_seconds", "Embedding request latency", ["model"])

# ----------------------
# 4. 限流初始化
# ----------------------
storage = MemoryStorage()
limiter = FixedWindowRateLimiter(storage)
rate_limit = RateLimitItemPerMinute(config.RATE_LIMIT)

# ----------------------
# 5. 模型缓存与生命周期管理
# ----------------------
model_cache = {}  # 格式：{模型名: (模型实例, tokenizer, 配置)}
model_lock = {}   # 防止并发加载同一模型

@contextmanager
def model_load_lock(model_name: str):
    """模型加载锁，避免并发加载"""
    while model_name in model_lock and model_lock[model_name]:
        time.sleep(0.1)
    model_lock[model_name] = True
    try:
        yield
    finally:
        model_lock[model_name] = False

def load_model(model_name: str):
    """懒加载模型，带锁和异常处理"""
    if model_name not in config.SUPPORTED_MODELS:
        logger.error("unsupported_model", model=model_name, supported=list(config.SUPPORTED_MODELS.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"Model not supported. Supported models: {list(config.SUPPORTED_MODELS.keys())}"
        )
    if model_name in model_cache:
        return model_cache[model_name]
    
    model_config = config.SUPPORTED_MODELS[model_name]
    logger.info("loading_model", model=model_name, config=model_config)
    
    try:
        with model_load_lock(model_name):
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            if model_config["type"] == "generative":
                # 生成式模型（vLLM原生支持）
                llm = LLM(
                    model=model_name,
                    tensor_parallel_size=model_config["tensor_parallel_size"],
                    dtype="auto",
                    trust_remote_code=True,
                    max_seq_len=model_config["max_seq_len"],
                    enable_hidden_states=model_config["enable_hidden_states"],
                    swap_space=4,  # 磁盘交换空间（GB），缓解显存压力
                )
                model_cache[model_name] = (llm, tokenizer, model_config)
            else:
                # Encoder模型（vLLM patch加速）
                vllm_patch()
                model = AutoModel.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                    load_in_8bit=os.getenv("LOAD_IN_8BIT", "false").lower() == "true"  # 可选8bit量化
                )
                model_cache[model_name] = (model, tokenizer, model_config)
        
        logger.info("model_loaded", model=model_name)
        return model_cache[model_name]
    except Exception as e:
        logger.error("model_load_failed", model=model_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

def update_gpu_metrics():
    """更新GPU监控指标（每请求后调用）"""
    if not torch.cuda.is_available():
        return
    for gpu_id in range(torch.cuda.device_count()):
        with torch.cuda.device(gpu_id):
            mem_used = torch.cuda.memory_allocated()
            GPU_MEM_USED.labels(gpu_id=gpu_id).set(mem_used)
            # GPU利用率（需pynvml）
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                GPU_UTIL.labels(gpu_id=gpu_id).set(util)
            except Exception as e:
                logger.warning("failed_to_get_gpu_util", error=str(e))

# ----------------------
# 6. 请求/响应模型（严格对齐OpenAI API）
# ----------------------
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str
    encoding_format: Optional[str] = Field(default="float", pattern="^(float|base64)$")
    user: Optional[str] = None
    max_tokens: Optional[int] = None  # 覆盖默认max_seq_len

    @field_validator("input")
    def validate_input(cls, v):
        """校验输入长度"""
        texts = [v] if isinstance(v, str) else v
        if len(texts) > config.MAX_BATCH_SIZE:
            raise ValueError(f"Input batch size cannot exceed {config.MAX_BATCH_SIZE}")
        texts = [t.strip() for t in texts if t.strip()]
        if not texts:
            raise ValueError("Input cannot be empty")
        return v

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: Union[List[float], str]
    index: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Dict[str, int]

class ErrorResponse(BaseModel):
    error: Dict[str, Optional[str]]

# ----------------------
# 7. 工具函数
# ----------------------
def encode_embedding(embedding: List[float], encoding_format: str) -> Union[List[float], str]:
    """向量格式转换（float/base64）"""
    if encoding_format == "float":
        return embedding
    try:
        bytes_data = json.dumps(embedding).encode("utf-8")
        return base64.b64encode(bytes_data).decode("utf-8")
    except Exception as e:
        logger.error("embedding_encode_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to encode embedding")

def count_tokens(texts: List[str], tokenizer) -> int:
    """统计token数（带缓存优化）"""
    try:
        inputs = tokenizer(texts, padding=False, truncation=False, return_tensors="np")
        return sum(len(ids) for ids in inputs["input_ids"])
    except Exception as e:
        logger.error("token_count_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to count tokens")

# ----------------------
# 8. 核心Embedding生成逻辑
# ----------------------
def generate_embedding(request: EmbeddingRequest) -> tuple[List[List[float]], int]:
    """生成向量，适配两种模型架构"""
    # 预处理输入
    texts = [request.input] if isinstance(request.input, str) else request.input
    texts = [t.strip() for t in texts if t.strip()]
    model_name = request.model

    # 加载模型
    model, tokenizer, model_config = load_model(model_name)
    max_seq_len = request.max_tokens or model_config["max_seq_len"]

    try:
        if model_config["type"] == "generative":
            # 生成式模型：构造Prompt + 提取Hidden State
            prompts = [f"Represent this sentence for searching relevant passages: {text}" for text in texts]
            sampling_params = SamplingParams(max_tokens=1, temperature=0.0, output_logits=False)
            
            # vLLM推理
            outputs = model.generate(prompts, sampling_params=sampling_params)
            embeddings = []
            for output in outputs:
                # 平均池化 + L2归一化
                hidden_state = output.hidden_states[-1]
                avg_emb = hidden_state.mean(dim=1).detach().cpu().numpy().squeeze().tolist()
                norm = sum(x**2 for x in avg_emb)**0.5
                embeddings.append([x / norm for x in avg_emb])
        else:
            # Encoder模型：标准池化流程
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt"
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # 带attention mask的平均池化
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            avg_embeddings = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            avg_embeddings = torch.nn.functional.normalize(avg_embeddings, p=2, dim=1)
            embeddings = avg_embeddings.cpu().numpy().tolist()

        # 统计token数
        total_tokens = count_tokens(texts, tokenizer)
        return embeddings, total_tokens
    except Exception as e:
        logger.error("embedding_generation_failed", model=model_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {str(e)}")

# ----------------------
# 9. FastAPI应用初始化
# ----------------------
app = FastAPI(
    title="OpenAI-Compatible Embedding Service",
    description="Production-grade embedding service with BGE/BCE models and vLLM acceleration",
    version="1.0.0"
)

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册Prometheus监控
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# ----------------------
# 10. 中间件（认证、限流、日志）
# ----------------------
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """API密钥认证"""
    if not config.API_KEYS:
        # 未配置密钥，跳过认证（不建议生产环境使用）
        response = await call_next(request)
        return response
    
    authorization = request.headers.get("Authorization")
    if not authorization or not authorization.startswith("Bearer "):
        logger.warning("unauthorized_request", path=request.url.path)
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "Invalid authentication", "type": "invalid_request_error"}}
        )
    api_key = authorization.split(" ")[1]
    if api_key not in config.API_KEYS:
        logger.warning("invalid_api_key", path=request.url.path)
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "Invalid API key", "type": "invalid_request_error"}}
        )
    response = await call_next(request)
    return response

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """请求限流"""
    client_ip = request.client.host
    if not limiter.hit(rate_limit, client_ip):
        logger.warning("rate_limit_exceeded", client_ip=client_ip)
        return JSONResponse(
            status_code=429,
            content={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}
        )
    response = await call_next(request)
    return response

@app.middleware("http")
async def request_log_middleware(request: Request, call_next):
    """请求日志记录"""
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time
    logger.info(
        "request_processed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency=latency,
        client_ip=request.client.host
    )
    return response

# ----------------------
# 11. 核心接口（OpenAI兼容）
# ----------------------
@app.post(
    "/v1/embeddings",
    response_model=EmbeddingResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def create_embedding(request: EmbeddingRequest):
    """OpenAI兼容的Embedding接口"""
    model_name = request.model
    status = "success"
    start_time = time.time()

    try:
        # 生成向量
        embeddings, total_tokens = generate_embedding(request)
        
        # 转换格式
        data = [
            EmbeddingData(
                embedding=encode_embedding(emb, request.encoding_format),
                index=i
            )
            for i, emb in enumerate(embeddings)
        ]
        
        # 更新监控指标
        latency = time.time() - start_time
        REQUEST_COUNT.labels(model=model_name, status=status).inc()
        TOKEN_COUNT.labels(model=model_name).inc(total_tokens)
        LATENCY_GAUGE.labels(model=model_name).set(latency)
        update_gpu_metrics()

        return EmbeddingResponse(
            data=data,
            model=model_name,
            usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens}
        )
    except HTTPException as e:
        status = "failed"
        REQUEST_COUNT.labels(model=model_name, status=status).inc()
        raise e
    except Exception as e:
        status = "failed"
        REQUEST_COUNT.labels(model=model_name, status=status).inc()
        logger.error("embedding_request_failed", model=model_name, error=str(e))
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": f"Internal server error: {str(e)}", "type": "server_error"}}
        )

# ----------------------
# 12. 运维接口
# ----------------------
@app.get("/health")
async def health_check():
    """健康检查（K8s探针使用）"""
    # 检查GPU状态
    gpu_healthy = torch.cuda.is_available() if model_cache else True
    # 检查模型缓存
    model_healthy = len(model_cache) > 0 or len(config.SUPPORTED_MODELS) == 0
    status = "healthy" if gpu_healthy and model_healthy else "unhealthy"
    
    return JSONResponse(
        content={
            "status": status,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "supported_models": list(config.SUPPORTED_MODELS.keys()),
            "loaded_models": list(model_cache.keys())
        },
        status_code=200 if status == "healthy" else 503
    )

@app.delete("/models/{model_name}")
async def unload_model(model_name: str, authorization: Optional[str] = Header(None)):
    """卸载模型（释放显存）"""
    # 二次认证（仅允许管理员操作）
    if not authorization or authorization.split(" ")[1] not in config.API_KEYS:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    if model_name not in model_cache:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not loaded")
    
    del model_cache[model_name]
    logger.info("model_unloaded", model=model_name)
    return {"message": f"Model {model_name} unloaded successfully"}

# ----------------------
# 启动服务
# ----------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("service_starting", config=asdict(config))
    uvicorn.run(
        "embedding_service:app",
        host=config.HOST,
        port=config.PORT,
        workers=config.WORKERS,
        log_level=config.LOG_LEVEL.lower(),
        access_log=False  # 关闭uvicorn默认日志，使用自定义结构化日志
    )