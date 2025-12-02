from pydantic_settings import BaseSettings
from typing import Dict, Optional, List

class Settings(BaseSettings):
    # 服务配置
    HOST: str = "0.0.0.0"
    PORT: int = 9090
    WORKERS: int = 1
    LOG_LEVEL: str = "info"
    
    # 模型配置
    DEFAULT_MODEL: str = "bce-m3"
    MODEL_DIR: str = "./models"
    SUPPORTED_MODELS: Dict[str, str] = {
        # 模型名称映射：OpenAI模型名 -> 本地模型路径/名称
        "text-embedding-ada-002": "bge-base-zh",  # 默认映射
        "bge-base-zh": "bge-base-zh",
        "bge-large-zh": "bge-large-zh",
        "bge-small-zh": "bge-small-zh",
        "bce-embedding-base_v1": "bce-embedding-base_v1",
        "bce-embedding-large_v1": "bce-embedding-large_v1",
        "bce-m3": "bge-m3",
    }
    
    # 嵌入配置
    DEFAULT_EMBEDDING_DIM: Optional[int] = None  # 自动获取
    NORMALIZE_EMBEDDINGS: bool = True  # 向量归一化（OpenAI默认开启）
    MAX_BATCH_SIZE: int = 32  # 批量处理大小
    
    # API 配置
    API_KEY_REQUIRED: bool = False
    VALID_API_KEYS: List[str] = ["sk-local-embedding-api-key", "EMPTY"]

settings = Settings()