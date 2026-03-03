# config/settings.py
from pydantic import Field, SecretStr, validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Literal
import os

class Settings(BaseSettings):
    """生产级配置中心"""
    # 基础配置
    APP_NAME: str = "OpenClaw Memory Service"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Literal["dev", "test", "prod"] = "dev"
    
    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False  # 生产环境关闭热重载

    # 嵌入服务配置（核心新增）
    EMBEDDING_MODE: Literal["local", "openai"] = "local"  # 切换本地/OpenAI模式
    EMBEDDING_CACHE_ENABLED: bool = True  # 嵌入向量缓存（避免重复计算）
    EMBEDDING_CACHE_TTL: int = 3600  # 缓存有效期（秒）

    # 本地嵌入模型配置
    LOCAL_EMBED_MODEL: str = "bge-m3"  # 本地嵌入模型

    # OpenAI Embedding配置
    OPENAI_API_KEY: Optional[SecretStr] = Field(default=None, description="OpenAI API密钥")
    OPENAI_EMBED_MODEL: str = "bge-m3"  # OpenAI嵌入模型
    OPENAI_API_BASE: Optional[str] = Field(default=None, description="自定义OpenAI API端点（如代理）")
    OPENAI_API_TIMEOUT: int = 30  # API超时时间（秒）
    OPENAI_MAX_RETRIES: int = 3  # API重试次数

    # OpenAI LLM配置
    LLM_MODEL_NAME: str = "gpt-3.5-turbo"  # LLM模型名称
    LLM_API_KEY: Optional[SecretStr] = Field(default=None, description="OpenAI API密钥")
    LLM_API_BASE_URL: Optional[str] = Field(default=None, description="自定义OpenAI API端点（如代理）")
    LLM_API_TIMEOUT: int = 30  # API超时时间（秒）
    LLM_MAX_RETRIES: int = 3  # API重试次数
    LLM_TEMPERATURE: float = 0.7  # LLM生成温度

    # DashVector 配置
    DASHVECTOR_API_KEY: Optional[SecretStr] = Field(default=None, description="DashVector API密钥")
    DASHVECTOR_ENDPOINT: str = "https://api.dashvector.com"  # DashVector API端点
    DASHVECTOR_COLLECTION: str = "default"  # DashVector集合名称
    DASHVECTOR_DIMENSION: int = 1024  # DashVector向量维度
    
    # 记忆模块配置
    EPISODIC_MAX_WINDOW: int = 20  # 短期记忆最大窗口
    SEMANTIC_EMBED_MODEL: str = "bge-m3"  # 嵌入模型
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.7  # 语义相似度阈值
    
    # 存储配置
    STORAGE_TYPE: Literal["json", "redis", "leveldb"] = "json"
    STORAGE_PATH: str = "./storage_data"  # JSON/LevelDB存储路径
    REDIS_URL: str = "redis://localhost:6379/0"  # Redis连接地址
    
    # 限流配置
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100  # 每分钟请求数
    RATE_LIMIT_WINDOW: int = 60  # 限流窗口（秒）
    
    # 日志配置
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    LOG_FILE: Optional[str] = "./logs/memory_service.log"
    LOG_ROTATION: str = "1 day"  # 日志轮转
    LOG_RETENTION: str = "7 days"  # 日志保留时间
    
    # CORS配置
    CORS_ORIGINS: list[str] = Field(default=["*"])
    
    model_config = SettingsConfigDict(
        env_file=".env",  # 支持.env文件
        env_file_encoding="utf-8",
        case_sensitive=False  # 环境变量不区分大小写
    )
    
    @validator("STORAGE_PATH")
    def validate_storage_path(cls, v):
        """确保存储目录存在"""
        os.makedirs(v, exist_ok=True)
        return v

    @validator("OPENAI_API_KEY")
    def validate_openai_key(cls, v, values):
        """当使用OpenAI模式时，必须配置API Key"""
        if values.get("EMBEDDING_MODE") == "openai" and not v:
            raise ValueError("使用OpenAI嵌入模式时，必须配置OPENAI_API_KEY")
        return v

# 全局配置实例
settings = Settings()