# config/settings.py
from pydantic import Field, validator
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

# 全局配置实例
settings = Settings()