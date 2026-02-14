# utils/logger.py
import logging
from loguru import logger
from config.settings import settings
import sys
import os

def setup_logger() -> None:
    """初始化生产级日志配置"""
    # 移除默认处理器
    logger.remove()
    
    # 控制台输出（开发环境）
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stdout,
        format=console_format,
        level=settings.LOG_LEVEL,
        filter=lambda record: settings.ENVIRONMENT != "prod"
    )
    
    # 文件输出（所有环境）
    if settings.LOG_FILE:
        # 确保日志目录存在
        log_dir = os.path.dirname(settings.LOG_FILE)
        os.makedirs(log_dir, exist_ok=True)
        
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} - "
            "{message}"
        )
        logger.add(
            settings.LOG_FILE,
            format=file_format,
            level=settings.LOG_LEVEL,
            rotation=settings.LOG_ROTATION,
            retention=settings.LOG_RETENTION,
            compression="zip",  # 压缩旧日志
            enqueue=True,  # 异步日志，提高性能
            serialize=False  # 生产环境可设为True输出JSON
        )
    
    # 替换标准logging为loguru
    logging.getLogger = lambda name: logger.bind(name=name)
    logging.info = logger.info
    logging.error = logger.error
    logging.warning = logger.warning
    logging.debug = logger.debug

# 初始化日志
setup_logger()

# 导出logger实例
from loguru import logger as log