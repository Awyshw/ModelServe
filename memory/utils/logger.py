import logging
import sys
import os
from loguru import logger
from config.settings import settings

class InterceptHandler(logging.Handler):
    """
    将标准 logging 的日志转发给 loguru
    """
    def emit(self, record):
        # 获取对应的 Loguru 级别
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 找到调用者帧，确保 loguru 显示正确的调用位置
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def setup_logger() -> None:
    """初始化生产级日志配置"""
    # 移除 loguru 默认处理器
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
            compression="zip",
            enqueue=True,
            serialize=False
        )

    # ---- 关键修改：拦截标准 logging 日志 ----
    # 移除所有已有的根处理器，使用 InterceptHandler
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

# 初始化日志
setup_logger()

# 导出 loguru 实例供应用代码使用
from loguru import logger as log