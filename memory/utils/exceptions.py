# utils/exceptions.py
from fastapi import HTTPException, status
from typing import Optional, Dict, Any

class MemoryError(Exception):
    """记忆模块基础异常"""
    def __init__(self, message: str, code: Optional[int] = None):
        self.message = message
        self.code = code or 500
        super().__init__(self.message)

class StorageError(MemoryError):
    """存储操作异常"""
    def __init__(self, message: str, code: int = 500):
        super().__init__(message, code)

class ValidationError(MemoryError):
    """数据验证异常"""
    def __init__(self, message: str, code: int = 400):
        super().__init__(message, code)

class ResourceNotFoundError(MemoryError):
    """资源不存在异常"""
    def __init__(self, message: str, code: int = 404):
        super().__init__(message, code)

class RateLimitExceededError(MemoryError):
    """限流异常"""
    def __init__(self, message: str = "请求过于频繁，请稍后再试", code: int = 429):
        super().__init__(message, code)

# FastAPI异常处理器
def http_exception_handler(exc: MemoryError) -> HTTPException:
    """转换自定义异常为FastAPI HTTPException"""
    return HTTPException(
        status_code=exc.code,
        detail={
            "code": exc.code,
            "message": exc.message,
            "success": False
        }
    )

# 通用响应模型
class APIResponse:
    """标准化API响应"""
    @staticmethod
    def success(data: Any = None, message: str = "操作成功") -> Dict[str, Any]:
        return {
            "code": 200,
            "message": message,
            "data": data,
            "success": True
        }
    
    @staticmethod
    def error(message: str, code: int = 500, data: Any = None) -> Dict[str, Any]:
        return {
            "code": code,
            "message": message,
            "data": data,
            "success": False
        }