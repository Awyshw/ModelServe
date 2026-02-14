# main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from config.settings import settings
from utils.logger import log
from utils.exceptions import MemoryError, http_exception_handler
from api.middleware import add_middlewares
from api.endpoints import health, episodic, semantic, transient, general
from api.models import BaseResponse

# 创建FastAPI应用（生产级配置）
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    docs_url="/docs" if settings.ENVIRONMENT != "prod" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "prod" else None,
    openapi_url="/openapi.json" if settings.ENVIRONMENT != "prod" else None
)

# 添加中间件
app = add_middlewares(app)

# 注册路由
app.include_router(health.router)
app.include_router(episodic.router)
app.include_router(semantic.router)
app.include_router(transient.router)
app.include_router(general.router)

# 全局异常处理
@app.exception_handler(MemoryError)
async def memory_exception_handler(request: Request, exc: MemoryError):
    """自定义记忆异常处理"""
    return JSONResponse(
        status_code=exc.code,
        content={
            "code": exc.code,
            "message": exc.message,
            "success": False,
            "request_id": getattr(request.state, "request_id", None)
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler_custom(request: Request, exc: HTTPException):
    """标准化HTTP异常响应"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.status_code,
            "message": exc.detail.get("message", str(exc.detail)),
            "success": False,
            "request_id": getattr(request.state, "request_id", None)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理（兜底）"""
    log.error(f"未捕获异常：{str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "code": 500,
            "message": "服务器内部错误",
            "success": False,
            "request_id": getattr(request.state, "request_id", None)
        }
    )

# 根路由
@app.get("/", response_model=BaseResponse)
async def root():
    """根路由"""
    return BaseResponse(
        message=f"{settings.APP_NAME} 运行中",
        data={"version": settings.APP_VERSION, "environment": settings.ENVIRONMENT}
    )

# 启动函数
def main():
    """生产级启动"""
    import uvicorn
    log.info(
        f"启动{settings.APP_NAME} | 版本: {settings.APP_VERSION} | "
        f"环境: {settings.ENVIRONMENT} | 地址: http://{settings.HOST}:{settings.PORT}"
    )
    
    # 使用uvicorn启动（生产级服务器）
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=4 if settings.ENVIRONMENT == "prod" else 1,  # 生产环境多进程
        log_level=settings.LOG_LEVEL.lower()
    )

if __name__ == "__main__":
    main()