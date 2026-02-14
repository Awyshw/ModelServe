# api/middleware.py
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from utils.logger import log
from utils.exceptions import RateLimitExceededError
from config.settings import settings
import time
import uuid
from typing import Dict
from collections import defaultdict

# 限流存储（生产环境建议用Redis）
rate_limit_storage: Dict[str, List[float]] = defaultdict(list)

def add_middlewares(app):
    """添加生产级中间件"""
    # 1. CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 2. GZip压缩中间件
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # 3. 自定义日志中间件
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        # 生成请求ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # 请求信息
        start_time = time.time()
        client_ip = request.client.host
        method = request.method
        url = str(request.url)
        
        log.info(
            f"请求开始 | ID: {request_id} | IP: {client_ip} | "
            f"Method: {method} | URL: {url}"
        )
        
        # 执行请求
        try:
            response = await call_next(request)
        except Exception as e:
            log.error(
                f"请求异常 | ID: {request_id} | 异常: {str(e)}"
            )
            raise
        
        # 响应信息
        process_time = time.time() - start_time
        status_code = response.status_code
        
        log.info(
            f"请求结束 | ID: {request_id} | 耗时: {process_time:.4f}s | "
            f"状态码: {status_code}"
        )
        
        # 添加请求ID响应头
        response.headers["X-Request-ID"] = request_id
        return response
    
    # 4. 限流中间件
    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        if not settings.RATE_LIMIT_ENABLED:
            return await call_next(request)
        
        # 获取用户标识（优先user_id，其次IP）
        user_id = request.headers.get("user_id") or request.client.host
        now = time.time()
        
        # 清理过期请求记录
        rate_limit_storage[user_id] = [
            t for t in rate_limit_storage[user_id]
            if now - t < settings.RATE_LIMIT_WINDOW
        ]
        
        # 检查限流
        if len(rate_limit_storage[user_id]) >= settings.RATE_LIMIT_REQUESTS:
            raise RateLimitExceededError()
        
        # 记录请求时间
        rate_limit_storage[user_id].append(now)
        
        return await call_next(request)
    
    return app