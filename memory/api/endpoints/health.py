# api/endpoints/health.py
from fastapi import APIRouter, Depends
from api.models import HealthCheckResponse, BaseResponse
from core.manager import MemoryManager
from api.dependencies import get_memory_manager
from config.settings import settings
from datetime import datetime

router = APIRouter(prefix="/health", tags=["健康检查"])

@router.get("", response_model=BaseResponse)
async def health_check(memory_manager: MemoryManager = Depends(get_memory_manager)):
    """服务健康检查"""
    health_status = memory_manager.health_check()
    data = HealthCheckResponse(
        status="healthy" if all(health_status.values()) else "unhealthy",
        service="openclaw-memory",
        version=settings.APP_VERSION,
        storage=health_status["storage"],
        timestamp=datetime.now()
    )
    return BaseResponse(data=data)