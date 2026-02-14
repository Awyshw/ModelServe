# api/dependencies.py
from fastapi import Depends, Header, HTTPException
from core.manager import MemoryManager
from utils.logger import log
from typing import Optional

def get_user_id(user_id: Optional[str] = Header(None, description="用户ID")) -> str:
    """获取用户ID（从Header）"""
    if not user_id:
        raise HTTPException(
            status_code=400,
            detail={"code": 400, "message": "Header中必须包含user_id", "success": False}
        )
    return user_id

def get_memory_manager(user_id: str = Depends(get_user_id)) -> MemoryManager:
    """获取MemoryManager实例（依赖注入）"""
    try:
        return MemoryManager(user_id=user_id)
    except Exception as e:
        log.error(f"MemoryManager初始化失败：{str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"code": 500, "message": "记忆服务初始化失败", "success": False}
        )