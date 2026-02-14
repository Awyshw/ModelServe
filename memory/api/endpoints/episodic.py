# api/endpoints/episodic.py
from fastapi import APIRouter, Depends, Query
from api.models import (
    BaseResponse, DialogueTurnRequest, DialogueTurnResponse
)
from core.manager import MemoryManager
from api.dependencies import get_memory_manager
from utils.exceptions import ValidationError
from typing import List

router = APIRouter(prefix="/episodic", tags=["短期记忆"])

@router.post("/turns", response_model=BaseResponse)
async def add_dialogue_turn(
    request: DialogueTurnRequest,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """添加对话轮次"""
    memory_manager.add_dialogue_turn(
        user_input=request.user_input,
        assistant_response=request.assistant_response,
        tags=request.tags
    )
    return BaseResponse(message="对话轮次添加成功")

@router.get("/context", response_model=BaseResponse)
async def get_context(
    last_n: int = Query(default=10, ge=1, le=20),
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """获取最近N轮上下文"""
    context = memory_manager.get_context(last_n=last_n)
    return BaseResponse(data=context)

@router.delete("/clear", response_model=BaseResponse)
async def clear_episodic_memory(
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """清空短期记忆"""
    memory_manager.episodic.clear()
    return BaseResponse(message="短期记忆已清空")