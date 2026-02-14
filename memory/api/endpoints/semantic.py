# api/endpoints/semantic.py
from fastapi import APIRouter, Depends, Query
from api.models import (
    BaseResponse, PreferenceRequest, PreferenceResponse
)
from core.manager import MemoryManager
from api.dependencies import get_memory_manager
from utils.exceptions import ValidationError
from typing import List, Optional
from utils.logger import log

router = APIRouter(prefix="/semantic", tags=["长期记忆"])

@router.post("/preferences", response_model=BaseResponse)
async def add_user_preference(
    request: PreferenceRequest,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """
    添加/更新用户偏好
    - 支持重复添加（自动更新置信度）
    - 置信度范围：0.0 ~ 1.0
    """
    try:
        memory_manager.semantic.add(
            key=request.key,
            value=request.value,
            confidence=request.confidence
        )
        log.info(f"添加偏好API调用成功，user_id={memory_manager.user_id}，key={request.key}")
        return BaseResponse(
            message=f"偏好[{request.key}]添加成功",
            data={"key": request.key, "value": request.value}
        )
    except ValidationError as e:
        raise
    except Exception as e:
        log.error(f"添加偏好API失败：{str(e)}", exc_info=True)
        return BaseResponse(
            code=500,
            message=f"添加偏好失败：{str(e)}",
            success=False
        )

@router.get("/preferences", response_model=BaseResponse)
async def retrieve_preferences(
    key: Optional[str] = Query(None, description="精确查询的偏好键"),
    query: Optional[str] = Query(None, description="语义检索的查询词"),
    threshold: float = Query(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="语义相似度阈值"
    ),
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """
    检索用户偏好
    - key不为空：精确查询指定键
    - query不为空：语义检索匹配的偏好
    - 都为空：返回所有偏好（按置信度排序）
    """
    try:
        # 确保key和query不同时传（避免歧义）
        if key and query:
            raise ValidationError("只能选择精确查询(key)或语义检索(query)中的一种方式")
        
        preferences = memory_manager.semantic.get(
            key=key,
            query=query,
            threshold=threshold
        )
        log.info(f"检索偏好API调用成功，user_id={memory_manager.user_id}，返回{len(preferences)}条结果")
        return BaseResponse(
            message="检索偏好成功",
            data=preferences
        )
    except ValidationError as e:
        raise
    except Exception as e:
        log.error(f"检索偏好API失败：{str(e)}", exc_info=True)
        return BaseResponse(
            code=500,
            message=f"检索偏好失败：{str(e)}",
            success=False
        )

@router.delete("/preferences/{key}", response_model=BaseResponse)
async def delete_preference(
    key: str,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """删除指定的用户偏好"""
    try:
        memory_manager.semantic.clear(key=key)
        log.info(f"删除偏好API调用成功，user_id={memory_manager.user_id}，key={key}")
        return BaseResponse(
            message=f"偏好[{key}]删除成功",
            data={"key": key}
        )
    except ValidationError as e:
        return BaseResponse(
            code=404,
            message=str(e),
            success=False
        )
    except Exception as e:
        log.error(f"删除偏好API失败：{str(e)}", exc_info=True)
        return BaseResponse(
            code=500,
            message=f"删除偏好失败：{str(e)}",
            success=False
        )

@router.delete("/clear", response_model=BaseResponse)
async def clear_semantic_memory(
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """清空所有用户偏好"""
    try:
        memory_manager.semantic.clear()
        log.warning(f"清空所有偏好API调用成功，user_id={memory_manager.user_id}")
        return BaseResponse(
            message="所有用户偏好已清空"
        )
    except Exception as e:
        log.error(f"清空偏好API失败：{str(e)}", exc_info=True)
        return BaseResponse(
            code=500,
            message=f"清空偏好失败：{str(e)}",
            success=False
        )