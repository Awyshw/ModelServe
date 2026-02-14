# api/endpoints/transient.py
from fastapi import APIRouter, Depends, Path
from api.models import (
    BaseResponse, TransientDataRequest
)
from core.manager import MemoryManager
from api.dependencies import get_memory_manager
from utils.exceptions import ValidationError
from typing import Any, Optional
from utils.logger import log

router = APIRouter(prefix="/transient", tags=["临时记忆"])

@router.post("/data", response_model=BaseResponse)
async def set_transient_data(
    request: TransientDataRequest,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """
    设置临时数据（线程隔离，会话结束后自动清理）
    - 支持任意JSON可序列化的数据类型
    - 仅在当前请求线程中有效
    """
    try:
        memory_manager.transient.add(
            key=request.key,
            value=request.value
        )
        log.info(f"设置临时数据API调用成功，user_id={memory_manager.user_id}，key={request.key}")
        return BaseResponse(
            message=f"临时数据[{request.key}]设置成功",
            data={"key": request.key, "value": request.value}
        )
    except ValidationError as e:
        raise
    except Exception as e:
        log.error(f"设置临时数据API失败：{str(e)}", exc_info=True)
        return BaseResponse(
            code=500,
            message=f"设置临时数据失败：{str(e)}",
            success=False
        )

@router.get("/data/{key}", response_model=BaseResponse)
async def get_transient_data(
    key: str = Path(..., description="临时数据键"),
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """获取指定的临时数据"""
    try:
        value = memory_manager.transient.get(key=key)
        if value is None:
            log.warning(f"临时数据不存在，user_id={memory_manager.user_id}，key={key}")
            return BaseResponse(
                code=404,
                message=f"临时数据[{key}]不存在",
                data=None,
                success=False
            )
        
        log.info(f"获取临时数据API调用成功，user_id={memory_manager.user_id}，key={key}")
        return BaseResponse(
            message="获取临时数据成功",
            data={"key": key, "value": value}
        )
    except Exception as e:
        log.error(f"获取临时数据API失败：{str(e)}", exc_info=True)
        return BaseResponse(
            code=500,
            message=f"获取临时数据失败：{str(e)}",
            success=False
        )

@router.get("/data", response_model=BaseResponse)
async def get_all_transient_data(
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """获取当前用户的所有临时数据"""
    try:
        all_data = memory_manager.transient.get(key=None)
        log.info(f"获取所有临时数据API调用成功，user_id={memory_manager.user_id}，共{len(all_data)}条")
        return BaseResponse(
            message="获取所有临时数据成功",
            data=all_data
        )
    except Exception as e:
        log.error(f"获取所有临时数据API失败：{str(e)}", exc_info=True)
        return BaseResponse(
            code=500,
            message=f"获取所有临时数据失败：{str(e)}",
            success=False
        )

@router.delete("/data/{key}", response_model=BaseResponse)
async def delete_transient_data(
    key: str = Path(..., description="临时数据键"),
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """删除指定的临时数据"""
    try:
        memory_manager.transient.clear(key=key)
        log.info(f"删除临时数据API调用成功，user_id={memory_manager.user_id}，key={key}")
        return BaseResponse(
            message=f"临时数据[{key}]删除成功",
            data={"key": key}
        )
    except ValidationError as e:
        return BaseResponse(
            code=404,
            message=str(e),
            success=False
        )
    except Exception as e:
        log.error(f"删除临时数据API失败：{str(e)}", exc_info=True)
        return BaseResponse(
            code=500,
            message=f"删除临时数据失败：{str(e)}",
            success=False
        )

@router.delete("/clear", response_model=BaseResponse)
async def clear_transient_memory(
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """清空当前用户的所有临时数据"""
    try:
        memory_manager.transient.clear(key=None)
        log.warning(f"清空临时数据API调用成功，user_id={memory_manager.user_id}")
        return BaseResponse(
            message="所有临时数据已清空"
        )
    except Exception as e:
        log.error(f"清空临时数据API失败：{str(e)}", exc_info=True)
        return BaseResponse(
            code=500,
            message=f"清空临时数据失败：{str(e)}",
            success=False
        )