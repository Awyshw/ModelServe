# api/endpoints/general.py
from fastapi import APIRouter, Depends, Query
from api.models import BaseResponse
from core.manager import MemoryManager
from api.dependencies import get_memory_manager
from utils.logger import log
from datetime import datetime

router = APIRouter(prefix="/general", tags=["通用操作"])

@router.get("/merged-memory", response_model=BaseResponse)
async def get_merged_memory(
    context_window: int = Query(default=10, ge=1, le=20, description="上下文窗口大小"),
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """
    获取融合后的记忆数据（供AI模型使用）
    - 包含最近N轮上下文 + 用户偏好
    - 返回格式化的文本和结构化数据
    """
    try:
        # 获取上下文
        context = memory_manager.get_context(last_n=context_window)
        
        # 获取用户偏好
        preferences = memory_manager.retrieve_preferences()
        
        # 生成格式化文本（供AI直接使用）
        context_str = "【对话上下文】\n"
        for turn in context:
            context_str += f"用户：{turn['user_input']}\n助手：{turn['assistant_response']}\n"
        
        pref_str = "\n【用户偏好】\n"
        for pref in preferences[:5]:  # 限制返回数量
            pref_str += f"{pref['key']}：{pref['value']}（置信度：{pref['confidence']:.2f}）\n"
        
        merged_text = context_str + pref_str
        
        # 结构化数据
        merged_data = {
            "context": context,
            "preferences": preferences,
            "merged_text": merged_text,
            "timestamp": datetime.now().isoformat()
        }
        
        log.info(f"获取融合记忆API调用成功，user_id={memory_manager.user_id}")
        return BaseResponse(
            message="获取融合记忆成功",
            data=merged_data
        )
    except Exception as e:
        log.error(f"获取融合记忆API失败：{str(e)}", exc_info=True)
        return BaseResponse(
            code=500,
            message=f"获取融合记忆失败：{str(e)}",
            success=False
        )

@router.delete("/clear-all", response_model=BaseResponse)
async def clear_all_memory(
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """
    清空当前用户的所有记忆
    - 短期记忆 + 长期记忆 + 临时记忆
    - 高危操作，谨慎使用
    """
    try:
        memory_manager.clear_all()
        log.warning(f"清空所有记忆API调用成功，user_id={memory_manager.user_id}")
        return BaseResponse(
            message="所有记忆已清空（短期+长期+临时）"
        )
    except Exception as e:
        log.error(f"清空所有记忆API失败：{str(e)}", exc_info=True)
        return BaseResponse(
            code=500,
            message=f"清空所有记忆失败：{str(e)}",
            success=False
        )

@router.get("/memory-stats", response_model=BaseResponse)
async def get_memory_stats(
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """获取当前用户的记忆统计信息"""
    try:
        # 获取上下文数量
        context = memory_manager.get_context(last_n=memory_manager.episodic.max_window_size)
        context_count = len(context)
        
        # 获取偏好数量
        preferences = memory_manager.retrieve_preferences()
        pref_count = len(preferences)
        
        # 获取临时数据数量
        transient_data = memory_manager.transient.get(key=None)
        transient_count = len(transient_data) if transient_data else 0
        
        stats = {
            "episodic": {
                "count": context_count,
                "max_window": memory_manager.episodic.max_window_size
            },
            "semantic": {
                "count": pref_count
            },
            "transient": {
                "count": transient_count
            },
            "total": context_count + pref_count + transient_count,
            "timestamp": datetime.now().isoformat()
        }
        
        log.info(f"获取记忆统计API调用成功，user_id={memory_manager.user_id}")
        return BaseResponse(
            message="获取记忆统计成功",
            data=stats
        )
    except Exception as e:
        log.error(f"获取记忆统计API失败：{str(e)}", exc_info=True)
        return BaseResponse(
            code=500,
            message=f"获取记忆统计失败：{str(e)}",
            success=False
        )