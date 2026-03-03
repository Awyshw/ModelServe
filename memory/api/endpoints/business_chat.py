from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel
from api.dependencies import get_memory_manager
from api.models import (
    ChatRequest, ChatResponse
)
from config.settings import settings
from core.manager import MemoryManager
from utils.exceptions import StorageError, ValidationError
from typing import Optional, List, Dict, Any
from utils.logger import log

from openai import OpenAI
import time

# 初始化大模型客户端
llm_client = OpenAI(
    api_key=settings.LLM_API_KEY.get_secret_value() if settings.LLM_API_KEY else None,
    base_url=settings.LLM_API_BASE_URL,
)

router = APIRouter(prefix="/business", tags=["业务场景对话 chat"])

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """智能客服对话接口（结合记忆模块+大模型）"""
    try:
        # 1. 验证输入
        if not request.user_input.strip():
            raise HTTPException(status_code=400, detail="用户输入不能为空")
        
        # 2. 获取用户记忆管理器
        memory_manager = get_memory_manager(request.user_id)
        
        # 3. 读取记忆数据
        # 3.1 读取最近10轮对话上下文（短期记忆）
        context = memory_manager.get_context(last_n=10)
        # 3.2 读取用户偏好（语义检索，匹配当前输入）
        preferences = memory_manager.retrieve_preferences(
            query=request.user_input,
            threshold=0.7
        )
        
        # 4. 构建大模型提示词（结合记忆）
        prompt = f"""
        你是一名智能客服助手，请结合用户的对话上下文和偏好，友好、准确地回复用户。
        
        # 用户对话上下文（按时间倒序）：
        {context}
        
        # 用户长期偏好：
        {preferences if preferences else "无"}
        
        # 用户当前问题：
        {request.user_input}
        
        # 回复要求：
        1. 基于上下文和偏好，回复要连贯、个性化；
        2. 避免重复提问，若用户之前问过相同问题，直接回复；
        3. 语气友好，符合客服规范；
        4. 只回复用户的问题，不要添加无关内容。
        """
        
        # 5. 调用大模型生成回复
        llm_response = llm_client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,  # 可替换为本地大模型
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=2048
        )
        assistant_response = llm_response.choices[0].message.content.strip()
        
        # 6. 保存当前对话到短期记忆
        memory_manager.add_dialogue_turn(
            user_input=request.user_input,
            assistant_response=assistant_response,
            tags=["customer_service", f"session:{request.session_id}"]
        )
        # 新增：主动触发偏好提取（补充每3轮的自动提取）
        extracted_preferences = memory_manager.auto_extract_and_save_preferences()
        
        # 7. 构建响应
        response = ChatResponse(
            response=assistant_response,
            context_used=context,
            preferences_used=preferences,
            extracted_preferences=extracted_preferences,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        log.info(f"对话完成，user_id={request.user_id}，session_id={request.session_id}")
        return response
    
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except StorageError as e:
        raise HTTPException(status_code=500, detail=f"记忆模块错误：{str(e)}")
    except Exception as e:
        log.error(f"客服对话失败：{str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="服务器内部错误")