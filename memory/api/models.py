# api/models.py
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

# 基础响应模型
class BaseResponse(BaseModel):
    code: int = 200
    message: str = "操作成功"
    data: Optional[Any] = None
    success: bool = True

# 短期记忆模型
class DialogueTurnRequest(BaseModel):
    user_input: str = Field(..., min_length=1, description="用户输入")
    assistant_response: str = Field(..., min_length=1, description="助手回复")
    tags: Optional[List[str]] = Field(default=[], description="对话标签")

class DialogueTurnResponse(BaseModel):
    user_input: str
    assistant_response: str
    timestamp: datetime
    tags: List[str]

# 长期记忆模型
class PreferenceRequest(BaseModel):
    key: str = Field(..., min_length=1, description="偏好键")
    value: str = Field(..., min_length=1, description="偏好值")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="置信度")

class PreferenceResponse(BaseModel):
    key: str
    value: str
    confidence: float
    similarity: Optional[float] = None

# 临时记忆模型
class TransientDataRequest(BaseModel):
    key: str = Field(..., min_length=1, description="临时数据键")
    value: Any = Field(..., description="临时数据值")

# 健康检查模型
class HealthCheckResponse(BaseModel):
    status: str = "healthy"
    service: str = "openclaw-memory"
    version: str
    storage: bool
    timestamp: datetime