# core/episodic.py
from core.abstractions import BaseMemory
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any
from utils.logger import log
from utils.exceptions import StorageError, ValidationError

@dataclass
class DialogueTurn:
    """单轮对话数据结构（生产级）"""
    user_input: str
    assistant_response: str
    timestamp: datetime = datetime.now()
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        
        # 数据验证
        if not isinstance(self.user_input, str) or len(self.user_input.strip()) == 0:
            raise ValidationError("用户输入不能为空")
        if not isinstance(self.assistant_response, str) or len(self.assistant_response.strip()) == 0:
            raise ValidationError("助手回复不能为空")

class EpisodicMemory(BaseMemory):
    """短期记忆：对话上下文管理（环形队列+生产级优化）"""
    def __init__(self, user_id: str, max_window_size: int = 20, storage: Optional[Any] = None):
        super().__init__(user_id=user_id, storage=storage)
        self.max_window_size = max_window_size
        self.context_queue: List[DialogueTurn] = []
        
        # 初始化时加载持久化数据
        if self.storage:
            self._load_from_storage()
        log.debug(f"EpisodicMemory初始化成功，user_id={user_id}，窗口大小={max_window_size}")

    def add(self, key: str, value: Dict[str, Any], **kwargs) -> None:
        """
        添加对话轮次（适配抽象接口）
        :param key: 固定为"turn"
        :param value: 包含user_input/assistant_response/tags的字典
        """
        if key != "turn":
            raise ValidationError(f"EpisodicMemory仅支持key='turn'，当前key={key}")
        
        try:
            # 转换为DialogueTurn对象（自动触发数据验证）
            turn = DialogueTurn(
                user_input=value["user_input"],
                assistant_response=value["assistant_response"],
                tags=value.get("tags", [])
            )
            
            with self.lock:
                # 环形队列：超出窗口移除最旧数据
                if len(self.context_queue) >= self.max_window_size:
                    removed_turn = self.context_queue.pop(0)
                    log.debug(f"移除最旧对话轮次：user_input={removed_turn.user_input[:20]}...")
                
                self.context_queue.append(turn)
                log.debug(f"添加对话轮次成功，当前队列长度={len(self.context_queue)}")
            
            # 持久化（异步/增量，减少IO开销）
            if self.storage:
                self._save_to_storage()
                
        except KeyError as e:
            raise ValidationError(f"缺失必要字段：{str(e)}")
        except Exception as e:
            log.error(f"添加对话轮次失败：{str(e)}", exc_info=True)
            raise

    def get(self, key: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        获取上下文数据
        :param kwargs: last_n=int - 获取最近n轮
        :return: 序列化后的对话轮次列表
        """
        try:
            last_n = kwargs.get("last_n", self.max_window_size)
            if not isinstance(last_n, int) or last_n <= 0:
                raise ValidationError(f"last_n必须为正整数，当前值={last_n}")
            
            with self.lock:
                # 限制获取数量不超过队列长度和最大窗口
                take = min(last_n, len(self.context_queue), self.max_window_size)
                start_idx = max(0, len(self.context_queue) - take)
                result = self.context_queue[start_idx:]
            
            # 序列化为字典（便于API返回）
            serialized = [
                {
                    "user_input": turn.user_input,
                    "assistant_response": turn.assistant_response,
                    "timestamp": turn.timestamp.isoformat(),
                    "tags": turn.tags
                }
                for turn in result
            ]
            log.debug(f"获取上下文成功，user_id={self.user_id}，获取{len(serialized)}轮")
            return serialized
        
        except Exception as e:
            log.error(f"获取上下文失败：{str(e)}", exc_info=True)
            raise

    def clear(self, key: Optional[str] = None) -> None:
        """清空短期记忆（key=None清空全部）"""
        try:
            with self.lock:
                self.context_queue.clear()
                log.warning(f"清空短期记忆成功，user_id={self.user_id}")
            
            # 清理存储后端数据
            if self.storage:
                self.storage.delete(self.user_id, "episodic")
                
        except Exception as e:
            log.error(f"清空短期记忆失败：{str(e)}", exc_info=True)
            raise

    def _save_to_storage(self) -> None:
        """持久化到存储后端（序列化DialogueTurn）"""
        try:
            with self.lock:
                serialized = [asdict(turn) for turn in self.context_queue]
                # 替换datetime为ISO字符串（便于JSON序列化）
                for item in serialized:
                    item["timestamp"] = item["timestamp"].isoformat()
                
                self.storage.save(self.user_id, "episodic", serialized)
            log.debug(f"短期记忆持久化成功，user_id={self.user_id}")
            
        except Exception as e:
            log.error(f"短期记忆持久化失败：{str(e)}", exc_info=True)
            raise StorageError(f"短期记忆存储失败：{str(e)}")

    def _load_from_storage(self) -> None:
        """从存储后端加载数据"""
        try:
            serialized = self.storage.load(self.user_id, "episodic")
            if not serialized:
                log.debug(f"无持久化短期记忆数据，user_id={self.user_id}")
                return
            
            with self.lock:
                self.context_queue = [
                    DialogueTurn(
                        user_input=item["user_input"],
                        assistant_response=item["assistant_response"],
                        timestamp=datetime.fromisoformat(item["timestamp"]),
                        tags=item["tags"]
                    )
                    for item in serialized
                ]
            log.debug(f"短期记忆加载成功，user_id={self.user_id}，加载{len(self.context_queue)}轮")
            
        except Exception as e:
            log.error(f"短期记忆加载失败：{str(e)}", exc_info=True)
            raise StorageError(f"短期记忆加载失败：{str(e)}")