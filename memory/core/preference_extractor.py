# core/preference_extractor.py
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, ValidationError
from utils.logger import log
from utils.exceptions import ValidationError as MemoryValidationError
from config.settings import settings
from openai import OpenAI

# 初始化大模型客户端（复用配置）
llm_client = OpenAI(
    api_key=settings.LLM_API_KEY.get_secret_value() if settings.LLM_API_KEY else None,
    base_url=settings.LLM_API_BASE_URL,
)

# 偏好数据模型（标准化）
class ExtractedPreference(BaseModel):
    key: str  # 偏好键（标准化，如favorite_product）
    value: str  # 偏好值
    confidence: float  # 提取置信度（0-1）
    source_text: str  # 原始对话文本（溯源）
    extract_rule: str  # 提取规则（如“产品偏好”）

# 偏好提取器
class PreferenceExtractor:
    """从对话上下文中自动提取用户偏好（基于大模型）"""
    def __init__(self):
        # 偏好提取提示词（核心：定义提取规则）
        self.extract_prompt = """
        你是一个用户偏好提取专家，请从用户对话文本中提取有价值的用户偏好，遵循以下规则：
        
        # 提取规则
        1. 只提取**稳定的、长期的**用户偏好，临时需求（如“帮我查订单”）不提取；
        2. Key必须使用标准化命名（小写+下划线），如：
           - 产品偏好：favorite_product / disliked_product
           - 服务偏好：preferred_service / contact_method
           - 时间偏好：preferred_time / avoid_time
           - 个性化设置：notification_type / language_preference
        3. Value必须是具体、可落地的文本，避免模糊表述；
        4. 置信度（confidence）：0-1，越高表示越确定这是用户长期偏好；
        5. 无有效偏好时，返回空列表；
        6. 输出格式必须是JSON数组，每个元素包含：key、value、confidence、source_text、extract_rule。
        
        # 禁止提取
        - 敏感信息：手机号、身份证、银行卡号等；
        - 临时请求：“帮我解决问题”“我要投诉”等；
        - 无意义内容：“好的”“谢谢”“再见”等。
        
        # 用户对话文本
        {context}
        
        # 输出示例
        [
            {{
                "key": "favorite_product",
                "value": "智能音箱",
                "confidence": 0.95,
                "source_text": "我最喜欢你们家的智能音箱，用了很久了",
                "extract_rule": "产品偏好"
            }}
        ]
        """

    def extract_from_context(self, context: List[Dict[str, Any]]) -> List[ExtractedPreference]:
        """
        从对话上下文中提取用户偏好
        :param context: 对话上下文（MemoryManager.get_context返回的结果）
        :return: 标准化的偏好列表
        """
        try:
            # 1. 上下文预处理：拼接用户输入，过滤空内容
            user_texts = []
            for turn in context:
                user_input = turn.get("user_input", "")
                if user_input and isinstance(user_input, str) and user_input.strip():
                    user_texts.append(user_input.strip())
            
            if not user_texts:
                log.debug("无有效用户输入，跳过偏好提取")
                return []
            
            # 2. 拼接上下文文本（取最近10轮）
            context_text = "\n".join(user_texts[-10:])
            log.debug(f"待提取的用户上下文：{context_text[:200]}...")
            
            # 3. 调用大模型提取偏好
            prompt = self.extract_prompt.format(context=context_text)
            response = llm_client.chat.completions.create(
                model=settings.LLM_MODEL_NAME,  # 可替换为本地大模型
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # 低温度，保证提取结果稳定
                max_tokens=2048,
                response_format={"type": "json_object"}  # 强制JSON输出
            )
            
            # 4. 解析大模型输出
            extract_result = eval(response.choices[0].message.content.strip())  # 或用json.loads
            if not isinstance(extract_result, list):
                log.warning("偏好提取结果不是列表，跳过")
                return []
            
            # 5. 数据验证和标准化
            valid_preferences = []
            for item in extract_result:
                try:
                    # 用Pydantic验证数据格式
                    preference = ExtractedPreference(**item)
                    
                    # 额外验证：置信度≥0.5才保留（过滤低置信度偏好）
                    if preference.confidence >= 0.5:
                        valid_preferences.append(preference)
                    else:
                        log.debug(f"低置信度偏好跳过：{preference.key}={preference.value}（置信度{preference.confidence}）")
                except (ValidationError, MemoryValidationError) as e:
                    log.warning(f"偏好数据验证失败：{str(e)}，原始数据：{item}")
                    continue
            
            log.info(f"成功提取{len(valid_preferences)}个有效用户偏好")
            return valid_preferences
        
        except Exception as e:
            log.error(f"偏好提取失败：{str(e)}", exc_info=True)
            return []

    def deduplicate_preferences(
        self,
        new_preferences: List[ExtractedPreference],
        existing_preferences: List[Dict[str, Any]]
    ) -> List[ExtractedPreference]:
        """
        偏好去重（基于Key+语义相似度）
        :param new_preferences: 新提取的偏好
        :param existing_preferences: 已有的偏好（从记忆模块读取）
        :return: 去重后的新偏好
        """
        if not new_preferences:
            return []
        
        # 1. 构建已有偏好的Key-Value映射
        existing_keys = {pref["key"]: pref["value"] for pref in existing_preferences}
        
        # 2. 去重逻辑
        unique_preferences = []
        from core.embedding_service import get_embedding_service
        embed_service = get_embedding_service()
        
        for new_pref in new_preferences:
            # 情况1：Key不存在，直接保留
            if new_pref.key not in existing_keys:
                unique_preferences.append(new_pref)
                continue
            
            # 情况2：Key存在，计算Value的语义相似度
            existing_value = existing_keys[new_pref.key]
            new_embed = embed_service.generate_embedding(new_pref.value)
            existing_embed = embed_service.generate_embedding(existing_value)
            
            # 计算余弦相似度
            similarity = self._cosine_similarity(new_embed, existing_embed)
            
            # 相似度<0.8，认为是新偏好（更新）；≥0.8，跳过（重复）
            if similarity < 0.8:
                log.info(f"偏好更新：{new_pref.key} 从 {existing_value} 改为 {new_pref.value}（相似度{similarity:.4f}）")
                unique_preferences.append(new_pref)
            else:
                log.debug(f"偏好重复跳过：{new_pref.key}={new_pref.value}（相似度{similarity:.4f}）")
        
        return unique_preferences

    @staticmethod
    def _cosine_similarity(vec1, vec2) -> float:
        """计算余弦相似度（复用记忆模块逻辑）"""
        import numpy as np
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        
        similarity = float(np.dot(vec1, vec2) / (norm1 * norm2))
        return max(0.0, min(1.0, similarity))