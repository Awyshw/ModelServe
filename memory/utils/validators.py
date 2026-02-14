# utils/validators.py
"""生产级数据验证工具（通用可复用）"""
import re
from typing import Any, Optional, List, Dict
from utils.exceptions import ValidationError
from utils.logger import log

# 通用正则表达式
USER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{3,64}$")  # 用户ID规则：3-64位字母/数字/下划线/短横线
KEY_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{1,128}$")     # 键名规则：1-128位字母/数字/下划线/点/短横线
TAG_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{1,32}$")      # 标签规则：1-32位字母/数字/下划线/点/短横线

def validate_user_id(user_id: str) -> str:
    """
    验证用户ID格式
    :param user_id: 用户ID
    :return: 验证通过的user_id
    :raise ValidationError: 验证失败
    """
    if not isinstance(user_id, str):
        raise ValidationError("用户ID必须为字符串类型")
    
    user_id = user_id.strip()
    if not USER_ID_PATTERN.match(user_id):
        raise ValidationError(
            "用户ID格式无效：必须为3-64位字母、数字、下划线(_)或短横线(-)"
        )
    
    log.debug(f"用户ID验证通过：{user_id}")
    return user_id

def validate_key(key: str, field_name: str = "键名") -> str:
    """
    验证键名格式
    :param key: 待验证的键名
    :param field_name: 字段名称（用于错误提示）
    :return: 验证通过的键名
    """
    if not isinstance(key, str):
        raise ValidationError(f"{field_name}必须为字符串类型")
    
    key = key.strip()
    if not KEY_PATTERN.match(key):
        raise ValidationError(
            f"{field_name}格式无效：必须为1-128位字母、数字、下划线(_)、点(.)或短横线(-)"
        )
    
    log.debug(f"{field_name}验证通过：{key}")
    return key

def validate_tags(tags: Optional[List[str]]) -> List[str]:
    """
    验证标签列表格式
    :param tags: 标签列表
    :return: 去重并验证后的标签列表
    """
    if tags is None:
        return []
    
    if not isinstance(tags, list):
        raise ValidationError("标签必须为列表类型")
    
    # 空列表直接返回
    if not tags:
        return []
    
    # 验证每个标签
    validated_tags = []
    for tag in tags:
        if not isinstance(tag, str):
            raise ValidationError(f"标签必须为字符串类型，当前值：{tag}")
        
        tag = tag.strip()
        if not TAG_PATTERN.match(tag):
            raise ValidationError(
                f"标签格式无效：{tag}（必须为1-32位字母、数字、下划线(_)、点(.)或短横线(-)）"
            )
        
        if tag not in validated_tags:  # 去重
            validated_tags.append(tag)
    
    log.debug(f"标签验证通过：{validated_tags}")
    return validated_tags

def validate_string_not_empty(
    value: Any,
    field_name: str = "字段",
    min_length: int = 1,
    max_length: Optional[int] = None
) -> str:
    """
    验证字符串非空且长度符合要求
    :param value: 待验证的值
    :param field_name: 字段名称
    :param min_length: 最小长度（默认1）
    :param max_length: 最大长度（None表示不限制）
    :return: 验证通过的字符串
    """
    if not isinstance(value, str):
        raise ValidationError(f"{field_name}必须为字符串类型")
    
    value = value.strip()
    if len(value) < min_length:
        raise ValidationError(f"{field_name}长度不能小于{min_length}")
    
    if max_length and len(value) > max_length:
        raise ValidationError(f"{field_name}长度不能超过{max_length}")
    
    log.debug(f"{field_name}验证通过（长度：{len(value)}）")
    return value

def validate_numeric_range(
    value: Any,
    field_name: str = "数值",
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> float:
    """
    验证数值在指定范围内
    :param value: 待验证的数值
    :param field_name: 字段名称
    :param min_value: 最小值（None表示不限制）
    :param max_value: 最大值（None表示不限制）
    :return: 验证通过的数值
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{field_name}必须为数值类型")
    
    if min_value is not None and value < min_value:
        raise ValidationError(f"{field_name}不能小于{min_value}")
    
    if max_value is not None and value > max_value:
        raise ValidationError(f"{field_name}不能大于{max_value}")
    
    log.debug(f"{field_name}验证通过：{value}（范围：{min_value}~{max_value}）")
    return value

def validate_dict_structure(
    data: Any,
    required_keys: List[str],
    field_name: str = "数据"
) -> Dict[str, Any]:
    """
    验证字典包含指定的必填键
    :param data: 待验证的字典
    :param required_keys: 必填键列表
    :param field_name: 字段名称
    :return: 验证通过的字典
    """
    if not isinstance(data, dict):
        raise ValidationError(f"{field_name}必须为字典类型")
    
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise ValidationError(
            f"{field_name}缺失必填字段：{', '.join(missing_keys)}"
        )
    
    log.debug(f"{field_name}结构验证通过，包含所有必填字段：{required_keys}")
    return data

def validate_list_items(
    data: Any,
    item_type: type,
    field_name: str = "列表",
    min_length: int = 0,
    max_length: Optional[int] = None
) -> List[Any]:
    """
    验证列表项类型和长度
    :param data: 待验证的列表
    :param item_type: 列表项类型
    :param field_name: 字段名称
    :param min_length: 最小长度
    :param max_length: 最大长度
    :return: 验证通过的列表
    """
    if not isinstance(data, list):
        raise ValidationError(f"{field_name}必须为列表类型")
    
    if len(data) < min_length:
        raise ValidationError(f"{field_name}长度不能小于{min_length}")
    
    if max_length and len(data) > max_length:
        raise ValidationError(f"{field_name}长度不能超过{max_length}")
    
    # 验证每个项的类型
    for idx, item in enumerate(data):
        if not isinstance(item, item_type):
            raise ValidationError(
                f"{field_name}第{idx+1}项类型错误：期望{item_type.__name__}，实际{type(item).__name__}"
            )
    
    log.debug(f"{field_name}验证通过：{len(data)}个{item_type.__name__}类型项")
    return data