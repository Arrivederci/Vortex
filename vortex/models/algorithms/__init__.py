"""模型算法子包初始化模块，导出通用模型接口与注册表。"""

from __future__ import annotations

from .registry import MODEL_REGISTRY, get_model_class, register_model

# 中文说明：导入内置模型模块以触发装饰器注册，便于通过配置直接使用。
from . import elastic_net  # noqa: F401
from . import lightgbm  # noqa: F401
from . import mlp  # noqa: F401
from . import random_forest  # noqa: F401

__all__ = [
    "MODEL_REGISTRY",
    "get_model_class",
    "register_model",
]
