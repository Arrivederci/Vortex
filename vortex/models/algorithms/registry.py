"""模型算法注册表模块，提供模型注册与获取工具。"""

from __future__ import annotations

from typing import Callable, Dict, Type

from .base import BaseModel

# 中文说明：全局模型注册表，键为模型名称，值为模型类。
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}


def register_model(name: str) -> Callable[[Type[BaseModel]], Type[BaseModel]]:
    """Register a model class under the provided name.

    中文说明：作为装饰器使用，将模型类与名称绑定到注册表，便于通过配置加载。
    """

    def decorator(cls: Type[BaseModel]) -> Type[BaseModel]:
        """Inner decorator that performs the actual registration."""

        # 中文说明：确保被注册对象是 BaseModel 的子类，避免错误使用。
        if not issubclass(cls, BaseModel):
            raise TypeError(f"Model class {cls.__name__} must inherit from BaseModel")

        # 中文说明：直接覆盖同名模型，便于热更新或扩展自定义实现。
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model_class(name: str) -> Type[BaseModel]:
    """Retrieve a registered model class by its name.

    中文说明：根据模型名称返回对应的模型类，若未注册会抛出 KeyError，提醒用户配置错误。
    """

    return MODEL_REGISTRY[name]


# 中文说明：在模块加载时导入内置模型文件，确保默认模型自动注册。
try:  # pragma: no cover - 导入失败会在使用阶段显式暴露
    from . import elastic_net  # noqa: F401
    from . import lightgbm  # noqa: F401
    from . import mlp  # noqa: F401
    from . import random_forest  # noqa: F401
except ImportError:
    # 中文说明：若依赖未安装（例如 LightGBM），保持静默，访问时会抛出明确异常。
    pass


__all__ = ["MODEL_REGISTRY", "register_model", "get_model_class"]
