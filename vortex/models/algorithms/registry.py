"""模型算法注册表模块，提供模型注册与获取工具。"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
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


def _auto_import_algorithms() -> None:
    """Import built-in algorithm modules so they register on access."""

    # 中文说明：遍历算法目录并导入其中的实现模块，确保默认模型自动注册。
    package_path = Path(__file__).resolve().parent
    for module_info in pkgutil.iter_modules([str(package_path)]):
        if module_info.ispkg or module_info.name in {"base", "registry"}:
            continue
        module_name = f"{__name__.rsplit('.', 1)[0]}.{module_info.name}"
        try:
            importlib.import_module(module_name)
        except ImportError:
            # 中文说明：当可选依赖缺失时跳过该模块，使用时会得到明确的异常提示。
            continue


_auto_import_algorithms()


__all__ = ["MODEL_REGISTRY", "register_model", "get_model_class"]
