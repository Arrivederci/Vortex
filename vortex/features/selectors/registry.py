"""特征选择器注册表模块，提供统一的注册装饰器与查询接口。"""

from __future__ import annotations

from typing import Callable, Dict, Type

from ..base import FeatureSelectorBase

# 中文说明：全局特征选择器注册表，键为选择器名称，值为对应的实现类。
SELECTOR_REGISTRY: Dict[str, Type[FeatureSelectorBase]] = {}


def register_selector(name: str) -> Callable[[Type[FeatureSelectorBase]], Type[FeatureSelectorBase]]:
    """Register a selector class under the provided name."""

    # 中文说明：作为装饰器使用，将特征选择器类绑定到指定名称，便于配置化调用。

    def decorator(cls: Type[FeatureSelectorBase]) -> Type[FeatureSelectorBase]:
        """Inner decorator storing the selector into the registry."""

        # 中文说明：确保被注册的类继承自 FeatureSelectorBase，避免误用普通类。
        if not issubclass(cls, FeatureSelectorBase):
            raise TypeError(f"Selector class {cls.__name__} must inherit from FeatureSelectorBase")
        SELECTOR_REGISTRY[name] = cls
        return cls

    return decorator


def get_selector_class(name: str) -> Type[FeatureSelectorBase]:
    """Retrieve a registered selector class by its name."""

    # 中文说明：根据名称返回对应的特征选择器类，若不存在则抛出 KeyError，提示配置错误。
    return SELECTOR_REGISTRY[name]


__all__ = ["SELECTOR_REGISTRY", "register_selector", "get_selector_class"]
