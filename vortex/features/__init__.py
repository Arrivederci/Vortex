"""特征工程子包初始化模块，负责自动导入子模块并导出公共接口。"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from .base import FeatureSelectorBase, FeatureSelectorPipeline, build_pipeline
from .selectors import (
    SELECTOR_REGISTRY,
    get_selector_class,
    register_selector,
)


def _auto_import_feature_packages() -> None:
    """Eagerly import subpackages so their registrations take effect."""

    # 中文说明：遍历特征工程子目录的所有子包并导入，确保新增特征模块的装饰器立即生效。
    package_path = Path(__file__).resolve().parent
    for module_info in pkgutil.iter_modules([str(package_path)]):
        if not module_info.ispkg:
            continue
        module_name = f"{__name__}.{module_info.name}"
        try:
            importlib.import_module(module_name)
        except ImportError:
            # 中文说明：若某个子包依赖缺失导致导入失败，则延迟到实际使用时再抛出异常。
            continue


_auto_import_feature_packages()

__all__ = [
    "FeatureSelectorBase",
    "FeatureSelectorPipeline",
    "build_pipeline",
    "SELECTOR_REGISTRY",
    "register_selector",
    "get_selector_class",
]
