"""特征选择器包初始化模块，负责自动注册与导出公共接口。"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import List

from ..base import build_pipeline
from .registry import SELECTOR_REGISTRY, get_selector_class, register_selector


def _auto_import_modules() -> List[str]:
    """Import selector modules dynamically so their decorators run."""

    # 中文说明：自动遍历当前目录下的模块，逐个导入以触发装饰器注册。
    package_path = Path(__file__).resolve().parent
    exported: List[str] = []
    for module_info in pkgutil.iter_modules([str(package_path)]):
        if module_info.ispkg or module_info.name in {"registry"}:
            continue
        module_name = f"{__name__}.{module_info.name}"
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            # 中文说明：当依赖缺失导致导入失败时跳过该模块，使用者在调用时会得到明确报错。
            continue
        for attr in getattr(module, "__all__", []):
            globals()[attr] = getattr(module, attr)
            exported.append(attr)
    return exported


_EXPORTED_CLASSES = _auto_import_modules()

__all__ = [
    "SELECTOR_REGISTRY",
    "register_selector",
    "get_selector_class",
    "build_pipeline",
    *_EXPORTED_CLASSES,
]
