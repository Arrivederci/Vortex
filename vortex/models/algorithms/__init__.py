"""模型算法子包初始化模块，导出通用模型接口与注册表。"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from .registry import MODEL_REGISTRY, get_model_class, register_model


def _auto_import_algorithms() -> None:
    """Import algorithm modules dynamically so decorators register models."""

    # 中文说明：遍历算法子包中的所有模块并导入，确保新增模型自动完成注册。
    package_path = Path(__file__).resolve().parent
    for module_info in pkgutil.iter_modules([str(package_path)]):
        if module_info.ispkg or module_info.name in {"base", "registry"}:
            continue
        module_name = f"{__name__}.{module_info.name}"
        try:
            importlib.import_module(module_name)
        except ImportError:
            # 中文说明：当模型依赖未安装时跳过导入，调用方在实际使用时会获得明确提示。
            continue


_auto_import_algorithms()

__all__ = [
    "MODEL_REGISTRY",
    "get_model_class",
    "register_model",
]
