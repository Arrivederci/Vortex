"""指标模块初始化文件，统一导出常用的评估函数。"""

from .evaluator import get_loss_function

__all__ = ["get_loss_function"]
