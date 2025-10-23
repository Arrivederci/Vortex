"""指标评估模块，提供模型调参与验证所需的损失函数。"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def _mse_loss(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute mean squared error loss (lower is better).

    中文说明：计算均方误差，衡量预测值与真实值之间的差异。值越小代表模型越优。
    """

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must contain the same number of observations")
    return float(mean_squared_error(y_true.values, y_pred.values))


def _rank_ic_loss(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute negative average Rank IC to turn maximization into minimization.

    中文说明：先按交易日分组计算秩相关系数，再取平均；为方便调参使用“越小越好”的准则，返回其相反数。
    """

    if not isinstance(y_true.index, pd.MultiIndex) or not isinstance(y_pred.index, pd.MultiIndex):
        raise ValueError("Rank IC loss requires MultiIndex inputs with datetime作为第一级")

    df = pd.DataFrame({"target": y_true, "prediction": y_pred}).dropna()
    if df.empty:
        return float("inf")

    ic_values = []
    for date, group in df.groupby(level=0):
        # 中文说明：若当日预测或真实收益无波动，则跳过避免无效相关系数。
        if group["prediction"].nunique() <= 1 or group["target"].nunique() <= 1:
            continue
        corr = group["prediction"].corr(group["target"], method="spearman")
        if not np.isnan(corr):
            ic_values.append(float(corr))

    if not ic_values:
        return float("inf")
    return -float(np.mean(ic_values))


_METRIC_REGISTRY: Dict[str, Callable[[pd.Series, pd.Series], float]] = {
    "mse": _mse_loss,
    "rank_ic": _rank_ic_loss,
}


def get_loss_function(metric_name: str) -> Callable[[pd.Series, pd.Series], float]:
    """Return loss function associated with the given metric name.

    中文说明：根据配置名称获取对应的损失函数，如找不到则抛出异常提醒调用者。
    """

    key = metric_name.lower()
    if key not in _METRIC_REGISTRY:
        raise ValueError(f"Unknown tuning metric: {metric_name}")
    return _METRIC_REGISTRY[key]


__all__ = ["get_loss_function"]
