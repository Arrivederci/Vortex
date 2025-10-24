"""LightGBM 模型模块，封装 LightGBM 回归器并完成注册。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseModel
from .registry import register_model


@register_model("lightgbm")
class LightGBMModel(BaseModel):
    """LightGBM regression model wrapper.

    中文说明：封装 LightGBM 模型，提供统一的训练与预测接口。
    """

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("LightGBM must be installed to use the LightGBM model") from exc
        self._lgb = lgb
        # 中文说明：保留外部传参，方便用户在配置中精细调整模型。
        self.model = lgb.LGBMRegressor(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMModel":
        """Fit the LightGBM model with missing value handling."""

        # 中文说明：让 LightGBM 自行处理缺失值，保证特征完整性。
        self.model.fit(X.fillna(0.0), y.values)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict using the fitted LightGBM model."""

        # 中文说明：预测阶段保持与训练一致的缺失值处理策略。
        preds = self.model.predict(X.fillna(0.0))
        return pd.Series(preds, index=X.index)


__all__ = ["LightGBMModel"]
