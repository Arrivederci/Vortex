"""随机森林模型模块，封装 sklearn 随机森林并提供统一接口。"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .base import BaseModel
from .registry import register_model


@register_model("random_forest")
class RandomForestModel(BaseModel):
    """Random forest regression model wrapper.

    中文说明：封装随机森林回归器并提供统一接口。
    """

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        defaults = {"n_estimators": 200, "random_state": 42}
        # 中文说明：默认参数提供稳定性，可被外部配置覆盖。
        defaults.update(params)
        self.model = RandomForestRegressor(**defaults)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        """Fit the random forest model."""

        # 中文说明：随机森林对缺失值不友好，先填充以确保训练稳定。
        self.model.fit(X.fillna(0.0).values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict using the fitted random forest model."""

        # 中文说明：预测时同样填充缺失值，并保持索引一致。
        preds = self.model.predict(X.fillna(0.0).values)
        return pd.Series(preds, index=X.index)


__all__ = ["RandomForestModel"]
