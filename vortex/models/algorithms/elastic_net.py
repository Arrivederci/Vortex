"""ElasticNet 模型模块，封装 sklearn 实现并注册到系统。"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.linear_model import ElasticNet

from .base import BaseModel
from .registry import register_model


@register_model("elasticnet")
class ElasticNetModel(BaseModel):
    """Wrapper around scikit-learn ElasticNet model.

    中文说明：封装 ElasticNet 回归以符合统一的模型接口。
    """

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        # 中文说明：将传入的参数直接用于构建底层 ElasticNet 模型。
        self.model = ElasticNet(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ElasticNetModel":
        """Fit the ElasticNet model on feature matrix."""

        # 中文说明：在训练前用 0 填充缺失值，确保模型能够顺利拟合。
        self.model.fit(X.fillna(0.0).values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict using the fitted ElasticNet model."""

        # 中文说明：对填充后的数据执行预测，并保持索引一致。
        preds = self.model.predict(X.fillna(0.0).values)
        return pd.Series(preds, index=X.index)


__all__ = ["ElasticNetModel"]
