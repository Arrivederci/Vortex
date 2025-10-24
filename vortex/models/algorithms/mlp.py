"""多层感知机模型模块，封装 sklearn MLP 并注册。"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.neural_network import MLPRegressor

from .base import BaseModel
from .registry import register_model


@register_model("mlp")
class MLPModel(BaseModel):
    """Multi-layer perceptron regression wrapper.

    中文说明：封装神经网络回归器，提供与其他模型一致的接口。
    """

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        defaults = {"hidden_layer_sizes": (64, 32), "random_state": 42, "max_iter": 500}
        # 中文说明：默认结构兼顾精度与训练速度，可由配置覆盖。
        defaults.update(params)
        self.model = MLPRegressor(**defaults)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MLPModel":
        """Fit the MLP regressor on the dataset."""

        # 中文说明：使用填充后的矩阵训练网络，避免缺失值导致计算异常。
        self.model.fit(X.fillna(0.0).values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict using the fitted MLP model."""

        # 中文说明：预测时同样进行缺失值填充，保持与训练流程一致。
        preds = self.model.predict(X.fillna(0.0).values)
        return pd.Series(preds, index=X.index)


__all__ = ["MLPModel"]
