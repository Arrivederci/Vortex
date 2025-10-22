from __future__ import annotations
"""模型算法注册表模块，集中维护多种预测模型实现。"""

from typing import Any, Dict, Type
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor

from .base import BaseModel


class ElasticNetModel(BaseModel):
    """Wrapper around scikit-learn ElasticNet model.

    中文说明：封装 ElasticNet 回归以符合统一的模型接口。
    """

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.model = ElasticNet(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ElasticNetModel":
        """Fit the ElasticNet model on feature matrix.

        中文说明：在填充缺失值后训练 ElasticNet 模型。
        """
        self.model.fit(X.fillna(0.0).values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict using the fitted ElasticNet model.

        中文说明：使用训练好的模型对输入特征进行预测。
        """
        preds = self.model.predict(X.fillna(0.0).values)
        return pd.Series(preds, index=X.index)


class RandomForestModel(BaseModel):
    """Random forest regression model wrapper.

    中文说明：封装随机森林回归器并提供统一接口。
    """

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        defaults = {"n_estimators": 200, "random_state": 42}
        defaults.update(params)
        self.model = RandomForestRegressor(**defaults)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        """Fit the random forest model.

        中文说明：在处理缺失值后训练随机森林模型。
        """
        self.model.fit(X.fillna(0.0).values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict using the fitted random forest model.

        中文说明：对输入特征执行预测，并保持索引一致。
        """
        preds = self.model.predict(X.fillna(0.0).values)
        return pd.Series(preds, index=X.index)


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
        self.model = lgb.LGBMRegressor(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMModel":
        """Fit the LightGBM model with missing value handling.

        中文说明：直接将缺失值交由 LightGBM 处理并训练模型。
        """
        self.model.fit(X.fillna(0.0), y.values)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict using the fitted LightGBM model.

        中文说明：生成预测结果并保持输入索引。
        """
        preds = self.model.predict(X.fillna(0.0))
        return pd.Series(preds, index=X.index)


class MLPModel(BaseModel):
    """Multi-layer perceptron regression wrapper.

    中文说明：封装神经网络回归器，提供与其他模型一致的接口。
    """

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        defaults = {"hidden_layer_sizes": (64, 32), "random_state": 42, "max_iter": 500}
        defaults.update(params)
        self.model = MLPRegressor(**defaults)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MLPModel":
        """Fit the MLP regressor on the dataset.

        中文说明：对填充后的数据训练多层感知机模型。
        """
        self.model.fit(X.fillna(0.0).values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict using the fitted MLP model.

        中文说明：返回模型预测，并保持索引与输入一致。
        """
        preds = self.model.predict(X.fillna(0.0).values)
        return pd.Series(preds, index=X.index)


MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "elasticnet": ElasticNetModel,
    "random_forest": RandomForestModel,
    "lightgbm": LightGBMModel,
    "mlp": MLPModel,
}

__all__ = [
    "ElasticNetModel",
    "RandomForestModel",
    "LightGBMModel",
    "MLPModel",
    "MODEL_REGISTRY",
]
