from __future__ import annotations

from typing import Any, Dict, Type
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor

from .base import BaseModel


class ElasticNetModel(BaseModel):
    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.model = ElasticNet(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ElasticNetModel":
        self.model.fit(X.fillna(0.0).values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = self.model.predict(X.fillna(0.0).values)
        return pd.Series(preds, index=X.index)


class RandomForestModel(BaseModel):
    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        defaults = {"n_estimators": 200, "random_state": 42}
        defaults.update(params)
        self.model = RandomForestRegressor(**defaults)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        self.model.fit(X.fillna(0.0).values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = self.model.predict(X.fillna(0.0).values)
        return pd.Series(preds, index=X.index)


class LightGBMModel(BaseModel):
    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("LightGBM must be installed to use the LightGBM model") from exc
        self._lgb = lgb
        self.model = lgb.LGBMRegressor(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMModel":
        self.model.fit(X.fillna(0.0), y.values)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = self.model.predict(X.fillna(0.0))
        return pd.Series(preds, index=X.index)


class MLPModel(BaseModel):
    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        defaults = {"hidden_layer_sizes": (64, 32), "random_state": 42, "max_iter": 500}
        defaults.update(params)
        self.model = MLPRegressor(**defaults)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MLPModel":
        self.model.fit(X.fillna(0.0).values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
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
