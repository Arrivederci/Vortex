"""树模型特征重要性选择器模块，支持随机森林与 LightGBM。"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ..base import FeatureSelectorBase
from .registry import register_selector


@register_selector("tree_based_importance")
class TreeBasedImportanceSelector(FeatureSelectorBase):
    """Tree-based feature importance selector."""

    # 中文说明：通过树模型的特征重要性指标挑选特征。

    def __init__(
        self,
        model_name: str = "random_forest",
        top_k: Optional[int] = None,
        importance_calculator: str = "native",
        random_state: Optional[int] = 42,
    ) -> None:
        super().__init__(
            model_name=model_name,
            top_k=top_k,
            importance_calculator=importance_calculator,
            random_state=random_state,
        )
        self.model_name = model_name
        self.top_k = top_k
        self.importance_calculator = importance_calculator
        self.random_state = random_state
        self.model = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.selected_columns: List[str] = []

    def _build_model(self, n_features: int):
        """Construct the underlying tree model according to configuration."""

        # 中文说明：根据配置创建树模型实例，可选随机森林或 LightGBM。
        if self.model_name == "random_forest":
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(n_estimators=200, random_state=self.random_state)
        if self.model_name == "lightgbm":
            try:
                import lightgbm as lgb
            except ImportError as exc:  # pragma: no cover - depends on optional package
                raise ImportError(
                    "LightGBM is required for model_name='lightgbm'. Install lightgbm to use this option."
                ) from exc

            return lgb.LGBMRegressor(random_state=self.random_state)
        raise ValueError(f"Unsupported model_name: {self.model_name}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TreeBasedImportanceSelector":
        X_filled = X.fillna(0.0)
        self.model = self._build_model(X_filled.shape[1])
        self.model.fit(X_filled.values, y.values)
        if hasattr(self.model, "feature_importances_"):
            importances = np.asarray(self.model.feature_importances_)
        else:
            raise AttributeError("Model does not provide feature importances")
        sorted_idx = np.argsort(importances)[::-1]
        columns = list(X.columns)
        sorted_columns = [columns[i] for i in sorted_idx if importances[i] > 0]
        if self.top_k is not None:
            sorted_columns = sorted_columns[: self.top_k]
        self.selected_columns = sorted_columns or columns[: self.top_k] if self.top_k else columns
        self.feature_importances_ = importances
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_columns:
            # 中文说明：需要先拟合模型以获取重要性排序。
            raise RuntimeError("TreeBasedImportanceSelector must be fitted before calling transform")
        return X.loc[:, self.selected_columns]


__all__ = ["TreeBasedImportanceSelector"]
