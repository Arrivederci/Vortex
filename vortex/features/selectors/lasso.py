"""Lasso 特征选择器模块，利用 L1 正则化系数稀疏性筛选特征。"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

from ..base import FeatureSelectorBase
from .registry import register_selector


@register_selector("lasso")
class LassoSelector(FeatureSelectorBase):
    """Lasso regression based feature selector."""

    # 中文说明：利用 Lasso 回归系数的稀疏性进行特征选择。

    def __init__(self, alpha: float = 0.001, top_k: Optional[int] = None) -> None:
        super().__init__(alpha=alpha, top_k=top_k)
        self.alpha = alpha
        self.top_k = top_k
        self.model = Lasso(alpha=self.alpha, max_iter=10_000)
        self.scaler = StandardScaler()
        self.selected_columns: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LassoSelector":
        """Fit the Lasso model and rank features by coefficients."""

        # 中文说明：训练 Lasso 模型并根据系数大小筛选重要特征。
        X_filled = X.fillna(0.0)
        X_scaled = self.scaler.fit_transform(X_filled.values)
        self.model.fit(X_scaled, y.values)
        coefs = np.abs(self.model.coef_)
        sorted_idx = np.argsort(coefs)[::-1]
        columns = list(X.columns)
        sorted_columns = [columns[i] for i in sorted_idx if coefs[i] > 0]
        if self.top_k is not None:
            sorted_columns = sorted_columns[: self.top_k]
        self.selected_columns = sorted_columns or columns[: self.top_k] if self.top_k else columns
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_columns:
            # 中文说明：未拟合时无法执行转换。
            raise RuntimeError("LassoSelector must be fitted before calling transform")
        return X.loc[:, self.selected_columns]


__all__ = ["LassoSelector"]
