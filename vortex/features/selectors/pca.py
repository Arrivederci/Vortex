"""PCA 特征转换模块，提供基于主成分分析的降维能力。"""

from __future__ import annotations

from typing import List

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from ..base import FeatureSelectorBase
from .registry import register_selector


@register_selector("pca")
class PCATransformer(FeatureSelectorBase):
    """Principal Component Analysis based transformer."""

    # 中文说明：通过主成分分析进行降维并生成新的特征列。

    def __init__(self, n_components: int = 10) -> None:
        super().__init__(n_components=n_components)
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.component_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PCATransformer":
        """Fit the PCA transformer on standardized data."""

        # 中文说明：对标准化后的数据拟合 PCA 模型并记录组件名称。
        X_filled = X.fillna(0.0)
        X_scaled = self.scaler.fit_transform(X_filled.values)
        transformed = self.pca.fit_transform(X_scaled)
        self.component_names = [f"pca_{i+1}" for i in range(transformed.shape[1])]
        self._last_transform = transformed
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted PCA model."""

        # 中文说明：使用已拟合的 PCA 将输入特征映射到主成分空间。
        check_is_fitted(self.pca)
        X_filled = X.fillna(0.0)
        X_scaled = self.scaler.transform(X_filled.values)
        transformed = self.pca.transform(X_scaled)
        return pd.DataFrame(
            transformed,
            index=X.index,
            columns=self.component_names,
        )


__all__ = ["PCATransformer"]
