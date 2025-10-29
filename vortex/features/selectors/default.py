"""默认特征选择器，实现空操作以保持特征集不变。"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ..base import FeatureSelectorBase
from .registry import register_selector


@register_selector("default")
class DefaultFeatureSelector(FeatureSelectorBase):
    """A no-op feature selector that returns the input unchanged.

    中文说明：默认特征选择器，不进行任何筛选，仅透传输入数据。
    """

    def __init__(self, **params: object) -> None:
        """Initialize selector with optional parameters for compatibility.

        中文说明：保留参数接口以兼容统一配置结构。
        """

        super().__init__(**params)
        self._feature_columns: Optional[pd.Index] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DefaultFeatureSelector":
        """Store original feature columns and return self.

        中文说明：记录输入特征列名称，确保 transform 时保持原顺序。
        """

        self._feature_columns = X.columns
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the input DataFrame unchanged.

        中文说明：直接返回输入数据的副本，不进行任何筛选或变换。
        """

        if self._feature_columns is None:
            raise RuntimeError("DefaultFeatureSelector must be fitted before calling transform")
        # 中文说明：返回浅拷贝以防调用方修改原始数据，保持列顺序一致。
        return X.loc[:, self._feature_columns].copy()


__all__ = ["DefaultFeatureSelector"]