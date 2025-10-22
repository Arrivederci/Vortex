"""模型预处理器模块，定义数据预处理基类与常用实现。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler


class BasePreprocessor(ABC):
    """Abstract base class for preprocessing steps.

    中文说明：定义预处理器的通用接口，包含拟合与转换方法。
    """

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BasePreprocessor":
        """Fit the preprocessor to the dataset.

        中文说明：默认实现直接返回自身，子类可根据需要重写。
        """

        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the dataset after fitting.

        中文说明：子类需实现具体的特征变换逻辑。
        """

        raise NotImplementedError

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Convenience method that fits and transforms data.

        中文说明：组合拟合与转换步骤，减少调用方样板代码。
        """

        self.fit(X, y)
        return self.transform(X)


class IdentityPreprocessor(BasePreprocessor):
    """Pass-through preprocessor that returns data unchanged.

    中文说明：直接返回原始数据，不做任何处理。
    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


class RankPreprocessor(BasePreprocessor):
    """Rank features within each timestamp to percentile scores.

    中文说明：按时间截面对特征进行百分位排名。
    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.groupby(level=0).rank(pct=True)


class StandardScalerPreprocessor(BasePreprocessor):
    """Standardize features using scikit-learn's StandardScaler.

    中文说明：利用标准化方法将特征缩放为零均值单位方差。
    """

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.columns: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "StandardScalerPreprocessor":
        """Fit the scaler and record column order.

        中文说明：记录列名并在缺失值填充后拟合标准化器。
        """

        self.columns = list(X.columns)
        self.scaler.fit(X.fillna(0.0).values)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using the fitted scaler.

        中文说明：对输入特征执行标准化并返回带列名的数据框。
        """

        transformed = self.scaler.transform(X.fillna(0.0).values)
        return pd.DataFrame(transformed, index=X.index, columns=self.columns)


PREPROCESSOR_REGISTRY: Dict[str, type] = {
    "none": IdentityPreprocessor,
    "rank": RankPreprocessor,
    "standard_scaler": StandardScalerPreprocessor,
}

__all__ = [
    "BasePreprocessor",
    "IdentityPreprocessor",
    "RankPreprocessor",
    "StandardScalerPreprocessor",
    "PREPROCESSOR_REGISTRY",
]
