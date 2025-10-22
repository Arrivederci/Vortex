from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler


class BasePreprocessor(ABC):
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BasePreprocessor":
        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)


class IdentityPreprocessor(BasePreprocessor):
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


class RankPreprocessor(BasePreprocessor):
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.groupby(level=0).rank(pct=True)


class StandardScalerPreprocessor(BasePreprocessor):
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.columns: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "StandardScalerPreprocessor":
        self.columns = list(X.columns)
        self.scaler.fit(X.fillna(0.0).values)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
