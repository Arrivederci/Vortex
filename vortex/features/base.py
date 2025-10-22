from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Sequence

import pandas as pd


class FeatureSelectorBase(ABC):
    """Abstract base class for feature selection/transform steps."""

    def __init__(self, **params: object) -> None:
        self.params = params

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureSelectorBase":
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)


class FeatureSelectorPipeline:
    """Chain multiple feature selection steps defined via configuration."""

    def __init__(self, steps: Sequence[FeatureSelectorBase]):
        self.steps = list(steps)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        transformed = X
        for step in self.steps:
            transformed = step.fit_transform(transformed, y)
        return transformed

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = X
        for step in self.steps:
            transformed = step.transform(transformed)
        return transformed


def build_pipeline(config: Sequence[Dict[str, object]], registry: Dict[str, type]) -> FeatureSelectorPipeline:
    steps: List[FeatureSelectorBase] = []
    for step_cfg in config:
        method = step_cfg["method"]
        params = step_cfg.get("params", {})
        if method not in registry:
            raise ValueError(f"Unknown feature selector method: {method}")
        cls = registry[method]
        steps.append(cls(**params))
    return FeatureSelectorPipeline(steps)
