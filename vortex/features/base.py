"""特征选择基础模块，定义通用接口与管线组件。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Sequence

import pandas as pd


class FeatureSelectorBase(ABC):
    """Abstract base class for feature selection/transform steps.

    中文说明：为所有特征选择或转换步骤定义统一接口。
    """

    def __init__(self, **params: object) -> None:
        """Store initialization parameters.

        中文说明：保存构造参数供子类使用。
        """
        self.params = params

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureSelectorBase":
        """Fit the selector using provided data.

        中文说明：基于给定特征与目标拟合选择器，返回自身实例。
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the dataset according to fitted state.

        中文说明：根据先前拟合的状态转换特征数据。
        """
        raise NotImplementedError

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Convenience method combining fit and transform.

        中文说明：依次执行拟合与转换，返回转换后的数据。
        """
        self.fit(X, y)
        return self.transform(X)


class FeatureSelectorPipeline:
    """Chain multiple feature selection steps defined via configuration.

    中文说明：将多个特征处理步骤串联成流水线。
    """

    def __init__(self, steps: Sequence[FeatureSelectorBase]):
        """Initialize pipeline with an ordered list of steps.

        中文说明：按照给定顺序保存特征处理步骤。
        """
        self.steps = list(steps)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform the dataset through each pipeline step.

        中文说明：在流水线中逐步拟合并转换输入特征。
        """
        transformed = X
        for step in self.steps:
            transformed = step.fit_transform(transformed, y)
        return transformed

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the dataset using already-fitted steps.

        中文说明：使用已拟合的步骤依次转换特征数据。
        """
        transformed = X
        for step in self.steps:
            transformed = step.transform(transformed)
        return transformed


def build_pipeline(config: Sequence[Dict[str, object]], registry: Dict[str, type]) -> FeatureSelectorPipeline:
    """Construct a feature selector pipeline from configuration.

    中文说明：根据配置和注册表实例化流水线中的每一个步骤。
    """
    steps: List[FeatureSelectorBase] = []
    for step_cfg in config:
        method = step_cfg["method"]
        params = step_cfg.get("params", {})
        if method not in registry:
            # 当找不到对应的特征选择器时提示调用方。
            raise ValueError(f"Unknown feature selector method: {method}")
        cls = registry[method]
        steps.append(cls(**params))
    return FeatureSelectorPipeline(steps)
