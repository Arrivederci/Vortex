"""模型算法基类模块，定义统一的训练、预测与持久化接口。"""

from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod
from typing import Any, Type

import joblib
import pandas as pd


class BaseModel(ABC):
    """Abstract base class for predictive models.

    中文说明：为所有模型实现提供统一的入参存储、训练和预测接口。
    """

    def __init__(self, **params: Any) -> None:
        """Store initialization parameters for concrete models.

        中文说明：保存模型初始化参数，供子类构建内部组件时使用。
        """
        self.params = params

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """Train the model using features and targets.

        中文说明：在子类中实现训练逻辑，并返回自身以支持链式调用。
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions for the given features.

        中文说明：在子类中实现预测逻辑，输出与输入索引对齐的序列。
        """
        raise NotImplementedError

    def save(self, path: str | pathlib.Path) -> None:
        """Persist the model instance using joblib.

        中文说明：利用 joblib 将模型对象保存到文件中。
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls: Type["BaseModel"], path: str | pathlib.Path) -> "BaseModel":
        """Load a persisted model instance from disk.

        中文说明：从文件恢复模型对象，返回对应的实例。
        """
        return joblib.load(path)
