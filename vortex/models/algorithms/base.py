from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod
from typing import Any, Type

import joblib
import pandas as pd


class BaseModel(ABC):
    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def save(self, path: str | pathlib.Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls: Type["BaseModel"], path: str | pathlib.Path) -> "BaseModel":
        return joblib.load(path)
