from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from .algorithms.registry import MODEL_REGISTRY
from .preprocessors.base import PREPROCESSOR_REGISTRY, BasePreprocessor


@dataclass
class PersistenceConfig:
    enable: bool
    save_path: str
    filename_template: str


class ModelTrainer:
    def __init__(
        self,
        preprocessor_cfg: Dict[str, Any],
        model_cfg: Dict[str, Any],
    ) -> None:
        method = preprocessor_cfg.get("method", "none")
        params = preprocessor_cfg.get("params", {})
        if method not in PREPROCESSOR_REGISTRY:
            raise ValueError(f"Unknown preprocessor method: {method}")
        self.preprocessor: BasePreprocessor = PREPROCESSOR_REGISTRY[method](**params)

        name = model_cfg.get("name")
        if name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model name: {name}")
        model_params = model_cfg.get("params", {})
        self.model = MODEL_REGISTRY[name](**model_params)

        persistence_cfg = model_cfg.get("persistence", {})
        if persistence_cfg.get("enable"):
            self.persistence = PersistenceConfig(
                enable=True,
                save_path=persistence_cfg["save_path"],
                filename_template=persistence_cfg["filename_template"],
            )
        else:
            self.persistence = PersistenceConfig(enable=False, save_path="", filename_template="")
        self.model_name = name

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X_processed = self.preprocessor.fit_transform(X, y)
        self.model.fit(X_processed, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed)

    def save_model(self, train_end_date: pd.Timestamp) -> Optional[pathlib.Path]:
        if not self.persistence.enable:
            return None
        path = pathlib.Path(self.persistence.save_path)
        path.mkdir(parents=True, exist_ok=True)
        filename = self.persistence.filename_template.format(
            model_name=self.model_name,
            train_end_date=train_end_date.strftime("%Y%m%d"),
        )
        full_path = path / f"{filename}.joblib"
        self.model.save(full_path)
        preprocessor_path = path / f"{filename}_preprocessor.joblib"
        from joblib import dump

        dump(self.preprocessor, preprocessor_path)
        return full_path

    @staticmethod
    def load_model(model_path: str | pathlib.Path, preprocessor_path: str | pathlib.Path):
        from joblib import load

        model = load(model_path)
        preprocessor = load(preprocessor_path)
        return model, preprocessor
