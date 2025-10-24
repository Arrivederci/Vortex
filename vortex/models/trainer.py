from __future__ import annotations

"""模型训练模块，负责预处理器与模型的组装、训练与持久化。"""

import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from .algorithms.registry import get_model_class
from .preprocessors.base import PREPROCESSOR_REGISTRY, BasePreprocessor


@dataclass
class PersistenceConfig:
    """Configuration for model persistence.

    中文说明：描述模型持久化时是否启用及保存路径与命名模板。
    """

    enable: bool
    save_path: str
    filename_template: str


class ModelTrainer:
    """Orchestrate preprocessing, model training, prediction and persistence.

    中文说明：负责调用预处理器与模型，统一训练、预测与保存流程。
    """

    def __init__(
        self,
        preprocessor_cfg: Dict[str, Any],
        model_cfg: Dict[str, Any],
    ) -> None:
        """Initialize trainer with configuration dictionaries.

        中文说明：根据配置创建预处理器与模型实例，并解析持久化设置。
        """
        method = preprocessor_cfg.get("method", "none")
        params = preprocessor_cfg.get("params", {})
        if method not in PREPROCESSOR_REGISTRY:
            raise ValueError(f"Unknown preprocessor method: {method}")
        self.preprocessor: BasePreprocessor = PREPROCESSOR_REGISTRY[method](**params)

        name = model_cfg.get("name")
        if not name:
            raise ValueError("Model configuration must include a 'name' field")
        try:
            model_cls = get_model_class(name)
        except KeyError as exc:
            raise ValueError(f"Unknown model name: {name}") from exc
        model_params = model_cfg.get("params", {})
        # 中文说明：通过注册表返回的模型类实例化模型，支持装饰器扩展的新模型。
        self.model = model_cls(**model_params)

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
        """Fit the underlying model using processed features.

        中文说明：先执行特征预处理，再训练模型。
        """
        X_processed = self.preprocessor.fit_transform(X, y)
        self.model.fit(X_processed, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions using the trained model.

        中文说明：对输入数据进行预处理后调用模型预测。
        """
        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed)

    def save_model(self, train_end_date: pd.Timestamp) -> Optional[pathlib.Path]:
        """Persist the trained model and preprocessor to disk.

        中文说明：根据配置保存模型与预处理器，返回模型文件路径。
        """
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
        """Load persisted model and preprocessor from disk.

        中文说明：从指定路径载入模型与预处理器对象。
        """
        from joblib import load

        model = load(model_path)
        preprocessor = load(preprocessor_path)
        return model, preprocessor
