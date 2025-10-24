"""模型注册表测试模块，验证装饰器注册与内置模型可用性。"""

from __future__ import annotations

import pandas as pd

from vortex.models.algorithms.base import BaseModel
from vortex.models.algorithms.registry import MODEL_REGISTRY, get_model_class, register_model


class _DummyDataset:
    """简单的数据包装器，构造测试所需的 DataFrame 与 Series。"""

    def __init__(self) -> None:
        self.features = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
        self.targets = pd.Series([1.0, 2.0, 3.0])


def test_builtin_models_registered() -> None:
    """内置模型应在导入时自动注册。"""

    # 中文说明：无需关心可选依赖 LightGBM，因此只校验基础模型是否存在。
    for name in {"elasticnet", "random_forest", "mlp"}:
        assert name in MODEL_REGISTRY, f"Expected built-in model '{name}' to be registered"


def test_register_model_decorator_supports_custom_models() -> None:
    """装饰器应允许用户注册新的模型并通过名称获取。"""

    @register_model("dummy_test_model")
    class DummyModel(BaseModel):
        """简单的占位模型用于验证注册流程。"""

        def fit(self, X: pd.DataFrame, y: pd.Series) -> DummyModel:
            """直接返回 self，模拟训练流程。"""

            # 中文说明：测试中仅检查接口，无需真实训练逻辑。
            return self

        def predict(self, X: pd.DataFrame) -> pd.Series:
            """返回全零序列以验证调用路径。"""

            return pd.Series([0.0] * len(X), index=X.index)

    try:
        dataset = _DummyDataset()
        model_cls = get_model_class("dummy_test_model")
        model = model_cls()
        model.fit(dataset.features, dataset.targets)
        preds = model.predict(dataset.features)
        assert len(preds) == len(dataset.features)
    finally:
        # 中文说明：测试结束后移除临时模型，避免影响其他用例。
        MODEL_REGISTRY.pop("dummy_test_model", None)
