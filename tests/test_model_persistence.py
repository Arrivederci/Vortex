"""测试模型持久化逻辑，确保自定义序列化流程稳定。"""

from __future__ import annotations

import pandas as pd
import pytest

from vortex.models.algorithms.lightgbm import LightGBMModel


@pytest.mark.parametrize("n_estimators", [5])
def test_lightgbm_model_can_be_saved_and_loaded(tmp_path, n_estimators: int) -> None:
    """Ensure LightGBM model persists correctly with custom pickling."""

    pytest.importorskip("lightgbm")

    X = pd.DataFrame({"f1": range(10), "f2": range(10, 20), "f3": range(20, 30)})
    y = pd.Series([float(v) for v in range(10)])

    model = LightGBMModel(n_estimators=n_estimators, random_state=0)
    model.fit(X, y)

    save_path = tmp_path / "lightgbm_model.joblib"
    model.save(save_path)

    loaded_model = LightGBMModel.load(save_path)

    # 中文说明：检查模块属性被重新导入，预测结果与原模型保持一致。
    assert hasattr(loaded_model, "_lgb")
    assert loaded_model._lgb.__name__.startswith("lightgbm")
    pd.testing.assert_series_equal(loaded_model.predict(X), model.predict(X))

