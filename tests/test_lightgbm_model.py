"""LightGBM 回归模型单元测试模块。"""

from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest

from vortex.models.algorithms.lightgbm import LightGBMModel, LightGBMRankerModel


_ = pytest.importorskip("lightgbm")


def _build_regression_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """构造带有 MultiIndex 的示例数据集。"""

    # 中文说明：构造 3 个交易日、2 只股票的简化因子矩阵，保证索引满足训练流程要求。
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    assets = ["AAA", "BBB"]
    index = pd.MultiIndex.from_product([dates, assets], names=["交易日期", "股票代码"])
    X = pd.DataFrame({"factor": range(len(index))}, index=index, dtype="float64")
    y = pd.Series(range(len(index)), index=index, dtype="float64")
    return X, y


def test_lightgbm_model_applies_time_decay_sample_weights() -> None:
    """验证 LightGBM 模型按半衰期生成时间衰减样本权重。"""

    half_life = 2.0
    model = LightGBMModel(
        sample_weight_half_life_days=half_life,
        n_estimators=5,
        learning_rate=0.1,
        min_data_in_leaf=1,
    )
    X, y = _build_regression_dataset()

    fitted = model.fit(X, y)
    weights = fitted.sample_weights_

    assert weights is not None
    assert len(weights) == len(X)

    grouped_weights = weights.groupby(level=0).first()
    latest_date = grouped_weights.index.max()
    age_in_days = (latest_date - grouped_weights.index) / pd.Timedelta(days=1)
    expected = pd.Series((0.5) ** (age_in_days / half_life), index=grouped_weights.index)

    pdt.assert_series_equal(grouped_weights, expected, check_names=False, atol=1e-12, rtol=1e-12)


def test_lightgbm_model_uses_uniform_weights_when_not_configured() -> None:
    """验证未配置半衰期时不应用时间权重。"""

    model = LightGBMModel(n_estimators=5, learning_rate=0.1, min_data_in_leaf=1)
    X, y = _build_regression_dataset()

    fitted = model.fit(X, y)

    assert fitted.sample_weights_ is None


def test_lightgbm_ranker_applies_time_decay_sample_weights() -> None:
    """验证 LightGBM 排序模型按半衰期生成时间衰减样本权重。"""

    half_life = 2.0
    model = LightGBMRankerModel(
        sample_weight_half_life_days=half_life,
        n_estimators=5,
        learning_rate=0.1,
        min_data_in_leaf=1,
    )
    X, y = _build_regression_dataset()

    fitted = model.fit(X, y)
    weights = fitted.sample_weights_

    assert weights is not None
    assert len(weights) == len(X)

    grouped_weights = weights.groupby(level=0).first()
    latest_date = grouped_weights.index.max()
    age_in_days = (latest_date - grouped_weights.index) / pd.Timedelta(days=1)
    expected = pd.Series((0.5) ** (age_in_days / half_life), index=grouped_weights.index)

    pdt.assert_series_equal(grouped_weights, expected, check_names=False, atol=1e-12, rtol=1e-12)


def test_lightgbm_ranker_uses_uniform_weights_when_not_configured() -> None:
    """验证排序模型未配置半衰期时不应用时间权重。"""

    model = LightGBMRankerModel(n_estimators=5, learning_rate=0.1, min_data_in_leaf=1)
    X, y = _build_regression_dataset()

    fitted = model.fit(X, y)

    assert fitted.sample_weights_ is None
