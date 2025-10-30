"""LightGBM 排序模型单元测试模块。"""

from __future__ import annotations

import pandas as pd
import pytest

from vortex.models.algorithms.registry import get_model_class


_ = pytest.importorskip("lightgbm")


def _build_rank_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Construct a small multi-index dataset with cross-sectional returns."""

    # 中文说明：构造 3 个交易日、3 只股票的截面数据，收益覆盖高/中/低三档。
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    assets = ["AAA", "BBB", "CCC"]
    index = pd.MultiIndex.from_product([dates, assets], names=["交易日期", "股票代码"])
    values = range(len(index))
    X = pd.DataFrame({"factor": values}, index=index)
    # 中文说明：收益依次为高/中/低，便于验证收益分组映射。
    y = pd.Series([0.12, 0.03, -0.05] * len(dates), index=index, dtype="float64")
    return X, y


def test_lightgbm_ranker_trains_with_inferred_groups() -> None:
    """Ensure LightGBM ranker infers group sizes and applies relevance grouping."""

    model_cls = get_model_class("lightgbm_ranker")
    model = model_cls(
        n_estimators=10,
        learning_rate=0.1,
        min_data_in_leaf=1,
        relevance_grouping={"num_groups": 3, "label_gain": [0.0, 1.0, 4.0]},
    )
    X, y = _build_rank_dataset()

    fitted = model.fit(X, y)
    preds = fitted.predict(X)

    assert len(preds) == len(X)
    assert getattr(fitted, "group_sizes_", []) == [3, 3, 3]
    assert getattr(fitted, "label_gain_", []) == [0.0, 1.0, 4.0]
    assert set(fitted.relevance_labels_.unique().tolist()) == {0, 1, 2}


def test_lightgbm_ranker_generates_default_label_gain() -> None:
    """Ensure label_gain defaults to递增序列且高收益获得最高等级。"""

    model_cls = get_model_class("lightgbm_ranker")
    model = model_cls(
        n_estimators=5,
        learning_rate=0.2,
        min_data_in_leaf=1,
        relevance_grouping={"num_groups": 3},
    )
    X, y = _build_rank_dataset()

    fitted = model.fit(X, y)
    preds = fitted.predict(X)

    assert len(preds) == len(X)
    assert getattr(fitted, "label_gain_", []) == [0.0, 1.0, 2.0]

    first_day_labels = fitted.relevance_labels_.xs("2024-01-01", level="交易日期")
    # 中文说明：验证最高收益分组标签为最大值，最低收益为 0。
    assert first_day_labels.loc["AAA"] == 2
    assert first_day_labels.loc["CCC"] == 0
