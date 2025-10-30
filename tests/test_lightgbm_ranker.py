"""LightGBM 排序模型单元测试模块。"""

from __future__ import annotations

import pandas as pd
import pytest

from vortex.models.algorithms.registry import get_model_class


_ = pytest.importorskip("lightgbm")


def _build_rank_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Construct a small multi-index dataset with integer ranking labels."""

    # 中文说明：构造 3 个交易日、3 只股票的截面数据，标签满足 LGBMRanker 的整数要求。
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    assets = ["AAA", "BBB", "CCC"]
    index = pd.MultiIndex.from_product([dates, assets], names=["交易日期", "股票代码"])
    values = range(len(index))
    X = pd.DataFrame({"factor": values}, index=index)
    # 中文说明：每个交易日按收益排名生成 2/1/0 三个等级。
    y = pd.Series([2, 1, 0] * len(dates), index=index, dtype="Int64")
    return X, y


def test_lightgbm_ranker_trains_with_inferred_groups() -> None:
    """Ensure LightGBM ranker infers group sizes and returns predictions."""

    model_cls = get_model_class("lightgbm_ranker")
    model = model_cls(n_estimators=10, learning_rate=0.1, min_data_in_leaf=1)
    X, y = _build_rank_dataset()

    fitted = model.fit(X, y)
    preds = fitted.predict(X)

    assert len(preds) == len(X)
    assert getattr(fitted, "group_sizes_", []) == [3, 3, 3]
