"""特征流水线单元测试，覆盖筛选器与降维器行为。"""

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from vortex.features.selectors import SELECTOR_REGISTRY, build_pipeline


def test_rank_ic_filter_selects_top_features():
    """RankICFilter should select the most informative factor.

    中文说明：验证 RankICFilter 能够保留相关性最强的特征。
    """

    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    assets = ["000001.SZ", "000002.SZ", "000003.SZ"]
    index = pd.MultiIndex.from_product([dates, assets], names=["交易日期", "股票代码"])
    rng = np.random.default_rng(0)
    factor_good = rng.normal(size=len(index))
    target = factor_good + rng.normal(scale=0.1, size=len(index))
    X = pd.DataFrame({
        "good_factor": factor_good,
        "noise_factor": rng.normal(size=len(index)),
    }, index=index)
    y = pd.Series(target, index=index)
    pipeline = build_pipeline(
        [
            {"method": "rank_ic_filter", "params": {"top_k": 1}},
        ],
        SELECTOR_REGISTRY,
    )
    transformed = pipeline.fit_transform(X, y)
    assert list(transformed.columns) == ["good_factor"]


def test_pca_transformer_shapes():
    """PCATransformer should produce expected shape and column names.

    中文说明：确认 PCA 转换器输出的形状与列名符合预期。
    """

    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    assets = ["000001.SZ", "000002.SZ"]
    index = pd.MultiIndex.from_product([dates, assets], names=["交易日期", "股票代码"])
    X = pd.DataFrame({
        "因子1": np.linspace(0, 1, len(index)),
        "因子2": np.linspace(1, 0, len(index)),
    }, index=index)
    y = pd.Series(np.linspace(0, 1, len(index)), index=index)
    pipeline = build_pipeline(
        [
            {"method": "pca", "params": {"n_components": 2}},
        ],
        SELECTOR_REGISTRY,
    )
    transformed = pipeline.fit_transform(X, y)
    assert transformed.shape == (len(index), 2)
    assert all(col.startswith("pca_") for col in transformed.columns)


def test_rank_ic_filter_handles_constant_target():
    """RankICFilter should gracefully handle constant targets.

    中文说明：当目标收益恒定时仍应返回有效的特征集合。
    """

    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    assets = ["000001.SZ", "000002.SZ"]
    index = pd.MultiIndex.from_product([dates, assets], names=["交易日期", "股票代码"])
    X = pd.DataFrame({"因子1": range(len(index)), "因子2": range(len(index))}, index=index)
    y = pd.Series(0.01, index=index, name="target_return")

    pipeline = build_pipeline(
        [
            {"method": "rank_ic_filter", "params": {"top_k": 1}},
        ],
        SELECTOR_REGISTRY,
    )

    transformed = pipeline.fit_transform(X, y)
    assert transformed.shape[1] == 1
    assert set(transformed.columns).issubset(X.columns)
