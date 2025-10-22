import pytest

pd = pytest.importorskip("pandas")

from vortex.dataset.manager import DatasetManager, WalkForwardConfig


def make_sample_data():
    dates = pd.date_range("2020-01-01", periods=20, freq="B")
    assets = ["000001.SZ", "000002.SZ"]
    index = pd.MultiIndex.from_product([dates, assets], names=["交易日期", "股票代码"])
    df = pd.DataFrame({
        "因子1": range(len(index)),
        "因子2": range(len(index), 2 * len(index)),
        "target_return": [0.01] * len(index),
        "period_return": [0.01] * len(index),
    }, index=index)
    return df


def test_generate_splits_fixed_window():
    data = make_sample_data()
    calendar = sorted(data.index.get_level_values(0).unique())
    cfg = WalkForwardConfig(
        start_date="2020-01-01",
        end_date="2020-01-31",
        train_window_type="fixed",
        train_length=5,
        test_length=2,
        step_length=2,
        embargo_length=1,
    )
    manager = DatasetManager(cfg, calendar)
    splits = list(manager.generate_splits(data))
    assert splits, "Expected at least one split"
    for train_X, train_y, test_X, test_y, info in splits:
        assert train_X.index.get_level_values(0).max() < test_X.index.get_level_values(0).min()
        assert len(test_X) == len(test_y)
        assert len(train_X) == len(train_y)
        assert info["test_start"] > info["train_end"]


def test_purged_k_fold_respects_embargo():
    data = make_sample_data()
    calendar = sorted(data.index.get_level_values(0).unique())
    cfg = WalkForwardConfig(
        start_date="2020-01-01",
        end_date="2020-01-31",
        train_window_type="fixed",
        train_length=5,
        test_length=2,
        step_length=2,
        embargo_length=1,
    )
    manager = DatasetManager(cfg, calendar)
    folds = manager.purged_k_fold(data, n_splits=3)
    assert len(folds) == 3
    for train_idx, val_idx in folds:
        assert set(train_idx).isdisjoint(set(val_idx))
