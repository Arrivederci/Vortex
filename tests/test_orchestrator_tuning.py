"""Orchestrator nested交叉验证相关测试。"""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from vortex.dataset.manager import DatasetManager, WalkForwardConfig
from vortex.orchestrator import _tune_hyperparameters


def make_time_series(rows: int = 40) -> pd.DataFrame:
    """Construct deterministic multi-index time-series dataset for tests.

    中文说明：生成固定的多层索引时间序列数据，便于验证切分逻辑。
    """

    dates = pd.date_range("2020-01-01", periods=rows, freq="B")
    assets = ["000001.SZ", "000002.SZ"]
    index = pd.MultiIndex.from_product([dates, assets], names=["交易日期", "股票代码"])
    base = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            "因子1": base,
            "target_return": base * 0.01 + 0.001,
            "period_return": np.full(len(index), 0.01),
        },
        index=index,
    )


class DummyTrainer:
    """Simple trainer stub for deterministic tuning tests.

    中文说明：为了测试调参流程，构造一个可控的训练器，预测值由参数直接决定。
    """

    def __init__(self, _preprocessor_cfg, model_cfg):
        params = model_cfg.get("params", {})
        self.alpha = params.get("alpha", 1.0)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:  # noqa: D401 - 简化实现
        """Record训练时所需的索引，便于预测阶段保持一致。"""

        self._train_index = X.index

    def predict(self, X: pd.DataFrame) -> pd.Series:
        base = X.iloc[:, 0].astype(float)
        predictions = base * self.alpha
        return pd.Series(predictions.values, index=X.index)


def test_tune_hyperparameters_prefers_low_error_candidate(monkeypatch):
    """Ensure nested CV selects candidate with lower validation error.

    中文说明：检查内层时间序列交叉验证能够挑选出误差更小的参数组合。
    """

    data = make_time_series()
    calendar = sorted(data.index.get_level_values(0).unique())
    cfg = WalkForwardConfig(
        start_date="2020-01-01",
        end_date="2020-02-28",
        train_window_type="fixed",
        train_length=12,
        test_length=2,
        step_length=2,
        embargo_length=1,
    )
    manager = DatasetManager(cfg, calendar)
    train_X, train_y, *_ = next(manager.generate_splits(data))

    pipeline_cfg: list[dict[str, object]] = []
    preprocessor_cfg = {"method": "none"}
    model_cfg_template = {
        "name": "dummy",
        "params": {"alpha": 0.02},
        "persistence": {"enable": False},
    }
    tuning_cfg = {"enable": True, "candidates": [{"alpha": 0.5}, {"alpha": 0.01}]}
    inner_cv_cfg = {"enable": True, "n_splits": 3, "embargo_length": 1, "metric": "mse"}

    monkeypatch.setattr("vortex.orchestrator.ModelTrainer", DummyTrainer)
    best_params = _tune_hyperparameters(
        manager,
        train_X,
        train_y,
        pipeline_cfg,
        preprocessor_cfg,
        model_cfg_template,
        tuning_cfg,
        inner_cv_cfg,
    )

    assert pytest.approx(best_params["alpha"], rel=1e-3) == 0.01


def test_tune_hyperparameters_uses_rank_ic_metric(monkeypatch):
    """Ensure configured Rank IC metric guides parameter selection correctly.

    中文说明：验证当评估指标设为 Rank IC 时，调参流程能选出秩相关性更高的候选项。
    """

    data = make_time_series()
    calendar = sorted(data.index.get_level_values(0).unique())
    cfg = WalkForwardConfig(
        start_date="2020-01-01",
        end_date="2020-02-28",
        train_window_type="fixed",
        train_length=12,
        test_length=2,
        step_length=2,
        embargo_length=1,
    )
    manager = DatasetManager(cfg, calendar)
    train_X, train_y, *_ = next(manager.generate_splits(data))

    pipeline_cfg: list[dict[str, object]] = []
    preprocessor_cfg = {"method": "none"}
    model_cfg_template = {
        "name": "dummy",
        "params": {"alpha": 1.0},
        "persistence": {"enable": False},
    }
    tuning_cfg = {"enable": True, "candidates": [{"alpha": -1.0}, {"alpha": 1.0}]}
    inner_cv_cfg = {"enable": True, "n_splits": 3, "embargo_length": 1, "metric": "rank_ic"}

    monkeypatch.setattr("vortex.orchestrator.ModelTrainer", DummyTrainer)
    best_params = _tune_hyperparameters(
        manager,
        train_X,
        train_y,
        pipeline_cfg,
        preprocessor_cfg,
        model_cfg_template,
        tuning_cfg,
        inner_cv_cfg,
    )

    assert pytest.approx(best_params["alpha"], rel=1e-3) == 1.0
