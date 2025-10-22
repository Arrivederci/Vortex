import json
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from vortex.orchestrator import run_workflow


def create_sample_files(tmp_path: Path):
    dates = pd.date_range("2020-01-01", periods=40, freq="B")
    assets = ["000001.SZ", "000002.SZ", "000003.SZ"]
    records = []
    for date in dates:
        for asset in assets:
            records.append(
                {
                    "交易日期": date,
                    "股票代码": asset,
                    "因子1": np.sin(len(records) / 5),
                    "因子2": np.cos(len(records) / 7),
                }
            )
    factors = pd.DataFrame(records)
    factor_path = tmp_path / "factors.parquet"
    factors.to_parquet(factor_path)

    ohlc_dir = tmp_path / "ohlc"
    ohlc_dir.mkdir()
    for asset in assets:
        asset_df = pd.DataFrame(
            {
                "交易日期": dates,
                "股票代码": asset,
                "收盘价_复权": np.linspace(10, 20, len(dates)) + np.random.default_rng(0).normal(scale=0.5, size=len(dates)),
            }
        )
        asset_df.to_csv(ohlc_dir / f"{asset}.csv", index=False)
    return factor_path, ohlc_dir


def test_run_workflow_end_to_end(tmp_path: Path):
    factor_path, ohlc_dir = create_sample_files(tmp_path)
    config = {
        "data_sources": {
            "factor_path": str(factor_path),
            "ohlc_path": str(ohlc_dir),
            "target_return": {"period": 5, "type": "forward_return"},
        },
        "dataset_manager": {
            "walk_forward": {
                "start_date": "2020-01-01",
                "end_date": "2020-03-31",
                "train_window_type": "fixed",
                "train_length": 15,
                "test_length": 5,
                "step_length": 5,
                "embargo_length": 1,
            }
        },
        "feature_selector": {
            "pipeline": [
                {"method": "rank_ic_filter", "params": {"top_k": 2}},
            ]
        },
        "model": {
            "preprocessor": {"method": "rank", "params": {}},
            "name": "random_forest",
            "params": {"n_estimators": 10, "random_state": 42},
            "persistence": {"enable": True, "save_path": str(tmp_path / "models"), "filename_template": "model_{train_end_date}"},
        },
        "performance_analyzer": {
            "quantiles": 5,
            "risk_free_rate": 0.02,
            "group_by_columns": [],
        },
    }
    config_path = tmp_path / "config.yaml"
    with config_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(config, fp)

    metrics = run_workflow(str(config_path), output_dir=str(tmp_path / "artifacts"))
    assert "ic_mean" in metrics
    results_path = tmp_path / "artifacts" / "results.json"
    report_path = tmp_path / "artifacts" / "report.html"
    assert results_path.exists()
    assert report_path.exists()
    with results_path.open("r", encoding="utf-8") as fp:
        stored = json.load(fp)
    assert stored["quantile_5_annual_return"] == pytest.approx(metrics["quantile_5_annual_return"])
