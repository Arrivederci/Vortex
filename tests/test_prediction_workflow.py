"""预测流程单元测试，验证样本外推理结果的正确性。"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from vortex.data.loader import DataLoader, TargetConfig
from vortex.dataset.manager import DatasetManager, WalkForwardConfig
from vortex.orchestrator import run_prediction_workflow, run_workflow


def _create_sample_files(tmp_path: Path) -> tuple[Path, Path]:
    """生成用于预测流程测试的示例因子与行情数据。"""

    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    assets = ["000001.SZ", "000002.SZ"]
    records = []
    for date in dates:
        for asset in assets:
            records.append(
                {
                    "交易日期": date,
                    "股票代码": asset,
                    "因子1": np.sin(len(records) / 3),
                    "因子2": np.cos(len(records) / 4),
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
                "收盘价_复权": np.linspace(10, 15, len(dates))
                + np.random.default_rng(0).normal(scale=0.1, size=len(dates)),
            }
        )
        asset_df.to_csv(ohlc_dir / f"{asset}.csv", index=False)
    return factor_path, ohlc_dir


def test_run_prediction_workflow(tmp_path: Path) -> None:
    """训练后生成样本外预测，并验证输出列与行数。"""

    factor_path, ohlc_dir = _create_sample_files(tmp_path)
    train_config = {
        "data_sources": {
            "factor_path": str(factor_path),
            "ohlc_path": str(ohlc_dir),
            "target_return": {
                "period": 5,
                "type": "forward_return",
                "standardization": {"method": "rank", "params": {"center": True}},
            },
        },
        "dataset_manager": {
            "walk_forward": {
                "start_date": "2020-01-01",
                "end_date": "2020-02-28",
                "train_window_type": "fixed",
                "train_length": 10,
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
            "preprocessor": {"method": "standard_scaler", "params": {}},
            "name": "random_forest",
            "params": {"n_estimators": 5, "random_state": 0},
            "persistence": {
                "enable": True,
                "save_path": str(tmp_path / "models"),
                "filename_template": "model_{train_end_date}",
            },
        },
        "performance_analyzer": {"quantiles": 3, "risk_free_rate": 0.0, "group_by_columns": []},
    }

    train_config_path = tmp_path / "train.yaml"
    with train_config_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(train_config, fp)

    # 中文说明：先运行训练流程以生成模型与预处理器持久化文件。
    run_workflow(str(train_config_path), output_dir=str(tmp_path / "artifacts"))

    predict_config = {
        "data_sources": train_config["data_sources"],
        "dataset_manager": train_config["dataset_manager"],
        "feature_selector": train_config["feature_selector"],
        "prediction": {
            "model_name": "random_forest",
            "artifacts_path": str(tmp_path / "models"),
            "filename_template": "model_{train_end_date}",
            "output_path": str(tmp_path / "predictions.pkl"),
        },
    }

    predict_config_path = tmp_path / "predict.yaml"
    with predict_config_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(predict_config, fp)

    output_path = run_prediction_workflow(str(predict_config_path))
    assert output_path.exists()

    predictions = pd.read_pickle(output_path)
    expected_columns = ["交易日期", "股票代码", "factor_random_forest"]
    assert list(predictions.columns) == expected_columns

    loader = DataLoader(
        factor_path=str(factor_path),
        ohlc_path=str(ohlc_dir),
        target_config=TargetConfig(period=5, type="forward_return"),
    )
    data = loader.load()
    dataset_manager = DatasetManager(
        WalkForwardConfig(**predict_config["dataset_manager"]["walk_forward"]),
        loader.trading_calendar(),
    )

    expected_rows = 0
    for _train_X, _train_y, test_X, _test_y, _info in dataset_manager.generate_splits(
        data, feature_columns=[col for col in data.columns if col.startswith("factor_")]
    ):
        expected_rows += len(test_X)

    assert len(predictions) == expected_rows
    assert predictions["factor_random_forest"].notnull().all()
