"""工作流编排模块，串联配置加载、特征处理、模型训练与评估。"""

from __future__ import annotations

import pathlib
from typing import Dict, List

import pandas as pd

from vortex.config.loader import ConfigLoader
from vortex.data.loader import DataLoader, TargetConfig
from vortex.dataset.manager import DatasetManager, WalkForwardConfig
from vortex.features.selectors import SELECTOR_REGISTRY, build_pipeline
from vortex.models.trainer import ModelTrainer
from vortex.performance.analyzer import PerformanceAnalyzer, PerformanceConfig


def run_workflow(config_path: str, output_dir: str = "./artifacts") -> Dict[str, float]:
    """Execute the full machine learning workflow described by configuration.

    中文说明：根据配置文件依次完成数据加载、特征处理、模型训练及绩效评估。
    """

    config = ConfigLoader(config_path).load()

    target_cfg = TargetConfig(**config["data_sources"]["target_return"])
    loader = DataLoader(
        factor_path=config["data_sources"]["factor_path"],
        ohlc_path=config["data_sources"]["ohlc_path"],
        target_config=target_cfg,
    )
    data = loader.load()
    trading_calendar = loader.trading_calendar()

    wf_cfg = WalkForwardConfig(**config["dataset_manager"]["walk_forward"])
    dataset_manager = DatasetManager(wf_cfg, trading_calendar)

    pipeline_cfg = config["feature_selector"]["pipeline"]
    feature_pipeline = build_pipeline(pipeline_cfg, SELECTOR_REGISTRY)

    model_trainer = ModelTrainer(config["model"]["preprocessor"], config["model"])

    evaluation_rows: List[pd.DataFrame] = []
    for train_X, train_y, test_X, test_y, info in dataset_manager.generate_splits(data):
        # 针对每个滚动窗口拟合特征流水线与模型。
        train_features = feature_pipeline.fit_transform(train_X, train_y)
        test_features = feature_pipeline.transform(test_X)
        model_trainer.fit(train_features, train_y)
        predictions = model_trainer.predict(test_features)

        evaluation = pd.DataFrame(
            {
                "datetime": test_features.index.get_level_values(0),
                "asset_id": test_features.index.get_level_values(1),
                "prediction": predictions.values,
                "target_return": test_y.values,
                "period_return": data.loc[test_features.index, "period_return"].values,
            }
        )
        evaluation_rows.append(evaluation)
        model_trainer.save_model(info["train_end"])

    evaluation_df = pd.concat(evaluation_rows, ignore_index=True)
    performance_cfg = PerformanceConfig(**config["performance_analyzer"])
    analyzer = PerformanceAnalyzer(performance_cfg, output_dir=output_dir)
    return analyzer.evaluate(evaluation_df)
