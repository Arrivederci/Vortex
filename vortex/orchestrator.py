"""工作流编排模块，串联配置加载、特征处理、模型训练与评估。"""

from __future__ import annotations

import pathlib
from copy import deepcopy
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from vortex.config.loader import ConfigLoader
from vortex.data.loader import DataLoader, TargetConfig
from vortex.dataset.manager import DatasetManager, WalkForwardConfig
from vortex.features.selectors import SELECTOR_REGISTRY, build_pipeline
from vortex.models.trainer import ModelTrainer
from vortex.metrics.evaluator import get_loss_function
from vortex.performance.analyzer import PerformanceAnalyzer, PerformanceConfig


def _tune_hyperparameters(
    dataset_manager: DatasetManager,
    train_X: pd.DataFrame,
    train_y: pd.Series,
    pipeline_cfg: Sequence[Dict[str, object]],
    preprocessor_cfg: Dict[str, object],
    model_cfg_template: Dict[str, object],
    tuning_cfg: Dict[str, object],
    inner_cv_cfg: Dict[str, object],
) -> Dict[str, object]:
    """Search best hyperparameters using nested purged time-series CV.

    中文说明：在内层时间序列交叉验证中遍历候选参数，
    通过比较配置化的评估指标选择最优模型配置。
    """

    inner_enabled = inner_cv_cfg.get("enable", False)
    tuning_enabled = tuning_cfg.get("enable", False)
    n_splits = inner_cv_cfg.get("n_splits", 0)
    if not (inner_enabled and tuning_enabled and n_splits and n_splits >= 2):
        return deepcopy(model_cfg_template.get("params", {}))

    embargo = inner_cv_cfg.get("embargo_length")
    candidate_params = tuning_cfg.get("candidates", [])
    base_params = deepcopy(model_cfg_template.get("params", {}))
    if not candidate_params:
        return base_params

    metric_name = inner_cv_cfg.get("metric", "mse")
    loss_fn = get_loss_function(metric_name)
    # 中文说明：根据配置获取损失函数，统一以“越小越好”的准则比较候选参数。

    best_score = float("inf")
    best_params = base_params

    try:
        inner_folds = dataset_manager.purged_k_fold(
            train_X,
            n_splits=n_splits,
            embargo_length=embargo,
        )
    except ValueError:
        # 当无法构造有效折时，直接返回基础参数。
        return base_params

    for candidate in candidate_params:
        merged_params = deepcopy(base_params)
        merged_params.update(candidate)
        scores: List[float] = []
        for train_idx, val_idx in inner_folds:
            inner_train_X = train_X.loc[train_idx]
            inner_train_y = train_y.loc[train_idx]
            inner_val_X = train_X.loc[val_idx]
            inner_val_y = train_y.loc[val_idx]

            feature_pipeline = build_pipeline(pipeline_cfg, SELECTOR_REGISTRY)
            cv_train_features = feature_pipeline.fit_transform(inner_train_X, inner_train_y)
            cv_val_features = feature_pipeline.transform(inner_val_X)

            model_cfg = {**model_cfg_template, "params": merged_params}
            trainer = ModelTrainer(preprocessor_cfg, model_cfg)
            trainer.fit(cv_train_features, inner_train_y)
            preds = trainer.predict(cv_val_features)
            score = loss_fn(inner_val_y, preds)
            scores.append(score)

        if scores:
            avg_score = float(np.mean(scores))
            if avg_score < best_score:
                best_score = avg_score
                best_params = merged_params

    return best_params


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
    model_section = config["model"]
    preprocessor_cfg = model_section["preprocessor"]
    model_cfg_template = {k: v for k, v in model_section.items() if k not in {"preprocessor", "tuning"}}
    tuning_cfg = model_section.get("tuning", {})
    inner_cv_cfg = config["dataset_manager"].get("inner_cv", {})

    evaluation_rows: List[pd.DataFrame] = []
    for train_X, train_y, test_X, test_y, info in dataset_manager.generate_splits(data, feature_columns=[col for col in data.columns if col.startswith("factor_")]):
        best_params = _tune_hyperparameters(
            dataset_manager,
            train_X,
            train_y,
            pipeline_cfg,
            preprocessor_cfg,
            model_cfg_template,
            tuning_cfg,
            inner_cv_cfg,
        )

        feature_pipeline = build_pipeline(pipeline_cfg, SELECTOR_REGISTRY)
        train_features = feature_pipeline.fit_transform(train_X, train_y)
        test_features = feature_pipeline.transform(test_X)

        model_cfg = {**model_cfg_template, "params": best_params}
        model_trainer = ModelTrainer(preprocessor_cfg, model_cfg)
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
