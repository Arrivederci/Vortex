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
    feature_importance_frames: List[pd.DataFrame] = []
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
        print(info)
        feature_pipeline = build_pipeline(pipeline_cfg, SELECTOR_REGISTRY)
        train_features = feature_pipeline.fit_transform(train_X, train_y)
        test_features = feature_pipeline.transform(test_X)

        model_cfg = {**model_cfg_template, "params": best_params}
        model_trainer = ModelTrainer(preprocessor_cfg, model_cfg)
        model_trainer.fit(train_features, train_y)
        predictions = model_trainer.predict(test_features)

        importance_series = model_trainer.get_feature_importances()
        if not importance_series.empty:
            # 中文说明：将每次滚动窗口的特征重要性记录下来，供绩效分析生成时序图表。
            feature_importance_frames.append(
                pd.DataFrame(
                    {
                        "datetime": info["test_end"],
                        "feature": importance_series.index,
                        "importance": importance_series.values,
                    }
                )
            )

        evaluation = pd.DataFrame(
            {
                "datetime": test_features.index.get_level_values(0),
                "asset_id": test_features.index.get_level_values(1),
                "prediction": predictions.values,
                "target_return": test_y.loc[test_features.index].values,
                "period_return": data.loc[test_features.index, "period_return"].values,
            }
        )
        evaluation_rows.append(evaluation)
        model_trainer.save_model(info["train_end"])

    evaluation_df = pd.concat(evaluation_rows, ignore_index=True)
    feature_importance_df = (
        pd.concat(feature_importance_frames, ignore_index=True)
        if feature_importance_frames
        else pd.DataFrame(columns=["datetime", "feature", "importance"])
    )
    performance_section = dict(config["performance_analyzer"])
    if not performance_section.get("holding_period"):
        performance_section["holding_period"] = target_cfg.period
    performance_cfg = PerformanceConfig(**performance_section)
    analyzer = PerformanceAnalyzer(performance_cfg, output_dir=output_dir)
    return analyzer.evaluate(evaluation_df, feature_importances=feature_importance_df)


def run_prediction_workflow(config_path: str) -> pathlib.Path:
    """Generate out-of-sample predictions using persisted models.

    中文说明：
        1. 读取预测配置与原始因子数据，构建与训练阶段一致的滚动窗口；
        2. 基于训练阶段保存的模型与预处理器，仅执行特征流水线拟合与模型推理；
        3. 将样本外预测结果整理为 ``[交易日期，股票代码，factor_{模型名}]`` 列并保存为 pkl。
    """

    config = ConfigLoader(config_path).load()

    if "prediction" not in config:
        raise KeyError("配置文件缺少 prediction 段落，无法执行预测流程。")

    prediction_cfg = config["prediction"]
    model_name = prediction_cfg.get("model_name")
    if not model_name:
        raise ValueError("prediction.model_name 为必填字段，用于定位模型与命名预测列。")

    artifacts_path = prediction_cfg.get("artifacts_path")
    if not artifacts_path:
        raise ValueError("prediction.artifacts_path 为必填字段，用于读取训练阶段保存的模型文件。")
    artifacts_dir = pathlib.Path(artifacts_path)

    filename_template = prediction_cfg.get("filename_template")
    if not filename_template:
        raise ValueError("prediction.filename_template 为必填字段，用于拼接模型文件名。")

    output_path_value = prediction_cfg.get("output_path", "./artifacts/predictions.pkl")
    output_path = pathlib.Path(output_path_value)

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

    prediction_frames: List[pd.DataFrame] = []
    evaluation_rows: List[pd.DataFrame] = []
    feature_importance_frames: List[pd.DataFrame] = []
    feature_columns = [col for col in data.columns if col.startswith("factor_")]

    for train_X, train_y, test_X, test_y, info in dataset_manager.generate_splits(
        data,
        feature_columns=feature_columns,
    ):
        feature_pipeline = build_pipeline(pipeline_cfg, SELECTOR_REGISTRY)
        # 中文说明：虽然预测阶段无需训练模型，但仍需拟合特征流水线以与训练阶段保持一致。
        feature_pipeline.fit_transform(train_X, train_y)
        test_features = feature_pipeline.transform(test_X)

        formatted_date = info["train_end"].strftime("%Y%m%d")
        model_filename = filename_template.format(
            model_name=model_name,
            train_end_date=formatted_date,
        )
        model_path = artifacts_dir / f"{model_filename}.joblib"
        preprocessor_path = artifacts_dir / f"{model_filename}_preprocessor.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"找不到模型文件：{model_path}")
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"找不到预处理器文件：{preprocessor_path}")

        model, preprocessor = ModelTrainer.load_model(model_path, preprocessor_path)
        processed_test = preprocessor.transform(test_features)
        predictions = model.predict(processed_test)

        if not isinstance(predictions, pd.Series):
            # 中文说明：部分模型可能返回 ndarray，此处统一转换为带索引的序列，确保后续对齐。
            predictions = pd.Series(predictions, index=processed_test.index)

        prediction_frame = pd.DataFrame(
            {
                loader.date_column: processed_test.index.get_level_values(0),
                loader.asset_column: processed_test.index.get_level_values(1),
                f"factor_{model_name}": predictions.values,
            }
        )
        prediction_frames.append(prediction_frame)

        evaluation_rows.append(
            pd.DataFrame(
                {
                    "datetime": processed_test.index.get_level_values(0),
                    "asset_id": processed_test.index.get_level_values(1),
                    "prediction": predictions.values,
                    "target_return": test_y.loc[processed_test.index].values,
                    "period_return": data.loc[processed_test.index, "period_return"].values,
                }
            )
        )

        importance_series = ModelTrainer.extract_feature_importances(model, processed_test.columns)
        if not importance_series.empty:
            feature_importance_frames.append(
                pd.DataFrame(
                    {
                        "datetime": info["test_end"],
                        "feature": importance_series.index,
                        "importance": importance_series.values,
                    }
                )
            )

    if not prediction_frames:
        raise ValueError("数据切分结果为空，预测流程无法生成任何测试样本。")

    prediction_df = pd.concat(prediction_frames, ignore_index=True)
    prediction_df.sort_values([loader.date_column, loader.asset_column], inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_pickle(output_path)

    if evaluation_rows:
        evaluation_df = pd.concat(evaluation_rows, ignore_index=True)
        feature_importance_df = (
            pd.concat(feature_importance_frames, ignore_index=True)
            if feature_importance_frames
            else pd.DataFrame(columns=["datetime", "feature", "importance"])
        )

        performance_section = dict(config.get("performance_analyzer", {}))
        if not performance_section.get("holding_period"):
            performance_section["holding_period"] = target_cfg.period
        performance_cfg = PerformanceConfig(**performance_section)

        performance_dir_value = prediction_cfg.get("performance_output_path")
        if performance_dir_value:
            performance_dir = pathlib.Path(performance_dir_value)
        else:
            performance_dir = output_path.parent / "performance"
        performance_dir.mkdir(parents=True, exist_ok=True)

        analyzer = PerformanceAnalyzer(performance_cfg, output_dir=str(performance_dir))
        analyzer.evaluate(evaluation_df, feature_importances=feature_importance_df)

    return output_path
