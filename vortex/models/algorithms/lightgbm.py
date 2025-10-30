"""LightGBM 模型模块，封装 LightGBM 回归器并完成注册。"""

from __future__ import annotations

from typing import Any, List

import pandas as pd

from .base import BaseModel
from .registry import register_model


@register_model("lightgbm")
class LightGBMModel(BaseModel):
    """LightGBM regression model wrapper.

    中文说明：封装 LightGBM 模型，提供统一的训练与预测接口。
    """

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("LightGBM must be installed to use the LightGBM model") from exc
        self._lgb = lgb
        # 中文说明：保留外部传参，方便用户在配置中精细调整模型。
        self.model = lgb.LGBMRegressor(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMModel":
        """Fit the LightGBM model with missing value handling."""

        # 中文说明：让 LightGBM 自行处理缺失值，保证特征完整性。
        self.model.fit(X.fillna(0.0), y.values)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict using the fitted LightGBM model."""

        # 中文说明：预测阶段保持与训练一致的缺失值处理策略。
        preds = self.model.predict(X.fillna(0.0))
        return pd.Series(preds, index=X.index)


@register_model("lightgbm_ranker")
class LightGBMRankerModel(BaseModel):
    """LightGBM learning-to-rank model wrapper.

    中文说明：封装 LightGBM 排序模型，自动根据时间截面推断分组并完成训练。
    """

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("LightGBM must be installed to use the LightGBM ranker") from exc
        self._lgb = lgb

        # 中文说明：提取排序标签分组配置，剩余参数传递给 LightGBMRanker。
        params_copy = dict(params)
        self.relevance_grouping_cfg = params_copy.pop("relevance_grouping", None)
        self.model = lgb.LGBMRanker(**params_copy)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMRankerModel":
        """Fit the LightGBM ranker with automatically inferred group sizes."""

        # 中文说明：排序模型需基于收益生成离散等级标签，训练前过滤缺失值并保持索引对齐。
        X_filled = X.fillna(0.0)
        target = y.copy()
        valid_mask = target.notna()
        if not valid_mask.all():
            X_filled = X_filled.loc[valid_mask]
            target = target.loc[valid_mask]

        if not isinstance(self.relevance_grouping_cfg, dict):
            raise ValueError(
                "LightGBMRanker requires 'relevance_grouping' configuration with 'num_groups'"
            )

        num_groups = int(self.relevance_grouping_cfg.get("num_groups", 0))
        if num_groups <= 0:
            raise ValueError("'num_groups' in relevance_grouping must be a positive integer")

        raw_label_gain: List[float] | None = self.relevance_grouping_cfg.get("label_gain")
        if raw_label_gain is not None and len(raw_label_gain) != num_groups:
            raise ValueError(
                "Length of 'label_gain' must equal 'num_groups' in relevance_grouping"
            )

        # 中文说明：若未显式给出 label_gain，则按照等级递增的重要性自动生成。
        label_gain = (
            [float(gain) for gain in raw_label_gain]
            if raw_label_gain is not None
            else [float(i) for i in range(num_groups)]
        )

        # 中文说明：根据收益的截面分位数生成组标签，保证同组样本获得相同等级。
        ranks = target.rank(method="first", pct=True)
        relevance_labels = ((ranks * num_groups) - 1e-12).astype("int64")
        relevance_labels = relevance_labels.clip(lower=0, upper=num_groups - 1)

        group_sizes = self._infer_group_sizes(X_filled.index)
        if not group_sizes:
            raise ValueError("LGBMRanker requires at least one valid group for training")

        if relevance_labels.size == 0:
            raise ValueError("LGBMRanker received empty target after preprocessing")

        self.group_sizes_: List[int] = group_sizes
        self.num_groups_: int = num_groups
        self.label_gain_: List[float] = label_gain
        self.relevance_grouping_ = {"num_groups": num_groups, "label_gain": label_gain}
        self.relevance_labels_ = relevance_labels

        # 中文说明：将 label_gain 通过 set_params 传递给 LightGBM 排序模型。
        self.model.set_params(label_gain=label_gain)
        self.model.fit(X_filled, relevance_labels.to_numpy(), group=group_sizes)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict relevance scores using the fitted ranker."""

        # 中文说明：预测阶段无需分组信息，仅需保持与训练一致的缺失值处理。
        preds = self.model.predict(X.fillna(0.0))
        return pd.Series(preds, index=X.index)

    @staticmethod
    def _infer_group_sizes(index: pd.Index) -> List[int]:
        """Infer per-timestamp group sizes from a MultiIndex."""

        # 中文说明：默认以索引第一层（交易日期）作为查询，生成 LightGBM 排序所需的 group 数量。
        if isinstance(index, pd.MultiIndex):
            counts = (
                pd.Series(1, index=index)
                .groupby(level=0, sort=False)
                .sum()
                .astype(int)
                .tolist()
            )
            return counts
        raise ValueError(
            "LGBMRanker requires features indexed by MultiIndex with date as the first level"
        )


__all__ = ["LightGBMModel", "LightGBMRankerModel"]
