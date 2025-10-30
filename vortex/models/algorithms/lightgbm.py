"""LightGBM 模型模块，封装 LightGBM 回归器并完成注册。"""

from __future__ import annotations

from typing import Any, List, Optional

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

        # 中文说明：支持通过参数 ``sample_weight_half_life_days`` 控制样本时间衰减权重，
        # 其余参数原样传递给 LightGBM 回归模型。
        params_copy = dict(params)
        half_life = params_copy.pop("sample_weight_half_life_days", None)
        self.sample_weight_half_life_days: Optional[float] = None
        if half_life is not None:
            try:
                half_life_value = float(half_life)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "sample_weight_half_life_days 必须为正数或不设置"
                ) from exc
            if half_life_value > 0:
                self.sample_weight_half_life_days = half_life_value
        self.sample_weights_: Optional[pd.Series] = None
        # 中文说明：保留外部传参，方便用户在配置中精细调整模型。
        self.model = lgb.LGBMRegressor(**params_copy)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMModel":
        """Fit the LightGBM model with missing value handling."""

        # 中文说明：让 LightGBM 自行处理缺失值，保证特征完整性。
        X_filled = X.fillna(0.0)
        sample_weight = None
        if self.sample_weight_half_life_days is not None and not X_filled.empty:
            weights = self._compute_time_decay_weights(
                X_filled.index, self.sample_weight_half_life_days
            )
            sample_weight = weights.to_numpy()
            self.sample_weights_ = weights
        else:
            self.sample_weights_ = None

        self.model.fit(X_filled, y.values, sample_weight=sample_weight)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict using the fitted LightGBM model."""

        # 中文说明：预测阶段保持与训练一致的缺失值处理策略。
        preds = self.model.predict(X.fillna(0.0))
        return pd.Series(preds, index=X.index)

    @staticmethod
    def _compute_time_decay_weights(
        index: pd.Index, half_life_days: float
    ) -> pd.Series:
        """Compute exponentially decayed sample weights based on timestamps."""

        # 中文说明：以索引中的日期为基准，按照 ``0.5 ** (age / half_life)``
        # 计算样本权重，越靠近最新日期权重越高。
        if half_life_days <= 0:
            return pd.Series(1.0, index=index, dtype="float64")

        if isinstance(index, pd.MultiIndex):
            dates = index.get_level_values(0)
        else:
            dates = index

        try:
            dates = pd.to_datetime(dates)
        except (TypeError, ValueError):
            return pd.Series(1.0, index=index, dtype="float64")

        if dates.empty:
            return pd.Series(dtype="float64")

        latest_date = dates.max()
        if pd.isna(latest_date):
            return pd.Series(1.0, index=index, dtype="float64")

        age_in_days = (latest_date - dates) / pd.Timedelta(days=1)
        weights = (0.5) ** (age_in_days / half_life_days)
        return pd.Series(weights.astype("float64"), index=index)


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
        half_life = params_copy.pop("sample_weight_half_life_days", None)
        self.sample_weight_half_life_days: Optional[float] = None
        if half_life is not None:
            try:
                half_life_value = float(half_life)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "sample_weight_half_life_days 必须为正数或不设置"
                ) from exc
            if half_life_value > 0:
                self.sample_weight_half_life_days = half_life_value
        # 中文说明：保存拟合阶段生成的样本权重，便于调试核查。
        self.sample_weights_: Optional[pd.Series] = None
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

        sample_weight = None
        if self.sample_weight_half_life_days is not None and not X_filled.empty:
            weights = LightGBMModel._compute_time_decay_weights(
                X_filled.index, self.sample_weight_half_life_days
            )
            sample_weight = weights.to_numpy()
            self.sample_weights_ = weights
        else:
            self.sample_weights_ = None

        relevance_labels = target.groupby(level=0, sort=False).rank(method="first")
        group_sizes = self._infer_group_sizes(X_filled.index)

        # 中文说明：若未显式给出 label_gain，则按照等级递增的重要性自动生成。
        label_gain = [float(i) ** 2 for i in range(max(group_sizes) + 1)]

        # 中文说明：将 label_gain 通过 set_params 传递给 LightGBM 排序模型。
        self.model.set_params(label_gain=label_gain)
        self.model.fit(
            X_filled,
            relevance_labels.to_numpy(),
            group=group_sizes,
            sample_weight=sample_weight,
        )
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
