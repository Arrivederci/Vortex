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
        self.model = lgb.LGBMRanker(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMRankerModel":
        """Fit the LightGBM ranker with automatically inferred group sizes."""

        # 中文说明：排序模型需使用整数标签，训练前过滤缺失值并保持索引对齐。
        X_filled = X.fillna(0.0)
        target = y.copy()
        valid_mask = target.notna()
        if not valid_mask.all():
            X_filled = X_filled.loc[valid_mask]
            target = target.loc[valid_mask]
        group_sizes = self._infer_group_sizes(X_filled.index)
        if not group_sizes:
            raise ValueError("LGBMRanker requires at least one valid group for training")
        target_int = target.astype("int64", copy=False)
        self.group_sizes_: List[int] = group_sizes
        self.model.fit(X_filled, target_int.to_numpy(), group=group_sizes)
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
