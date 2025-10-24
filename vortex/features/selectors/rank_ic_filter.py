"""Rank IC 特征筛选器模块，提供基于秩信息系数的特征过滤实现。"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..base import FeatureSelectorBase
from .registry import register_selector


@register_selector("rank_ic_filter")
class RankICFilter(FeatureSelectorBase):
    """Rank IC based feature filter."""

    # 中文说明：按照秩信息系数排名筛选特征，支持设置阈值与数量限制。

    def __init__(self, top_k: Optional[int] = None, min_icir: Optional[float] = None) -> None:
        super().__init__(top_k=top_k, min_icir=min_icir)
        self.top_k = top_k
        self.min_icir = min_icir
        self.selected_columns: List[str] = []
        self.ic_stats: Dict[str, Dict[str, float]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RankICFilter":
        """Fit the filter by computing Rank IC statistics."""

        # 中文说明：计算特征与目标的秩相关系数，并保存筛选后的列。
        combined = X.copy()
        combined["__target"] = y
        ic_records: Dict[str, List[float]] = {col: [] for col in X.columns}
        for _, group in combined.groupby(level=0):
            target = group["__target"]
            if target.nunique() <= 1:
                continue
            for col in X.columns:
                feature = group[col]
                if feature.nunique() <= 1:
                    continue
                corr = feature.corr(target, method="spearman")
                if not math.isnan(corr):
                    ic_records[col].append(corr)
        stats: Dict[str, Dict[str, float]] = {}
        for col, values in ic_records.items():
            if not values:
                continue
            arr = np.asarray(values)
            mean = float(np.nanmean(arr))
            std = float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else float(np.nan)
            icir = mean / std if std and not math.isnan(std) else 0.0
            stats[col] = {"ic_mean": mean, "ic_std": std, "icir": icir}
        if not stats:
            # 中文说明：若所有分组的目标或因子均无有效波动，则退化为保留原始特征。
            stats = {
                col: {"ic_mean": 0.0, "ic_std": float("nan"), "icir": 0.0}
                for col in X.columns
            }
        sorted_columns = sorted(stats, key=lambda c: abs(stats[c]["ic_mean"]), reverse=True)
        if self.min_icir is not None:
            sorted_columns = [c for c in sorted_columns if stats[c]["icir"] >= self.min_icir]
        if self.top_k is not None:
            sorted_columns = sorted_columns[: self.top_k]
        if not sorted_columns:
            # 中文说明：若阈值筛选后为空，则选择前 top_k 或全部特征作为兜底策略。
            sorted_columns = list(X.columns[: self.top_k]) if self.top_k else list(X.columns)
        self.selected_columns = sorted_columns
        self.ic_stats = stats
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_columns:
            # 中文说明：在调用前需确保已经完成拟合步骤。
            raise RuntimeError("RankICFilter must be fitted before calling transform")
        return X.loc[:, self.selected_columns]


__all__ = ["RankICFilter"]
