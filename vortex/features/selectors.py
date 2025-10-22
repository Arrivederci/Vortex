"""特征选择器实现模块，包含多种特征筛选与降维方法。"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .base import FeatureSelectorBase, build_pipeline


class RankICFilter(FeatureSelectorBase):
    """Rank IC based feature filter.

    中文说明：按照秩信息系数排名筛选特征，支持设置阈值与数量限制。
    """

    def __init__(self, top_k: Optional[int] = None, min_icir: Optional[float] = None) -> None:
        super().__init__(top_k=top_k, min_icir=min_icir)
        self.top_k = top_k
        self.min_icir = min_icir
        self.selected_columns: List[str] = []
        self.ic_stats: Dict[str, Dict[str, float]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RankICFilter":
        """Fit the filter by computing Rank IC statistics.

        中文说明：计算特征与目标的秩相关系数，并保存筛选后的列。
        """
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
            # 在目标值或特征缺乏波动时，Rank IC 会全部为 NaN，此处回退为保留原始列。
            stats = {
                col: {"ic_mean": 0.0, "ic_std": float("nan"), "icir": 0.0}
                for col in X.columns
            }
        sorted_columns = sorted(stats, key=lambda c: abs(stats[c]["ic_mean"]), reverse=True)
        if self.min_icir is not None:
            sorted_columns = [c for c in sorted_columns if stats[c]["icir"] >= self.min_icir]
        if self.top_k is not None:
            sorted_columns = sorted_columns[: self.top_k]
        self.selected_columns = sorted_columns
        self.ic_stats = stats
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_columns:
            # 在调用前需确保已经完成拟合步骤。
            raise RuntimeError("RankICFilter must be fitted before calling transform")
        return X.loc[:, self.selected_columns]


class LassoSelector(FeatureSelectorBase):
    """Lasso regression based feature selector.

    中文说明：利用 Lasso 回归系数的稀疏性进行特征选择。
    """

    def __init__(self, alpha: float = 0.001, top_k: Optional[int] = None) -> None:
        super().__init__(alpha=alpha, top_k=top_k)
        self.alpha = alpha
        self.top_k = top_k
        self.model = Lasso(alpha=self.alpha, max_iter=10_000)
        self.scaler = StandardScaler()
        self.selected_columns: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LassoSelector":
        """Fit the Lasso model and rank features by coefficients.

        中文说明：训练 Lasso 模型并根据系数大小筛选重要特征。
        """
        X_filled = X.fillna(0.0)
        X_scaled = self.scaler.fit_transform(X_filled.values)
        self.model.fit(X_scaled, y.values)
        coefs = np.abs(self.model.coef_)
        sorted_idx = np.argsort(coefs)[::-1]
        columns = list(X.columns)
        sorted_columns = [columns[i] for i in sorted_idx if coefs[i] > 0]
        if self.top_k is not None:
            sorted_columns = sorted_columns[: self.top_k]
        self.selected_columns = sorted_columns or columns[: self.top_k] if self.top_k else columns
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_columns:
            # 未拟合时无法执行转换。
            raise RuntimeError("LassoSelector must be fitted before calling transform")
        return X.loc[:, self.selected_columns]


class TreeBasedImportanceSelector(FeatureSelectorBase):
    """Tree-based feature importance selector.

    中文说明：通过树模型的特征重要性指标挑选特征。
    """

    def __init__(
        self,
        model_name: str = "random_forest",
        top_k: Optional[int] = None,
        importance_calculator: str = "native",
        random_state: Optional[int] = 42,
    ) -> None:
        super().__init__(
            model_name=model_name,
            top_k=top_k,
            importance_calculator=importance_calculator,
            random_state=random_state,
        )
        self.model_name = model_name
        self.top_k = top_k
        self.importance_calculator = importance_calculator
        self.random_state = random_state
        self.model = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.selected_columns: List[str] = []

    def _build_model(self, n_features: int):
        """Construct the underlying tree model according to configuration.

        中文说明：根据配置创建树模型实例，可选随机森林或 LightGBM。
        """
        if self.model_name == "random_forest":
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(n_estimators=200, random_state=self.random_state)
        if self.model_name == "lightgbm":
            try:
                import lightgbm as lgb
            except ImportError as exc:  # pragma: no cover - depends on optional package
                raise ImportError(
                    "LightGBM is required for model_name='lightgbm'. Install lightgbm to use this option."
                ) from exc

            return lgb.LGBMRegressor(random_state=self.random_state)
        raise ValueError(f"Unsupported model_name: {self.model_name}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TreeBasedImportanceSelector":
        X_filled = X.fillna(0.0)
        self.model = self._build_model(X_filled.shape[1])
        self.model.fit(X_filled.values, y.values)
        if hasattr(self.model, "feature_importances_"):
            importances = np.asarray(self.model.feature_importances_)
        else:
            raise AttributeError("Model does not provide feature importances")
        sorted_idx = np.argsort(importances)[::-1]
        columns = list(X.columns)
        sorted_columns = [columns[i] for i in sorted_idx if importances[i] > 0]
        if self.top_k is not None:
            sorted_columns = sorted_columns[: self.top_k]
        self.selected_columns = sorted_columns or columns[: self.top_k] if self.top_k else columns
        self.feature_importances_ = importances
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_columns:
            # 需要先拟合模型以获取重要性排序。
            raise RuntimeError("TreeBasedImportanceSelector must be fitted before calling transform")
        return X.loc[:, self.selected_columns]


class PCATransformer(FeatureSelectorBase):
    """Principal Component Analysis based transformer.

    中文说明：通过主成分分析进行降维并生成新的特征列。
    """

    def __init__(self, n_components: int = 10) -> None:
        super().__init__(n_components=n_components)
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.component_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PCATransformer":
        """Fit the PCA transformer on standardized data.

        中文说明：对标准化后的数据拟合 PCA 模型并记录组件名称。
        """
        X_filled = X.fillna(0.0)
        X_scaled = self.scaler.fit_transform(X_filled.values)
        transformed = self.pca.fit_transform(X_scaled)
        self.component_names = [f"pca_{i+1}" for i in range(transformed.shape[1])]
        self._last_transform = transformed
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted PCA model.

        中文说明：使用已拟合的 PCA 将输入特征映射到主成分空间。
        """
        check_is_fitted(self.pca)
        X_filled = X.fillna(0.0)
        X_scaled = self.scaler.transform(X_filled.values)
        transformed = self.pca.transform(X_scaled)
        return pd.DataFrame(
            transformed,
            index=X.index,
            columns=self.component_names,
        )


SELECTOR_REGISTRY: Dict[str, type] = {
    "rank_ic_filter": RankICFilter,
    "lasso": LassoSelector,
    "tree_based_importance": TreeBasedImportanceSelector,
    "pca": PCATransformer,
}

__all__ = [
    "RankICFilter",
    "LassoSelector",
    "TreeBasedImportanceSelector",
    "PCATransformer",
    "SELECTOR_REGISTRY",
    "build_pipeline",
]
