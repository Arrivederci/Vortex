"""数据集管理模块，负责创建时间序列模型所需的训练测试切分。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Sequence, Tuple

import pandas as pd

IndexSlice = pd.IndexSlice


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward data splitting.

    中文说明：定义滚动训练的起止日期、窗口类型及禁区长度等参数。
    """

    start_date: str
    end_date: str
    train_window_type: str
    train_length: int
    test_length: int
    step_length: int
    embargo_length: int


class DatasetManager:
    """Create purged walk-forward train/test splits for time-series data.

    中文说明：根据交易日历生成去污染的滚动训练与测试数据集。
    """

    def __init__(
        self,
        config: WalkForwardConfig,
        trading_calendar: Sequence[pd.Timestamp],
        target_column: str = "target_return",
    ) -> None:
        """Initialize the manager with configuration and calendar.

        中文说明：保存配置、标准化交易日历并记录目标列名称。
        """
        self.config = config
        self.trading_calendar = [pd.Timestamp(d) for d in trading_calendar]
        self.target_column = target_column

    def generate_splits(
        self,
        data: pd.DataFrame,
        feature_columns: Optional[Sequence[str]] = None,
    ) -> Generator[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict[str, pd.Timestamp]], None, None]:
        """Yield successive train/test splits according to the configuration.

        中文说明：根据滚动窗口配置依次产出训练集与测试集切分。
        """
        cfg = self.config
        start = pd.Timestamp(cfg.start_date)
        end = pd.Timestamp(cfg.end_date)
        calendar = [d for d in self.trading_calendar if start <= d <= end]
        if not calendar:
            raise ValueError("Trading calendar is empty for the specified window")

        embargo = cfg.embargo_length
        test_length = cfg.test_length
        train_length = cfg.train_length
        step_length = cfg.step_length

        test_start_idx = train_length + embargo
        while test_start_idx + test_length <= len(calendar):
            test_end_idx = test_start_idx + test_length
            if cfg.train_window_type == "expanding":
                train_start_idx = 0
            elif cfg.train_window_type == "fixed":
                train_start_idx = max(0, test_start_idx - embargo - train_length)
            else:
                raise ValueError(f"Unknown train window type: {cfg.train_window_type}")
            train_end_idx = test_start_idx - embargo

            train_dates = calendar[train_start_idx:train_end_idx]
            test_dates = calendar[test_start_idx:test_end_idx]
            if not train_dates or not test_dates:
                break

            train_start_date = train_dates[0]
            train_end_date = train_dates[-1]
            test_start_date = test_dates[0]
            test_end_date = test_dates[-1]

            train_X, train_y = self._slice_dataset(data, train_start_date, train_end_date, feature_columns)
            test_X, test_y = self._slice_dataset(data, test_start_date, test_end_date, feature_columns)

            info = {
                "train_start": train_start_date,
                "train_end": train_end_date,
                "test_start": test_start_date,
                "test_end": test_end_date,
            }
            yield train_X, train_y, test_X, test_y, info

            test_start_idx += step_length

    def purged_k_fold(
        self,
        data: pd.DataFrame,
        n_splits: int,
        embargo_length: Optional[int] = None,
    ) -> List[Tuple[pd.Index, pd.Index]]:
        """Generate purged k-fold cross-validation indices.

        中文说明：在考虑禁区的情况下生成时间序列交叉验证索引。
        """
        embargo = self.config.embargo_length if embargo_length is None else embargo_length
        unique_dates = sorted(data.index.get_level_values(0).unique())
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        fold_sizes = [len(unique_dates) // n_splits] * n_splits
        remainder = len(unique_dates) % n_splits
        for i in range(remainder):
            fold_sizes[i] += 1
        indices: List[Tuple[pd.Index, pd.Index]] = []
        start_idx = 0
        for fold_size in fold_sizes:
            val_start_idx = start_idx
            val_end_idx = start_idx + fold_size
            val_dates = unique_dates[val_start_idx:val_end_idx]
            start_idx = val_end_idx

            embargo_start_idx = max(0, val_start_idx - embargo)
            embargo_end_idx = min(len(unique_dates), val_end_idx + embargo)

            # 训练集需要剔除验证区间及其禁区范围，确保索引完全互斥。
            train_dates = unique_dates[:embargo_start_idx] + unique_dates[embargo_end_idx:]
            val_index = data.index.get_level_values(0).isin(val_dates)
            train_index = data.index.get_level_values(0).isin(train_dates)
            indices.append((data.index[train_index], data.index[val_index]))
        return indices

    def _slice_dataset(
        self,
        data: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        feature_columns: Optional[Sequence[str]],
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Select a subset of the dataset for given date range.

        中文说明：按照日期范围切片数据集并拆分特征与目标。
        """
        idx = IndexSlice[start_date:end_date, :]
        subset = data.loc[idx]
        subset = subset.sort_index()
        features = feature_columns or [
            c for c in subset.columns if c not in {self.target_column, "period_return"}
        ]
        X = subset[features]
        y = subset[self.target_column]
        return X, y
