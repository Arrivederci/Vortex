"""数据加载模块，负责组合因子与行情数据并构建训练数据集。"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd


@dataclass
class TargetConfig:
    """Target configuration definition.

    中文说明：描述目标收益的滚动周期与计算方式。
    """

    period: int
    type: str = "forward_return"


class DataLoader:
    """Load factor and market data and assemble a unified DataFrame.

    中文说明：读取因子与行情数据，整合并生成多层索引的特征数据集。
    """

    def __init__(
        self,
        factor_path: str | pathlib.Path,
        ohlc_path: str | pathlib.Path,
        target_config: TargetConfig,
        date_column: str = "交易日期",
        asset_column: str = "股票代码",
        close_column: str = "收盘价_复权",
    ) -> None:
        """Initialize the loader with paths and configuration.

        中文说明：将输入路径转换为 ``Path`` 对象并保存目标配置与列名。
        """
        self.factor_path = pathlib.Path(factor_path)
        self.ohlc_path = pathlib.Path(ohlc_path)
        self.target_config = target_config
        self.date_column = date_column
        self.asset_column = asset_column
        self.close_column = close_column

    def load(self) -> pd.DataFrame:
        """Load, merge and post-process factor and market data.

        中文说明：载入因子与行情数据，执行合并、排序与目标计算。
        """
        factors = self._load_factors()
        market = self._load_market_data()
        merged = factors.merge(
            market,
            on=[self.date_column, self.asset_column],
            how="inner",
        )
        merged.sort_values([self.date_column, self.asset_column], inplace=True)
        merged = self._compute_targets(merged)
        merged.sort_values([self.date_column, self.asset_column], inplace=True)
        merged.set_index([self.date_column, self.asset_column], inplace=True)
        merged.index = pd.MultiIndex.from_arrays(
            [
                pd.to_datetime(merged.index.get_level_values(0)),
                merged.index.get_level_values(1),
            ],
            names=[self.date_column, self.asset_column],
        )
        merged.sort_index(level=[0, 1], inplace=True)
        return merged

    def trading_calendar(self, data: Optional[pd.DataFrame] = None) -> Iterable[pd.Timestamp]:
        if data is None:
            # 如果未提供数据，则从因子文件中推导交易日历。
            data = self._load_factors()
        dates = pd.to_datetime(data[self.date_column].unique())
        return sorted(dates)

    def _load_factors(self) -> pd.DataFrame:
        """Read factor parquet file and normalize column types.

        中文说明：从 Parquet 文件载入因子数据并格式化日期和代码类型。
        """
        factors = pd.read_parquet(self.factor_path)
        factors[self.date_column] = pd.to_datetime(factors[self.date_column])
        factors[self.asset_column] = factors[self.asset_column].astype(str)
        return factors

    def _load_market_data(self) -> pd.DataFrame:
        """Read OHLC CSV files and concatenate them into a DataFrame.

        中文说明：遍历行情 CSV 文件并合并为统一数据框。
        """
        frames = []
        for csv_path in sorted(self.ohlc_path.glob("*.csv")):
            df = pd.read_csv(csv_path)
            frames.append(df)
        if not frames:
            # 如果没有读取到任何 CSV 文件，主动抛出异常。
            raise FileNotFoundError(f"No CSV files found in {self.ohlc_path}")
        market = pd.concat(frames, ignore_index=True)
        market[self.date_column] = pd.to_datetime(market[self.date_column])
        market[self.asset_column] = market[self.asset_column].astype(str)
        return market

    def _compute_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute forward return targets based on the configuration.

        中文说明：按照配置指定的周期计算前瞻收益率目标。
        """
        if self.target_config.type != "forward_return":
            raise ValueError(f"Unsupported target type: {self.target_config.type}")
        period = self.target_config.period
        df = df.copy()
        df.sort_values([self.asset_column, self.date_column], inplace=True)
        df["period_return"] = (
            df.groupby(self.asset_column)[self.close_column]
            .pct_change(period)
            .shift(-period)
        )
        df["target_return"] = df["period_return"]
        return df
