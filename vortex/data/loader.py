from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd


@dataclass
class TargetConfig:
    period: int
    type: str = "forward_return"


class DataLoader:
    """Load factor and market data and assemble a unified DataFrame."""

    def __init__(
        self,
        factor_path: str | pathlib.Path,
        ohlc_path: str | pathlib.Path,
        target_config: TargetConfig,
        date_column: str = "交易日期",
        asset_column: str = "股票代码",
        close_column: str = "收盘价_复权",
    ) -> None:
        self.factor_path = pathlib.Path(factor_path)
        self.ohlc_path = pathlib.Path(ohlc_path)
        self.target_config = target_config
        self.date_column = date_column
        self.asset_column = asset_column
        self.close_column = close_column

    def load(self) -> pd.DataFrame:
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
            data = self._load_factors()
        dates = pd.to_datetime(data[self.date_column].unique())
        return sorted(dates)

    def _load_factors(self) -> pd.DataFrame:
        factors = pd.read_parquet(self.factor_path)
        factors[self.date_column] = pd.to_datetime(factors[self.date_column])
        factors[self.asset_column] = factors[self.asset_column].astype(str)
        return factors

    def _load_market_data(self) -> pd.DataFrame:
        frames = []
        for csv_path in sorted(self.ohlc_path.glob("*.csv")):
            df = pd.read_csv(csv_path)
            frames.append(df)
        if not frames:
            raise FileNotFoundError(f"No CSV files found in {self.ohlc_path}")
        market = pd.concat(frames, ignore_index=True)
        market[self.date_column] = pd.to_datetime(market[self.date_column])
        market[self.asset_column] = market[self.asset_column].astype(str)
        return market

    def _compute_targets(self, df: pd.DataFrame) -> pd.DataFrame:
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
