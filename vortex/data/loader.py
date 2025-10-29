"""数据加载模块，负责组合因子与行情数据并构建训练数据集。"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import pandas as pd


@dataclass
class TargetConfig:
    """Target configuration definition.

    中文说明：描述目标收益的滚动周期、标准化方式，并允许自定义开/平仓价来源。
    """

    period: int
    type: str = "forward_return"
    entry_price_column: Optional[str] = None
    exit_price_column: Optional[str] = None
    entry_shift: int = 0
    exit_shift: Optional[int] = None
    standardization: Optional[Dict[str, object]] = None


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
        df = self._load_factors()

        df.sort_values([self.date_column, self.asset_column], inplace=True)
        df = self._compute_targets(df)

        df.sort_values([self.date_column, self.asset_column], inplace=True)
        df.set_index([self.date_column, self.asset_column], inplace=True)
        df.index = pd.MultiIndex.from_arrays(
            [
                pd.to_datetime(df.index.get_level_values(0)),
                df.index.get_level_values(1),
            ],
            names=[self.date_column, self.asset_column],
        )
        df.sort_index(level=[0, 1], inplace=True)
        return df

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
        try:
            import polars as pl

            factors = pl.read_parquet(self.factor_path).to_pandas()
        except ModuleNotFoundError:
            # 中文说明：若运行环境未安装 Polars，则回退到 Pandas 读取，保证单元测试可执行。
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
        cfg = self.target_config
        entry_column = cfg.entry_price_column or self.close_column
        exit_column = cfg.exit_price_column or self.close_column
        entry_shift = cfg.entry_shift
        if entry_shift < 0:
            raise ValueError("entry_shift must be non-negative for forward returns")
        # 中文说明：若未指定平仓偏移，则默认在持有 ``period`` 日后离场。
        exit_shift = cfg.exit_shift if cfg.exit_shift is not None else entry_shift + period
        if exit_shift < entry_shift:
            raise ValueError(
                "exit_shift must be greater than or equal to entry_shift for forward returns"
            )
        if exit_shift < 0:
            raise ValueError("exit_shift must be non-negative for forward returns")

        required_price_columns = {entry_column, exit_column}
        missing_columns = [col for col in required_price_columns if col not in df.columns]
        if missing_columns:
            market = self._load_market_data()
            merge_columns = [self.date_column, self.asset_column, *missing_columns]
            market_subset = market[merge_columns].drop_duplicates(
                subset=[self.date_column, self.asset_column]
            )
            # 中文说明：仅在缺失价格列时引入行情数据，避免生成带有后缀的重复列。
            df = df.merge(
                market_subset,
                on=[self.date_column, self.asset_column],
                how="left",
            )

        grouped = df.groupby(self.asset_column, group_keys=False)
        # 中文说明：使用向量化偏移在分组内部直接获取未来的开/平仓价，避免显式循环。
        entry_price = grouped[entry_column].shift(-entry_shift)
        exit_price = grouped[exit_column].shift(-exit_shift)
        df["period_return"] = exit_price / entry_price - 1
        df["target_return"] = self._apply_target_standardization(df)
        return df

    def _apply_target_standardization(self, df: pd.DataFrame) -> pd.Series:
        """Apply configured standardization on the computed returns.

        中文说明：依据配置对 ``period_return`` 执行指定的标准化算法，例如秩标准化。
        """

        standardization_cfg = self.target_config.standardization or {}
        if not isinstance(standardization_cfg, dict):
            raise TypeError("standardization configuration must be a dictionary if provided")
        method = standardization_cfg.get("method", "none")
        params = standardization_cfg.get("params", {}) or {}
        series = df["period_return"].astype(float)

        if method == "none" or series.empty:
            return series
        if method == "rank":
            rank_method = str(params.get("rank_method", "average"))
            center = bool(params.get("center", True))
            pct = bool(params.get("pct", True))

            def _rank_transform(group: pd.Series) -> pd.Series:
                ranked = group.rank(method=rank_method, pct=pct)
                if center:
                    if pct:
                        ranked = ranked - 0.5
                    else:
                        n = len(group)
                        ranked = (ranked - (n + 1) / 2) / max(n, 1)
                return ranked

            return (
                df.groupby(self.date_column)["period_return"]
                .transform(_rank_transform)
                .astype(float)
            )
        raise ValueError(f"Unsupported target standardization method: {method}")
