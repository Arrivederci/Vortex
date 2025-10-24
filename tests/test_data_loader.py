"""针对数据加载器的单元测试。"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from vortex.data.loader import DataLoader, TargetConfig


def _prepare_sample_data(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, pd.DatetimeIndex, pd.Series, pd.Series]:
    """生成测试所需的示例因子与行情数据。"""

    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    assets = ["1"] * len(dates)

    factors = pd.DataFrame(
        {
            "交易日期": dates,
            "股票代码": assets,
            "测试因子": range(len(dates)),
        }
    )
    factor_path = tmp_path / "factors.parquet"
    factors.to_parquet(factor_path)

    open_prices = pd.Series([10.0, 10.2, 10.4, 10.8, 11.0, 11.5], index=dates)
    close_prices = pd.Series([10.1, 10.3, 10.5, 10.9, 11.2, 11.6], index=dates)

    market = pd.DataFrame(
        {
            "交易日期": dates,
            "股票代码": assets,
            "开盘价_复权": open_prices.values,
            "收盘价_复权": close_prices.values,
        }
    )
    ohlc_dir = tmp_path / "ohlc"
    ohlc_dir.mkdir()
    market.to_csv(ohlc_dir / "market.csv", index=False)

    return factor_path, ohlc_dir, dates, open_prices, close_prices


def test_forward_return_default_close_prices(tmp_path: pathlib.Path) -> None:
    """验证默认的收盘价换仓逻辑仍保持兼容。"""

    factor_path, ohlc_dir, dates, _, close_prices = _prepare_sample_data(tmp_path)

    loader = DataLoader(
        factor_path=factor_path,
        ohlc_path=ohlc_dir,
        target_config=TargetConfig(period=2),
    )
    result = loader.load()

    first_date = dates[0]
    # 中文说明：历史逻辑为持有 period 日，以同一列收盘价换仓，因此应等价于未来第 ``period`` 日收盘收益。
    expected_return = close_prices.iloc[2] / close_prices.iloc[0] - 1
    assert pytest.approx(expected_return) == result.loc[(first_date, "1"), "target_return"]


def test_forward_return_open_to_open(tmp_path: pathlib.Path) -> None:
    """验证自定义开/平仓列与偏移后的收益计算。"""

    factor_path, ohlc_dir, dates, open_prices, _ = _prepare_sample_data(tmp_path)

    loader = DataLoader(
        factor_path=factor_path,
        ohlc_path=ohlc_dir,
        target_config=TargetConfig(
            period=2,
            entry_price_column="开盘价_复权",
            exit_price_column="开盘价_复权",
            entry_shift=1,
        ),
    )
    result = loader.load()

    first_date = dates[0]
    # 中文说明：设定次日开盘买入，持有 period 日后再以次日开盘卖出。
    expected_return = open_prices.iloc[3] / open_prices.iloc[1] - 1
    assert pytest.approx(expected_return) == result.loc[(first_date, "1"), "target_return"]
    # 中文说明：period_return 与 target_return 应保持一致，方便后续复用。
    assert result.loc[(first_date, "1"), "target_return"] == result.loc[
        (first_date, "1"),
        "period_return",
    ]
