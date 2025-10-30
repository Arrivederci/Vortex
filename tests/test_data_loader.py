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

    open_prices = pd.Series([10.0, 10.2, 10.4, 10.8, 11.0, 11.5], index=dates)
    close_prices = pd.Series([10.1, 10.3, 10.5, 10.9, 11.2, 11.6], index=dates)

    factors = pd.DataFrame(
        {
            "交易日期": dates,
            "股票代码": assets,
            "测试因子": range(len(dates)),
            "收盘价_复权": close_prices.values,
            "开盘价_复权": open_prices.values,
        }
    )
    factor_path = tmp_path / "factors.parquet"
    factors.to_parquet(factor_path)

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


def test_target_standardization_rank(tmp_path: pathlib.Path) -> None:
    """验证秩标准化能够在截面维度正确转换收益率。"""

    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    price_map = {
        "A": [10.0, 11.0, 12.0, 13.0],
        "B": [10.0, 12.0, 14.0, 16.0],
        "C": [10.0, 10.5, 11.0, 11.5],
    }
    records = []
    for asset, prices in price_map.items():
        for date, price in zip(dates, prices):
            records.append(
                {
                    "交易日期": date,
                    "股票代码": asset,
                    "收盘价_复权": price,
                    "开盘价_复权": price,
                    "测试因子": price,
                }
            )
    factors = pd.DataFrame(records)
    factor_path = tmp_path / "rank_factors.parquet"
    factors.to_parquet(factor_path)

    loader = DataLoader(
        factor_path=factor_path,
        ohlc_path=tmp_path,
        target_config=TargetConfig(
            period=1,
            standardization={"method": "rank", "params": {"center": True}},
        ),
    )
    result = loader.load()

    first_date = dates[0]
    expected_raw = {
        "A": price_map["A"][1] / price_map["A"][0] - 1,
        "B": price_map["B"][1] / price_map["B"][0] - 1,
        "C": price_map["C"][1] / price_map["C"][0] - 1,
    }
    expected_rank = {
        "A": pytest.approx(2 / 3 - 0.5),
        "B": pytest.approx(1.0 - 0.5),
        "C": pytest.approx(1 / 3 - 0.5),
    }
    for asset, raw_return in expected_raw.items():
        assert pytest.approx(raw_return) == result.loc[(first_date, asset), "period_return"]
        assert expected_rank[asset] == result.loc[(first_date, asset), "target_return"]


def test_target_standardization_rank_integer(tmp_path: pathlib.Path) -> None:
    """验证整数标签标准化可以生成排序学习所需的离散目标。"""

    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    price_map = {
        "A": [10.0, 11.0, 12.0],
        "B": [10.0, 12.0, 14.0],
        "C": [10.0, 10.5, 11.0],
    }
    records = []
    for asset, prices in price_map.items():
        for date, price in zip(dates, prices):
            records.append(
                {
                    "交易日期": date,
                    "股票代码": asset,
                    "收盘价_复权": price,
                    "开盘价_复权": price,
                    "测试因子": price,
                }
            )
    factors = pd.DataFrame(records)
    factor_path = tmp_path / "rank_integer.parquet"
    factors.to_parquet(factor_path)

    loader = DataLoader(
        factor_path=factor_path,
        ohlc_path=tmp_path,
        target_config=TargetConfig(
            period=1,
            standardization={
                "method": "rank_integer",
                "params": {"rank_method": "first", "higher_is_better": True},
            },
        ),
    )
    result = loader.load()

    first_date = dates[0]
    assert str(result["target_return"].dtype) == "Int64"
    assert result.loc[(first_date, "B"), "target_return"] == 2
    assert result.loc[(first_date, "A"), "target_return"] == 1
    assert result.loc[(first_date, "C"), "target_return"] == 0
