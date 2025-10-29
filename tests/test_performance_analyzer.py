"""性能分析器单元测试，验证指标计算的稳定性。"""

from __future__ import annotations

import math

import pytest

pd = pytest.importorskip("pandas")

from vortex.performance.analyzer import PerformanceAnalyzer, PerformanceConfig


def test_r2_computation_skips_constant_targets(tmp_path):
    """PerformanceAnalyzer should skip R² calculation when targets lack variance.

    中文说明：当目标序列不含波动时，R² 计算应跳过以避免 NaN 引发异常。
    """

    data = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2020-01-01")] * 3,
            "asset_id": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "prediction": [0.1, 0.2, 0.3],
            "target_return": [0.05, 0.05, 0.05],
            "period_return": [0.01, 0.01, 0.01],
        }
    )
    config = PerformanceConfig(quantiles=3, risk_free_rate=0.0)
    analyzer = PerformanceAnalyzer(config, output_dir=str(tmp_path / "artifacts"))

    metrics = analyzer.evaluate(data)

    assert math.isnan(metrics["r2_mean"])
    assert math.isnan(metrics["r2_std"])


def test_portfolio_metrics_respect_holding_period(tmp_path):
    """Annualized returns should respect the configured holding period.

    中文说明：验证绩效分析器根据持有期仅在每个周期起点开仓一次，
    年化收益需按照周期频率换算。
    """

    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    records = []
    for dt in dates:
        records.append(
            {
                "datetime": dt,
                "asset_id": "000001.SZ",
                "prediction": 1.0,
                "target_return": 0.0,
                "period_return": 0.05,
            }
        )
        records.append(
            {
                "datetime": dt,
                "asset_id": "000002.SZ",
                "prediction": 0.0,
                "target_return": 0.0,
                "period_return": 0.01,
            }
        )
    data = pd.DataFrame(records)
    config = PerformanceConfig(quantiles=2, risk_free_rate=0.0, holding_period=5)
    analyzer = PerformanceAnalyzer(config, output_dir=str(tmp_path / "artifacts"))

    metrics = analyzer.evaluate(data)

    periods_per_year = 252 / 5
    expected_top = (1 + 0.05) ** periods_per_year - 1
    expected_long_short = (1 + (0.05 - 0.01)) ** periods_per_year - 1

    assert metrics["quantile_2_annual_return"] == pytest.approx(expected_top)
    assert metrics["top_quantile_annual_return"] == pytest.approx(expected_top)
    assert metrics["long_short_annual_return"] == pytest.approx(expected_long_short)
