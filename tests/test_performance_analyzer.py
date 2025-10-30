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

def test_r2_prefers_rank_targets_for_rank_predictions(tmp_path):
    """R² should compare rank-standardized predictions with ranked targets.

    中文说明：当预测结果与目标均为秩标准化时，R² 应与秩目标对齐，
    以免直接与原收益比较导致拟合效果被低估。
    """

    data = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2020-01-01")] * 3,
            "asset_id": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "prediction": [-0.5, 0.0, 0.5],
            "target_return": [-0.5, 0.0, 0.5],
            "period_return": [0.12, -0.03, 0.07],
        }
    )
    analyzer = PerformanceAnalyzer(
        PerformanceConfig(quantiles=3, risk_free_rate=0.0),
        output_dir=str(tmp_path / "artifacts"),
    )

    metrics = analyzer.evaluate(data)

    assert metrics["r2_mean"] == pytest.approx(1.0)


def test_feature_importance_chart_in_report(tmp_path):
    """Feature importance time-series chart should be embedded in the report.

    中文说明：当提供滚动特征重要性数据时，报表需包含可视化曲线。
    """

    dates = [pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-04")]
    data = pd.DataFrame(
        {
            "datetime": dates * 2,
            "asset_id": ["000001.SZ", "000002.SZ"] * 2,
            "prediction": [0.2, 0.1, 0.3, 0.25],
            "target_return": [0.1, -0.1, 0.15, -0.05],
            "period_return": [0.11, -0.12, 0.14, -0.04],
        }
    )
    feature_importances = pd.DataFrame(
        {
            "datetime": [dates[0], dates[1]],
            "feature": ["因子A", "因子A"],
            "importance": [0.15, 0.3],
        }
    )
    output_dir = tmp_path / "artifacts"
    analyzer = PerformanceAnalyzer(
        PerformanceConfig(quantiles=2, risk_free_rate=0.0),
        output_dir=str(output_dir),
    )

    analyzer.evaluate(data, feature_importances=feature_importances)

    report_path = output_dir / "report.html"
    content = report_path.read_text(encoding="utf-8")
    assert "Feature importance over time" in content
