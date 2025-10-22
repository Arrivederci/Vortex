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
