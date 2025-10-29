from __future__ import annotations
"""绩效分析模块，负责评估模型预测结果并生成报表。"""

import json
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import r2_score


@dataclass
class PerformanceConfig:
    """Configuration for performance evaluation.

    中文说明：配置绩效分析所需的分位数数量、无风险利率以及持有期长度等参数。
    """

    quantiles: int = 5
    risk_free_rate: float = 0.0
    group_by_columns: Optional[List[str]] = None
    holding_period: int = 1


class PerformanceAnalyzer:
    """Compute various evaluation metrics and artifacts.

    中文说明：根据预测结果计算多种绩效指标，并保存分析报告。
    """

    def __init__(self, config: PerformanceConfig, output_dir: str = "./artifacts") -> None:
        """Initialize analyzer with configuration and output directory.

        中文说明：保存分析配置并确保输出目录存在。
        """

        self.config = config
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, results: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance metrics.

        中文说明：计算 Rank IC、R²、分位组合收益等指标并输出字典。
        """

        results = results.copy()
        results["datetime"] = pd.to_datetime(results["datetime"])
        ic_series = self._compute_rank_ic(results)
        r2_series = self._compute_r2(results)
        quantile_returns = self._compute_quantile_returns(results)
        portfolio_metrics = self._compute_portfolio_metrics(quantile_returns)

        metrics = {
            "ic_mean": float(ic_series.mean()),
            "ic_std": float(ic_series.std(ddof=1)),
            "icir": float(ic_series.mean() / ic_series.std(ddof=1)) if ic_series.std(ddof=1) else float("nan"),
            "r2_mean": float(r2_series.mean()),
            "r2_std": float(r2_series.std(ddof=1)),
        }
        metrics.update(quantile_returns["annual_returns"])
        metrics.update(portfolio_metrics)

        self._save_results(metrics)
        self._save_report(ic_series, quantile_returns["cumulative"])
        return metrics

    def _compute_rank_ic(self, df: pd.DataFrame) -> pd.Series:
        """Calculate daily Rank IC series.

        中文说明：逐日计算预测与原始收益之间的秩相关系数。
        """

        ic_values = []
        dates = []
        for dt, group in df.groupby("datetime"):
            if group["prediction"].nunique() <= 1 or group["period_return"].nunique() <= 1:
                continue
            corr = group["prediction"].corr(group["period_return"], method="spearman")
            if pd.notna(corr):
                ic_values.append(corr)
                dates.append(dt)
        return pd.Series(ic_values, index=pd.DatetimeIndex(dates), name="rank_ic")

    def _compute_r2(self, df: pd.DataFrame) -> pd.Series:
        """Compute daily coefficient of determination.

        中文说明：对每个交易日计算预测与原始收益值的 R² 指标。
        """

        scores = []
        dates = []
        for dt, group in df.groupby("datetime"):
            valid = group[["period_return", "prediction"]].dropna()
            if valid.empty:
                continue
            if valid["prediction"].nunique() <= 1 or valid["period_return"].nunique() <= 1:
                continue
            # 先移除缺失值并确保目标存在波动，避免 R² 计算返回 NaN。
            scores.append(r2_score(valid["period_return"], valid["prediction"]))
            dates.append(dt)
        return pd.Series(scores, index=pd.DatetimeIndex(dates), name="r2")

    def _compute_quantile_returns(self, df: pd.DataFrame) -> Dict[str, Dict[str, pd.Series]]:
        """Compute quantile portfolio returns and statistics.

        中文说明：构建分位数组合并计算分组收益，基于持有期模拟实际开仓节奏。
        """

        q = self.config.quantiles
        holding_period = max(1, int(self.config.holding_period or 1))
        df = df.copy()

        def assign_quantiles(series: pd.Series) -> pd.Series:
            # 根据预测值排序并划分到不同分位数组合。
            group_size = series.size
            q_eff = max(1, min(q, group_size))
            ranked = series.rank(method="first")
            return pd.qcut(ranked, q=q_eff, labels=False) + 1

        df["quantile"] = (
            df.groupby("datetime")["prediction"].transform(assign_quantiles)
        ).astype("Int64")
        quantile_daily = (
            df.groupby(["datetime", "quantile"], dropna=True)["period_return"]
            .mean()
            .unstack("quantile")
        )
        quantile_daily = quantile_daily.reindex(columns=range(1, q + 1))
        quantile_daily = quantile_daily.sort_index().fillna(0.0).astype(float)

        if quantile_daily.empty:
            periodic_returns = quantile_daily.copy()
        else:
            # 中文说明：仅在每个持有期起点开仓一次，期间保持仓位不变。
            rebalance_positions = np.arange(0, len(quantile_daily), holding_period, dtype=int)
            periodic_returns = quantile_daily.iloc[rebalance_positions].copy()

        cumulative = pd.DataFrame(index=quantile_daily.index, columns=quantile_daily.columns, dtype=float)
        if not periodic_returns.empty:
            cumulative_periodic = (1 + periodic_returns).cumprod()
            cumulative = cumulative_periodic.reindex(quantile_daily.index, method="ffill")
        cumulative = cumulative.fillna(1.0)

        periods_per_year = 252 / holding_period if holding_period else 252.0
        annual_returns: Dict[str, float] = {}
        for col in quantile_daily.columns:
            series = periodic_returns[col]
            if series.empty:
                annual_returns[f"quantile_{int(col)}_annual_return"] = float("nan")
            else:
                annual_returns[f"quantile_{int(col)}_annual_return"] = float(
                    (1 + series.mean()) ** periods_per_year - 1
                )

        long_short_periodic = periodic_returns[q] - periodic_returns[1]
        if long_short_periodic.empty:
            annual_returns["long_short_annual_return"] = float("nan")
        else:
            annual_returns["long_short_annual_return"] = float(
                (1 + long_short_periodic.mean()) ** periods_per_year - 1
            )

        return {
            "daily": quantile_daily,
            "periodic": periodic_returns,
            "cumulative": cumulative,
            "annual_returns": annual_returns,
            "long_short_series": long_short_periodic,
        }

    def _compute_portfolio_metrics(self, quantile_results: Dict[str, Dict[str, pd.Series]]) -> Dict[str, float]:
        """Derive portfolio-level statistics from quantile returns.

        中文说明：基于分位数组合与多空组合计算多项绩效指标。
        """

        risk_free = self.config.risk_free_rate
        metrics = {}
        quantile_periodic = quantile_results["periodic"]
        long_short = quantile_results["long_short_series"]
        if quantile_periodic.empty:
            return {
                "top_quantile_annual_return": float("nan"),
                "top_quantile_annual_volatility": float("nan"),
                "top_quantile_sharpe_ratio": float("nan"),
                "top_quantile_max_drawdown": float("nan"),
                "top_quantile_turnover": float("nan"),
                "long_short_annual_return": float("nan"),
                "long_short_annual_volatility": float("nan"),
                "long_short_sharpe_ratio": float("nan"),
                "long_short_max_drawdown": float("nan"),
                "long_short_turnover": float("nan"),
            }
        top_quantile = quantile_periodic[quantile_periodic.columns[-1]]
        metrics.update(
            self._portfolio_stats(
                top_quantile,
                prefix="top_quantile",
                risk_free=risk_free,
                holding_period=self.config.holding_period,
            )
        )
        metrics.update(
            self._portfolio_stats(
                long_short,
                prefix="long_short",
                risk_free=risk_free,
                holding_period=self.config.holding_period,
            )
        )
        return metrics

    def _portfolio_stats(
        self, series: pd.Series, prefix: str, risk_free: float, holding_period: int
    ) -> Dict[str, float]:
        """Compute annualized return, volatility, Sharpe, drawdown, and turnover.

        中文说明：计算组合的年化收益、波动率、夏普比率、最大回撤与换手率。
        """

        holding_period = max(1, int(holding_period or 1))
        valid = series.dropna()
        if valid.empty:
            return {
                f"{prefix}_annual_return": float("nan"),
                f"{prefix}_annual_volatility": float("nan"),
                f"{prefix}_sharpe_ratio": float("nan"),
                f"{prefix}_max_drawdown": float("nan"),
                f"{prefix}_turnover": float("nan"),
            }

        mean_periodic = float(valid.mean())
        vol_periodic = float(valid.std(ddof=1))
        periods_per_year = 252 / holding_period if holding_period else 252.0
        annual_return = (1 + mean_periodic) ** periods_per_year - 1
        annual_vol = vol_periodic * np.sqrt(periods_per_year) if not np.isnan(vol_periodic) else float("nan")
        risk_free_periodic = (1 + risk_free) ** (holding_period / 252) - 1
        if np.isnan(vol_periodic) or vol_periodic == 0.0:
            sharpe = float("nan")
        else:
            sharpe = ((mean_periodic - risk_free_periodic) / vol_periodic) * np.sqrt(periods_per_year)

        cum = (1 + series.fillna(0.0)).cumprod()
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max
        turnover = self._compute_turnover(series)
        return {
            f"{prefix}_annual_return": float(annual_return),
            f"{prefix}_annual_volatility": float(annual_vol),
            f"{prefix}_sharpe_ratio": float(sharpe),
            f"{prefix}_max_drawdown": float(drawdown.min()),
            f"{prefix}_turnover": float(turnover),
        }

    def _compute_turnover(self, series: pd.Series) -> float:
        """Approximate portfolio turnover based on changes in weights.

        中文说明：通过序列差分估算换手率，属于占位实现。
        """

        # Approximate turnover using sign changes in weights (placeholder for MVP)
        changes = series.diff().abs()
        return float(changes.mean())

    def _save_results(self, metrics: Dict[str, float]) -> None:
        """Persist aggregated metrics to JSON file.

        中文说明：将绩效指标以 JSON 形式写入磁盘。
        """

        path = self.output_dir / "results.json"
        with path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2, ensure_ascii=False)

    def _save_report(self, ic_series: pd.Series, cumulative_quantiles: pd.DataFrame) -> None:
        """Generate interactive HTML report for metrics.

        中文说明：使用 Plotly 生成 Rank IC 与分位数组合的可视化报表。
        """

        fig_ic = go.Figure()
        fig_ic.add_trace(go.Scatter(x=ic_series.index, y=ic_series.values, mode="lines", name="Rank IC"))
        fig_ic.update_layout(title="Rank IC over time", xaxis_title="Date", yaxis_title="Rank IC")

        fig_quantiles = go.Figure()
        for col in cumulative_quantiles.columns:
            fig_quantiles.add_trace(
                go.Scatter(x=cumulative_quantiles.index, y=cumulative_quantiles[col], mode="lines", name=f"Q{col}")
            )
        fig_quantiles.update_layout(title="Quantile cumulative returns", xaxis_title="Date", yaxis_title="Cumulative")

        report_path = self.output_dir / "report.html"
        with report_path.open("w", encoding="utf-8") as fp:
            fp.write("<html><head><title>Performance Report</title></head><body>")
            fp.write(fig_ic.to_html(full_html=False, include_plotlyjs="cdn"))
            fp.write(fig_quantiles.to_html(full_html=False, include_plotlyjs=False))
            fp.write("</body></html>")
