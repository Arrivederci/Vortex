from __future__ import annotations

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
    quantiles: int = 5
    risk_free_rate: float = 0.0
    group_by_columns: Optional[List[str]] = None


class PerformanceAnalyzer:
    def __init__(self, config: PerformanceConfig, output_dir: str = "./artifacts") -> None:
        self.config = config
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, results: pd.DataFrame) -> Dict[str, float]:
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
        ic_values = []
        dates = []
        for dt, group in df.groupby("datetime"):
            if group["prediction"].nunique() <= 1 or group["target_return"].nunique() <= 1:
                continue
            corr = group["prediction"].corr(group["target_return"], method="spearman")
            if pd.notna(corr):
                ic_values.append(corr)
                dates.append(dt)
        return pd.Series(ic_values, index=pd.DatetimeIndex(dates), name="rank_ic")

    def _compute_r2(self, df: pd.DataFrame) -> pd.Series:
        scores = []
        dates = []
        for dt, group in df.groupby("datetime"):
            if group["prediction"].nunique() <= 1:
                continue
            scores.append(r2_score(group["target_return"], group["prediction"]))
            dates.append(dt)
        return pd.Series(scores, index=pd.DatetimeIndex(dates), name="r2")

    def _compute_quantile_returns(self, df: pd.DataFrame) -> Dict[str, Dict[str, pd.Series]]:
        q = self.config.quantiles
        df = df.copy()
        df["quantile"] = df.groupby("datetime")["prediction"].transform(
            lambda x: pd.qcut(x.rank(method="first"), q=q, labels=False) + 1
        )
        quantile_daily = (
            df.groupby(["datetime", "quantile"])["period_return"]
            .mean()
            .unstack("quantile")
            .sort_index()
            .fillna(0.0)
        )
        cumulative = (1 + quantile_daily).cumprod()
        annual_returns = {
            f"quantile_{int(col)}_annual_return": float((1 + quantile_daily[col].mean()) ** 252 - 1)
            for col in quantile_daily.columns
        }
        long_short = quantile_daily[q] - quantile_daily[1]
        annual_returns["long_short_annual_return"] = float((1 + long_short.mean()) ** 252 - 1)
        return {
            "daily": quantile_daily,
            "cumulative": cumulative,
            "annual_returns": annual_returns,
            "long_short_series": long_short,
        }

    def _compute_portfolio_metrics(self, quantile_results: Dict[str, Dict[str, pd.Series]]) -> Dict[str, float]:
        risk_free = self.config.risk_free_rate
        metrics = {}
        quantile_daily = quantile_results["daily"]
        long_short = quantile_results["long_short_series"]
        top_quantile = quantile_daily[quantile_daily.columns[-1]]
        metrics.update(self._portfolio_stats(top_quantile, prefix="top_quantile", risk_free=risk_free))
        metrics.update(self._portfolio_stats(long_short, prefix="long_short", risk_free=risk_free))
        return metrics

    def _portfolio_stats(self, series: pd.Series, prefix: str, risk_free: float) -> Dict[str, float]:
        mean_daily = series.mean()
        vol_daily = series.std(ddof=1)
        annual_return = (1 + mean_daily) ** 252 - 1
        annual_vol = vol_daily * np.sqrt(252)
        sharpe = (mean_daily * 252 - risk_free) / annual_vol if annual_vol else float("nan")
        cum = (1 + series).cumprod()
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
        # Approximate turnover using sign changes in weights (placeholder for MVP)
        changes = series.diff().abs()
        return float(changes.mean())

    def _save_results(self, metrics: Dict[str, float]) -> None:
        path = self.output_dir / "results.json"
        with path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2, ensure_ascii=False)

    def _save_report(self, ic_series: pd.Series, cumulative_quantiles: pd.DataFrame) -> None:
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
