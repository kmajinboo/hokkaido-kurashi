"""Time-series forecasting models and evaluation."""

from src.forecasting.backtest import (
    compare_to_baseline,
    describe_backtest,
    overall_metrics,
    rolling_backtest,
    rolling_metrics,
)
from src.forecasting.base import Forecaster
from src.forecasting.baseline import (
    NaiveForecaster,
    SeasonalNaiveForecaster,
    evaluate_baselines,
    forecast_baselines,
)
from src.forecasting.intervals import (
    IntervalRule,
    error_quantiles,
    forecast_with_intervals,
    interval_coverage,
    realtime_coverage,
    select_errors,
)
from src.forecasting.registry import available_models, get_forecaster, register_forecaster
from src.forecasting.significance import compare_models, diebold_mariano
from src.forecasting.statistical import (
    ARIMAForecaster,
    DriftForecaster,
    ETSForecaster,
    RecentDriftForecaster,
)

__all__ = [
    "ARIMAForecaster",
    "DriftForecaster",
    "ETSForecaster",
    "Forecaster",
    "IntervalRule",
    "NaiveForecaster",
    "RecentDriftForecaster",
    "SeasonalNaiveForecaster",
    "available_models",
    "compare_models",
    "compare_to_baseline",
    "describe_backtest",
    "diebold_mariano",
    "error_quantiles",
    "forecast_with_intervals",
    "interval_coverage",
    "realtime_coverage",
    "select_errors",
    "evaluate_baselines",
    "forecast_baselines",
    "get_forecaster",
    "overall_metrics",
    "register_forecaster",
    "rolling_backtest",
    "rolling_metrics",
]
