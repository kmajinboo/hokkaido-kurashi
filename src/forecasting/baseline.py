"""Naive and seasonal-naive baselines for time-series forecasting."""

import numpy as np
import pandas as pd

from src.core.exceptions import DataValidationError
from src.forecasting.base import Forecaster, validate_positive_int
from src.processing import normalize_series


class NaiveForecaster(Forecaster):
    """Repeat the last valid observation for every future month."""

    name = "naive"

    def _predict(self, history: pd.DataFrame, horizon: int) -> np.ndarray:
        valid_values = history["value"].dropna()
        if valid_values.empty:
            raise DataValidationError("At least one non-missing value is required to forecast")
        return np.full(horizon, float(valid_values.iloc[-1]))


class SeasonalNaiveForecaster(Forecaster):
    """Repeat the values of the last ``period`` months (default: one year)."""

    name = "seasonal_naive"

    def __init__(self, period: int = 12) -> None:
        validate_positive_int(period, "period")
        self.period = period

    def _predict(self, history: pd.DataFrame, horizon: int) -> np.ndarray:
        seasonal_values = history["value"].tail(self.period)
        if len(seasonal_values) < self.period or seasonal_values.isna().any():
            raise DataValidationError(
                f"Seasonal-naive forecasting requires {self.period} recent valid observations"
            )
        return np.resize(seasonal_values.to_numpy(dtype=float), horizon)


def evaluate_baselines(
    data: pd.DataFrame, test_periods: int = 12, horizon: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one-month-ahead naive and seasonal-naive forecasts.

    Each of the last ``test_periods`` months is forecast from earlier
    observations only. This measures one-step-ahead accuracy, not the accuracy
    of a 12-month-ahead forecast; use ``rolling_backtest`` for that.

    ``horizon`` is the former name of ``test_periods`` and is kept so older
    callers keep working.
    """
    if horizon is not None:
        test_periods = horizon
    validate_positive_int(test_periods, "test_periods")
    normalized = normalize_series(data)
    if len(normalized) <= test_periods:
        raise DataValidationError("Not enough observations for the requested backtest period")

    values = normalized["value"]
    all_predictions = pd.DataFrame(
        {
            "date": normalized["date"],
            "actual": values,
            "naive_prediction": values.shift(1),
            "seasonal_naive_prediction": values.shift(12),
        }
    )
    predictions = all_predictions.tail(test_periods).reset_index(drop=True)

    metric_rows = []
    for model, prediction_column in (
        ("naive", "naive_prediction"),
        ("seasonal_naive", "seasonal_naive_prediction"),
    ):
        valid = predictions[["actual", prediction_column]].dropna()
        if valid.empty:
            raise DataValidationError(f"No valid observations available for {model} backtest")
        errors = valid["actual"] - valid[prediction_column]
        metric_rows.append(
            {
                "model": model,
                "observations": int(len(valid)),
                "mae": float(errors.abs().mean()),
                "rmse": float(np.sqrt((errors ** 2).mean())),
            }
        )
    return predictions, pd.DataFrame(metric_rows)


def forecast_baselines(data: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    """Forecast future monthly values with naive and seasonal-naive baselines."""
    naive = NaiveForecaster().forecast(data, horizon)
    seasonal = SeasonalNaiveForecaster().forecast(data, horizon)
    return pd.DataFrame(
        {
            "date": naive["date"],
            "naive_forecast": naive["forecast"],
            "seasonal_naive_forecast": seasonal["forecast"],
        }
    )
