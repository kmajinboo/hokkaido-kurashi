"""Power BI table that joins recent actual values with the forecast and its interval."""

import numpy as np
import pandas as pd

from src.core.exceptions import DataValidationError
from src.forecasting.base import validate_positive_int
from src.processing import normalize_series

COLUMNS = ["date", "model", "kind", "actual", "forecast", "lower", "upper"]


def build_history_forecast(
    data: pd.DataFrame, interval_forecasts: pd.DataFrame, history_months: int = 60
) -> pd.DataFrame:
    """One table for a single chart: past actual values followed by the forecast.

    For each model in ``interval_forecasts`` the latest ``history_months`` of
    actual values are repeated (``kind = actual``), so filtering the chart by
    ``model`` keeps the history. On the last actual month, ``forecast``,
    ``lower`` and ``upper`` are set to the actual value so the forecast lines
    start from the last observation instead of leaving a gap.

    Returns ``date, model, kind, actual, forecast, lower, upper``.
    """
    validate_positive_int(history_months, "history_months")
    history = normalize_series(data).tail(history_months)
    if history.empty:
        raise DataValidationError("At least one actual observation is required for the chart")
    missing = {"date", "model", "forecast", "lower", "upper"} - set(interval_forecasts.columns)
    if missing:
        raise DataValidationError(f"Forecast table is missing columns: {', '.join(sorted(missing))}")
    if interval_forecasts.empty:
        raise DataValidationError("Forecast table is empty")

    last_date = history["date"].iloc[-1]
    last_value = history["value"].iloc[-1]
    frames = []
    for model, forecast in interval_forecasts.groupby("model", sort=False):
        if forecast["date"].min() <= last_date:
            raise DataValidationError(f"Forecasts for {model} must start after the last actual month")
        past = pd.DataFrame(
            {
                "date": history["date"].to_numpy(),
                "model": model,
                "kind": "actual",
                "actual": history["value"].to_numpy(dtype=float),
                "forecast": np.nan,
                "lower": np.nan,
                "upper": np.nan,
            }
        )
        past.loc[past.index[-1], ["forecast", "lower", "upper"]] = last_value
        future = pd.DataFrame(
            {
                "date": forecast["date"].to_numpy(),
                "model": model,
                "kind": "forecast",
                "actual": np.nan,
                "forecast": forecast["forecast"].to_numpy(dtype=float),
                "lower": forecast["lower"].to_numpy(dtype=float),
                "upper": forecast["upper"].to_numpy(dtype=float),
            }
        )
        frames.extend([past, future])
    return pd.concat(frames, ignore_index=True)[COLUMNS]
