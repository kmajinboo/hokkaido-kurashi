"""Leakage-safe lag features for time-series forecasting."""

import pandas as pd

from src.core.exceptions import DataValidationError
from src.processing import normalize_series


def build_lag_features(
    data: pd.DataFrame,
    lags: tuple[int, ...] = (1, 3, 12),
    rolling_windows: tuple[int, ...] = (3, 12),
) -> pd.DataFrame:
    """Build a supervised frame using only observations before each target date.

    The current observation is returned as ``target_value``. Lag, change, and
    rolling-average columns only use earlier observations. Leading rows remain
    present with null feature values so observation dates are not hidden.
    """
    if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in lags):
        raise DataValidationError("lags must contain positive integers")
    if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in rolling_windows):
        raise DataValidationError("rolling_windows must contain positive integers")
    if len(set(lags)) != len(lags) or len(set(rolling_windows)) != len(rolling_windows):
        raise DataValidationError("lags and rolling_windows must not contain duplicates")

    normalized = normalize_series(data)
    values = normalized["value"]
    result = pd.DataFrame({"date": normalized["date"], "target_value": values})

    for lag in lags:
        result[f"lag_{lag}"] = values.shift(lag)

    result["change_1"] = values.shift(1) - values.shift(2)
    prior_values = values.shift(1)
    for window in rolling_windows:
        result[f"mean_prev_{window}"] = prior_values.rolling(
            window=window, min_periods=window
        ).mean()

    return result
