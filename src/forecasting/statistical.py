"""Statistical forecasting models: drift, damped-trend ETS and ARIMA.

statsmodels is imported only when an ETS or ARIMA forecast is made, so the
rest of EFP keeps working without it. Install it with
``python -m pip install statsmodels``.
"""

import warnings

import numpy as np
import pandas as pd

from src.core.exceptions import ConfigurationError, DataValidationError
from src.forecasting.base import Forecaster, validate_positive_int


def _training_values(history: pd.DataFrame, window: int | None, min_obs: int, model: str) -> np.ndarray:
    """Return the values a model is fitted on (the last ``window`` months, or all).

    Missing values inside the training window are not filled in silently;
    they raise DataValidationError so the gap is noticed.
    """
    values = history["value"] if window is None else history["value"].tail(window)
    if values.isna().any():
        raise DataValidationError(f"{model} requires the training window to have no missing values")
    if len(values) < min_obs:
        raise DataValidationError(f"{model} requires at least {min_obs} observations, got {len(values)}")
    return values.to_numpy(dtype=float)


def _apply_lower_bound(values: np.ndarray, lower_bound: float | None) -> np.ndarray:
    return values if lower_bound is None else np.maximum(values, lower_bound)


def _import_statsmodels():
    try:
        import statsmodels  # noqa: F401
    except ImportError as exc:
        raise ConfigurationError(
            "statsmodels is missing. Run: python -m pip install statsmodels"
        ) from exc


class _StatisticalForecaster(Forecaster):
    """Shared settings: training window and lower bound for the forecasts.

    ``window``: fit on the last ``window`` months only (None = all history).
    ``lower_bound``: forecasts below this value are raised to it. The default
    0.0 keeps interest-rate forecasts from going negative; use None to turn it off.
    """

    min_obs = 2

    def __init__(self, window: int | None = None, lower_bound: float | None = 0.0) -> None:
        if window is not None:
            validate_positive_int(window, "window")
            if window < self.min_obs:
                raise DataValidationError(f"window must be at least {self.min_obs}")
        self.window = window
        self.lower_bound = lower_bound

    def _predict(self, history: pd.DataFrame, horizon: int) -> np.ndarray:
        values = _training_values(history, self.window, self.min_obs, self.name)
        return _apply_lower_bound(np.asarray(self._fit_predict(values, horizon), dtype=float), self.lower_bound)

    def _fit_predict(self, values: np.ndarray, horizon: int) -> np.ndarray:
        raise NotImplementedError


class DriftForecaster(_StatisticalForecaster):
    """Extend the line from the first to the last value of the training window.

    With the full history (default) this is the textbook drift method. Its
    slope is the average monthly change since the series began.
    """

    name = "drift"

    def _fit_predict(self, values: np.ndarray, horizon: int) -> np.ndarray:
        slope = (values[-1] - values[0]) / (len(values) - 1)
        return values[-1] + slope * np.arange(1, horizon + 1)


class RecentDriftForecaster(DriftForecaster):
    """Drift over the last 24 months: continues the recent trend."""

    name = "drift_24m"

    def __init__(self, window: int | None = 24, lower_bound: float | None = 0.0) -> None:
        super().__init__(window=window, lower_bound=lower_bound)


class ETSForecaster(_StatisticalForecaster):
    """Exponential smoothing with an additive, damped trend (Holt's damped method).

    The trend fades out over the forecast horizon, so the forecast levels off
    instead of extending a straight line.
    """

    name = "ets_damped"
    min_obs = 24

    def _fit_predict(self, values: np.ndarray, horizon: int) -> np.ndarray:
        _import_statsmodels()
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ExponentialSmoothing(
                values, trend="add", damped_trend=True, initialization_method="estimated"
            ).fit()
        return fitted.forecast(horizon)


class ARIMAForecaster(_StatisticalForecaster):
    """ARIMA model; the default order (1, 1, 0) models the monthly changes.

    With d=1 and no constant, the forecast starts from the last value and
    follows the momentum of recent changes, which fades out over time.
    """

    name = "arima"
    min_obs = 36

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 0),
        window: int | None = None,
        lower_bound: float | None = 0.0,
    ) -> None:
        super().__init__(window=window, lower_bound=lower_bound)
        if len(order) != 3 or not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in order):
            raise DataValidationError("order must be three non-negative integers (p, d, q)")
        self.order = tuple(order)

    def _fit_predict(self, values: np.ndarray, horizon: int) -> np.ndarray:
        _import_statsmodels()
        from statsmodels.tsa.arima.model import ARIMA

        trend = "n" if self.order[1] > 0 else "c"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ARIMA(values, order=self.order, trend=trend).fit()
        return fitted.forecast(horizon)
