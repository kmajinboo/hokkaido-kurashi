"""Forecaster interface shared by every forecasting model."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from src.core.exceptions import DataValidationError
from src.processing import normalize_series


def validate_positive_int(value: int, name: str) -> None:
    """Raise DataValidationError unless ``value`` is a positive integer."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise DataValidationError(f"{name} must be a positive integer")


def future_month_starts(last_date: str, horizon: int) -> pd.DatetimeIndex:
    """Return the ``horizon`` month-start dates following ``last_date``."""
    last_period = pd.Period(last_date, freq="M")
    return pd.period_range(last_period + 1, periods=horizon, freq="M").to_timestamp()


class Forecaster(ABC):
    """Interface for monthly forecasting models.

    A forecaster is stateless with respect to the data: ``forecast`` receives
    the history it may use and nothing else. This is what lets the rolling
    backtest hand each forecast origin only the observations available at that
    time. Subclasses implement ``_predict``; validation, normalization and the
    forecast dates are handled here so every model behaves the same way.
    """

    #: Unique model name used in output files and the registry.
    name: str = ""

    def forecast(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Forecast ``horizon`` months after the last row of ``history``.

        ``history`` needs ``date`` and ``value`` columns. Returns a frame with
        ``date`` (YYYY-MM-DD month starts) and ``forecast`` columns.
        """
        validate_positive_int(horizon, "horizon")
        normalized = normalize_series(history)
        values = np.asarray(self._predict(normalized, horizon), dtype=float)
        if values.shape != (horizon,):
            raise DataValidationError(
                f"{self.name or type(self).__name__} returned {values.size} values for horizon {horizon}"
            )
        dates = future_month_starts(normalized["date"].iloc[-1], horizon)
        return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "forecast": values})

    @abstractmethod
    def _predict(self, history: pd.DataFrame, horizon: int) -> np.ndarray:
        """Return ``horizon`` forecast values from a normalized history."""
        raise NotImplementedError
