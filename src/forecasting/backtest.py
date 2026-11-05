"""Rolling-origin, multi-step backtest that works with any Forecaster."""

import numpy as np
import pandas as pd

from src.core.exceptions import ConfigurationError, DataValidationError
from src.forecasting.base import Forecaster, validate_positive_int
from src.processing import normalize_series

PREDICTION_COLUMNS = ["model", "origin_date", "target_date", "step", "actual", "forecast"]


def _require_regular_monthly(data: pd.DataFrame) -> None:
    periods = pd.PeriodIndex(pd.to_datetime(data["date"]), freq="M")
    if len(periods) > 1 and not (np.diff(periods.asi8) == 1).all():
        raise DataValidationError("Rolling backtest requires a regular monthly series without gaps")


def rolling_backtest(
    data: pd.DataFrame,
    forecaster: Forecaster,
    horizon: int = 12,
    n_origins: int = 120,
    min_train: int = 24,
) -> pd.DataFrame:
    """Forecast ``horizon`` months ahead from each of the latest ``n_origins`` origins.

    An origin is the last month whose actual value the forecaster may use.
    The forecaster receives only the observations up to and including the
    origin (an expanding window), so no later data leaks into a forecast. Only
    origins whose full ``horizon`` has actual values are used.

    Returns one row per model, origin and step with the columns
    ``model, origin_date, target_date, step, actual, forecast``.
    """
    if not isinstance(forecaster, Forecaster) or not forecaster.name:
        raise DataValidationError("forecaster must be a Forecaster with a non-empty name")
    validate_positive_int(horizon, "horizon")
    validate_positive_int(n_origins, "n_origins")
    validate_positive_int(min_train, "min_train")

    normalized = normalize_series(data)
    _require_regular_monthly(normalized)
    last_origin = len(normalized) - 1 - horizon
    first_origin = last_origin - n_origins + 1
    if first_origin + 1 < min_train:
        raise DataValidationError(
            "Not enough observations for the requested rolling backtest "
            f"(need at least {min_train + n_origins - 1 + horizon})"
        )

    dates = normalized["date"].tolist()
    values = normalized["value"].to_numpy(dtype=float)
    steps = np.arange(1, horizon + 1)
    frames = []
    for origin in range(first_origin, last_origin + 1):
        history = normalized.iloc[: origin + 1]
        try:
            forecast = forecaster.forecast(history, horizon)
        except ConfigurationError:
            raise
        except Exception as exc:
            raise DataValidationError(
                f"{forecaster.name} failed at forecast origin {dates[origin]}: {exc}"
            ) from exc
        target_dates = dates[origin + 1 : origin + 1 + horizon]
        if forecast["date"].tolist() != target_dates:
            raise DataValidationError("Forecast dates do not match the observed dates")
        frames.append(
            pd.DataFrame(
                {
                    "model": forecaster.name,
                    "origin_date": dates[origin],
                    "target_date": target_dates,
                    "step": steps,
                    "actual": values[origin + 1 : origin + 1 + horizon],
                    "forecast": forecast["forecast"].to_numpy(dtype=float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)[PREDICTION_COLUMNS]


def _error_table(predictions: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    valid = predictions.dropna(subset=["actual", "forecast"])
    if valid.empty:
        raise DataValidationError("No valid observations available for the backtest metrics")
    error = valid["actual"] - valid["forecast"]
    work = valid.assign(abs_error=error.abs(), sq_error=error ** 2)
    table = (
        work.groupby(keys, sort=True)
        .agg(n=("abs_error", "size"), mae=("abs_error", "mean"), mse=("sq_error", "mean"))
        .reset_index()
    )
    table["rmse"] = np.sqrt(table.pop("mse"))
    return table


def rolling_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """MAE and RMSE for each model and forecast step (1..horizon months ahead)."""
    return _error_table(predictions, ["model", "step"])


def compare_to_baseline(metrics: pd.DataFrame, baseline: str = "naive") -> pd.DataFrame:
    """Add ``mae_ratio_to_<baseline>``: each model's MAE divided by the baseline's MAE.

    Compared at the same forecast step. Below 1 means more accurate than the
    baseline at that step; above 1 means less accurate.
    """
    base = metrics.loc[metrics["model"] == baseline, ["step", "mae"]]
    if base.empty:
        raise DataValidationError(f"Baseline model {baseline!r} is not in the metrics")
    base = base.rename(columns={"mae": "_baseline_mae"})
    result = metrics.merge(base, on="step", how="left")
    result[f"mae_ratio_to_{baseline}"] = result["mae"] / result.pop("_baseline_mae")
    return result


def overall_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """MAE and RMSE per model, pooled over all steps (a quick summary only)."""
    return _error_table(predictions, ["model"])


def describe_backtest(predictions: pd.DataFrame) -> dict:
    """Summarize the conditions of a rolling backtest for reports."""
    return {
        "method": "rolling origin, expanding window (each forecast uses only data up to its origin)",
        "models": sorted(predictions["model"].unique().tolist()),
        "n_origins": int(predictions["origin_date"].nunique()),
        "first_origin": str(predictions["origin_date"].min()),
        "last_origin": str(predictions["origin_date"].max()),
        "max_forecast_horizon_months": int(predictions["step"].max()),
    }
