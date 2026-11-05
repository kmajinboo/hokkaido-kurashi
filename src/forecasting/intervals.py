"""Prediction intervals built from rolling-backtest errors.

The interval for "h months ahead" is the forecast plus the spread of the
errors the same model made h months ahead in past forecasts. An
``IntervalRule`` decides which past errors are used and how:

* ``symmetric=False``: lower/upper error percentiles (e.g. 10% and 90% for an
  80% interval). Keeps any lean towards rises or falls seen in the past.
* ``symmetric=True``: forecast plus/minus the 80th percentile of the absolute
  error. Ignores the direction of past surprises.
* ``n_origins``: use only the errors of the latest ``n_origins`` forecast origins.
* ``exclude_below``: drop errors of forecasts made when the series was below
  this level (e.g. 0.5 to leave out zero-interest-rate periods, when rates
  hardly moved and errors were close to zero).

The intervals assume future errors resemble the selected past errors; they
are a guide, not a guarantee. ``realtime_coverage`` checks how often they
would have contained the actual value.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.exceptions import DataValidationError
from src.forecasting.base import Forecaster, validate_positive_int
from src.processing import normalize_series


@dataclass(frozen=True)
class IntervalRule:
    """Which past errors build the interval, and whether it is symmetric."""

    symmetric: bool = False
    n_origins: int | None = None
    exclude_below: float | None = None

    def __post_init__(self) -> None:
        if self.n_origins is not None:
            validate_positive_int(self.n_origins, "n_origins")


def _validate_level(level: float) -> None:
    if not isinstance(level, (int, float)) or isinstance(level, bool) or not 0 < level < 1:
        raise DataValidationError("level must be a number between 0 and 1 (e.g. 0.8)")


def _errors(predictions: pd.DataFrame) -> pd.DataFrame:
    valid = predictions.dropna(subset=["actual", "forecast"])
    return valid.assign(error=valid["actual"] - valid["forecast"])


def _origin_values(data: pd.DataFrame) -> pd.Series:
    normalized = normalize_series(data)
    return pd.Series(normalized["value"].to_numpy(), index=normalized["date"])


def select_errors(
    errors: pd.DataFrame,
    origin_values: pd.Series,
    rule: IntervalRule,
    known_until: str | None = None,
) -> pd.DataFrame:
    """Errors the rule allows, optionally only those known by ``known_until``.

    An error is known once its target month's actual value is known, so with
    ``known_until`` only errors whose target is on or before that month are
    kept. The latest ``n_origins`` origins are taken first; ``exclude_below``
    then removes origins inside that window.
    """
    selected = errors if known_until is None else errors[errors["target_date"] <= known_until]
    if rule.n_origins is not None:
        latest = np.sort(selected["origin_date"].unique())[-rule.n_origins:]
        selected = selected[selected["origin_date"].isin(latest)]
    if rule.exclude_below is not None:
        level_at_origin = selected["origin_date"].map(origin_values)
        selected = selected[level_at_origin >= rule.exclude_below]
    return selected


def _bounds(errors: pd.DataFrame, level: float, symmetric: bool) -> pd.DataFrame:
    if errors.empty:
        return pd.DataFrame(columns=["model", "step", "n_errors", "error_lower", "error_upper"])
    tail = (1 - level) / 2
    grouped = errors.groupby(["model", "step"], sort=True)["error"]
    if symmetric:
        table = grouped.agg(
            n_errors="size", _q=lambda e: float(np.quantile(np.abs(e), level))
        ).reset_index()
        table["error_lower"] = -table["_q"]
        table["error_upper"] = table.pop("_q")
        return table
    return grouped.agg(
        n_errors="size",
        error_lower=lambda e: float(np.quantile(e, tail)),
        error_upper=lambda e: float(np.quantile(e, 1 - tail)),
    ).reset_index()


def error_quantiles(predictions: pd.DataFrame, level: float = 0.8, symmetric: bool = False) -> pd.DataFrame:
    """Error bounds per model and step (errors are actual minus forecast).

    Returns ``model, step, n_errors, error_lower, error_upper``.
    """
    _validate_level(level)
    errors = _errors(predictions)
    if errors.empty:
        raise DataValidationError("No valid backtest errors to build intervals from")
    return _bounds(errors, level, symmetric)


def forecast_with_intervals(
    data: pd.DataFrame,
    forecaster: Forecaster,
    predictions: pd.DataFrame,
    horizon: int = 12,
    level: float = 0.8,
    lower_bound: float | None = 0.0,
    rule: IntervalRule | None = None,
) -> pd.DataFrame:
    """Forecast the next ``horizon`` months with an interval for each month.

    ``predictions`` is rolling-backtest output containing this model's errors
    for every step up to ``horizon``. ``rule`` selects the errors (default:
    all of them, asymmetric). The interval is not allowed below
    ``lower_bound`` (None = no limit).

    Returns ``model, base_date, date, step, forecast, lower, upper,
    interval_level, n_errors``. ``base_date`` is the last observed month.
    """
    _validate_level(level)
    rule = rule or IntervalRule()
    errors = select_errors(
        _errors(predictions[predictions["model"] == forecaster.name]), _origin_values(data), rule
    )
    quantiles = _bounds(errors, level, rule.symmetric)
    missing = sorted(set(range(1, horizon + 1)) - set(quantiles["step"]))
    if missing:
        raise DataValidationError(
            f"Backtest errors for {forecaster.name} are missing for steps {missing}"
        )
    forecast = forecaster.forecast(data, horizon)
    forecast["step"] = np.arange(1, horizon + 1)
    result = forecast.merge(quantiles, on="step", how="left")
    lower = result["forecast"] + result["error_lower"]
    upper = result["forecast"] + result["error_upper"]
    if lower_bound is not None:
        lower = lower.clip(lower=lower_bound)
        upper = upper.clip(lower=lower_bound)
    return pd.DataFrame(
        {
            "model": forecaster.name,
            "base_date": normalize_series(data)["date"].iloc[-1],
            "date": result["date"],
            "step": result["step"],
            "forecast": result["forecast"],
            "lower": lower,
            "upper": upper,
            "interval_level": level,
            "n_errors": result["n_errors"].astype(int),
        }
    )


def realtime_coverage(
    predictions: pd.DataFrame,
    data: pd.DataFrame,
    rule: IntervalRule,
    level: float = 0.8,
    test_origins: int = 120,
    test_exclude_below: float | None = None,
    min_errors: int = 24,
) -> pd.DataFrame:
    """How often the rule's intervals would have contained the actual value.

    For each of the latest ``test_origins`` forecast origins, the interval is
    built only from errors already known at that origin (target month on or
    before it), exactly as it would have been at the time. Every rule is
    therefore tested on the same forecasts. Origins whose level is below
    ``test_exclude_below`` are left out of the test (e.g. to judge intervals
    for periods like now, away from zero rates). Steps with fewer than
    ``min_errors`` known errors are skipped and counted in ``skipped``.

    Returns per model ``coverage, test_forecasts, skipped, mean_width``
    (average upper minus lower, before any lower bound is applied).
    """
    _validate_level(level)
    validate_positive_int(test_origins, "test_origins")
    validate_positive_int(min_errors, "min_errors")
    errors = _errors(predictions)
    values = _origin_values(data)
    tests = np.sort(errors["origin_date"].unique())[-test_origins:]
    if test_exclude_below is not None:
        tests = [o for o in tests if values.get(o, np.nan) >= test_exclude_below]

    rows = []
    for model, model_errors in errors.groupby("model", sort=True):
        hits, widths, skipped = [], [], 0
        for origin in tests:
            bounds = _bounds(
                select_errors(model_errors, values, rule, known_until=origin), level, rule.symmetric
            )
            usable = bounds[bounds["n_errors"] >= min_errors]
            test_rows = model_errors[model_errors["origin_date"] == origin]
            merged = test_rows.merge(usable, on=["model", "step"], how="inner")
            skipped += len(test_rows) - len(merged)
            hits.extend((merged["error"] >= merged["error_lower"]) & (merged["error"] <= merged["error_upper"]))
            widths.extend(merged["error_upper"] - merged["error_lower"])
        rows.append(
            {
                "model": model,
                "symmetric": rule.symmetric,
                "error_origins": rule.n_origins,
                "exclude_below": rule.exclude_below,
                "coverage": float(np.mean(hits)) if hits else float("nan"),
                "test_forecasts": len(hits),
                "skipped": skipped,
                "mean_width": float(np.mean(widths)) if widths else float("nan"),
                "interval_level": level,
            }
        )
    return pd.DataFrame(rows)


def interval_coverage(predictions: pd.DataFrame, level: float = 0.8) -> pd.DataFrame:
    """Simple check: intervals from the earlier half of origins, tested on the later half.

    Kept for reference; ``realtime_coverage`` is the fairer comparison
    because it tests every rule on the same forecasts.

    Returns ``model, calibration_errors, test_forecasts, coverage, interval_level``.
    """
    _validate_level(level)
    errors = _errors(predictions)
    origins = np.sort(errors["origin_date"].unique())
    if len(origins) < 4:
        raise DataValidationError("At least 4 forecast origins are needed to check coverage")
    split_origin = origins[len(origins) // 2]
    calibration = errors[errors["target_date"] <= split_origin]
    test = errors[errors["origin_date"] >= split_origin]

    rows = []
    for model, test_rows in test.groupby("model", sort=True):
        bounds = _bounds(calibration[calibration["model"] == model], level, symmetric=False)
        merged = test_rows.merge(bounds, on=["model", "step"], how="inner")
        if merged.empty:
            continue
        inside = (merged["error"] >= merged["error_lower"]) & (merged["error"] <= merged["error_upper"])
        rows.append(
            {
                "model": model,
                "calibration_errors": int(bounds["n_errors"].sum()),
                "test_forecasts": int(len(merged)),
                "coverage": float(inside.mean()),
                "interval_level": level,
            }
        )
    return pd.DataFrame(rows)
