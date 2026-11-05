"""Build tidy, Power BI-friendly CSV tables."""

import pandas as pd

from src.core.exceptions import DataValidationError
from src.processing import normalize_series


def build_bi_series(
    data: pd.DataFrame, series_id: str = "FEDFUNDS"
) -> pd.DataFrame:
    """Create one row per observed month with calendar and rate-change fields."""
    if not series_id or not series_id.strip():
        raise DataValidationError("series_id must not be empty")

    normalized = normalize_series(data)
    dates = pd.to_datetime(normalized["date"])
    values = normalized["value"]
    return pd.DataFrame(
        {
            "series_id": series_id.strip(),
            "date": normalized["date"],
            "year": dates.dt.year,
            "quarter": dates.dt.quarter,
            "month": dates.dt.month,
            "year_month": dates.dt.strftime("%Y-%m"),
            "value_percent": values,
            "previous_month_percent": values.shift(1),
            "change_percentage_points": values.diff(),
        }
    )


def build_bi_forecast(
    forecasts: pd.DataFrame, series_id: str = "FEDFUNDS"
) -> pd.DataFrame:
    """Convert baseline forecast columns to a tidy one-row-per-model table."""
    forecast_columns = {
        "naive_forecast": "naive",
        "seasonal_naive_forecast": "seasonal_naive",
    }
    required = {"date", *forecast_columns}
    missing = required.difference(forecasts.columns)
    if missing:
        raise DataValidationError(
            "Forecast data is missing required columns: " + ", ".join(sorted(missing))
        )
    if not series_id or not series_id.strip():
        raise DataValidationError("series_id must not be empty")

    tidy = forecasts.melt(
        id_vars=["date"],
        value_vars=list(forecast_columns),
        var_name="forecast_column",
        value_name="forecast_value_percent",
    )
    tidy["model"] = tidy["forecast_column"].map(forecast_columns)
    tidy["series_id"] = series_id.strip()
    return tidy[["series_id", "date", "model", "forecast_value_percent"]].sort_values(
        ["date", "model"], kind="stable"
    ).reset_index(drop=True)
