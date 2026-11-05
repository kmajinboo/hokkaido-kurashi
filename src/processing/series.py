"""Validation and normalization for economic time series."""

import pandas as pd

from src.core.exceptions import DataValidationError


def normalize_series(data: pd.DataFrame) -> pd.DataFrame:
    """Return a validated, date-sorted frame with ``date`` and ``value`` columns.

    Empty FRED values such as ``.`` are kept as missing values. Invalid dates,
    duplicate dates, and non-empty non-numeric values raise DataValidationError.
    """
    if not isinstance(data, pd.DataFrame):
        raise DataValidationError("Series data must be a pandas DataFrame")

    required = {"date", "value"}
    missing_columns = required.difference(data.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise DataValidationError(f"Series data is missing required columns: {names}")

    result = data.loc[:, ["date", "value"]].copy()
    if result.empty:
        return pd.DataFrame({"date": pd.Series(dtype="object"), "value": pd.Series(dtype="float64")})

    dates = pd.to_datetime(result["date"], errors="coerce")
    if dates.isna().any():
        raise DataValidationError("Series data contains an invalid or missing date")
    result["date"] = dates.dt.strftime("%Y-%m-%d")

    original_values = result["value"]
    numeric_values = pd.to_numeric(original_values, errors="coerce")
    value_text = original_values.astype("string").str.strip()
    allowed_missing = original_values.isna() | value_text.isin(["", "."])
    invalid_values = numeric_values.isna() & ~allowed_missing
    if invalid_values.any():
        raise DataValidationError("Series data contains a non-numeric value")
    result["value"] = numeric_values.astype("float64")

    if result["date"].duplicated().any():
        raise DataValidationError("Series data contains duplicate dates")

    return result.sort_values("date", kind="stable").reset_index(drop=True)
