"""Descriptive and annual statistics for economic time series."""

import pandas as pd

from src.processing import normalize_series


def descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Return key descriptive statistics as a two-column metric/value table."""
    normalized = normalize_series(data)
    values = normalized["value"].dropna()
    dates = normalized["date"]

    latest_change = values.diff().iloc[-1] if len(values) > 1 else float("nan")
    metrics = [
        ("observations", int(len(normalized))),
        ("valid_values", int(values.count())),
        ("missing_values", int(normalized["value"].isna().sum())),
        ("start_date", dates.iloc[0] if len(dates) else None),
        ("end_date", dates.iloc[-1] if len(dates) else None),
        ("mean", values.mean()),
        ("median", values.median()),
        ("sample_std_dev", values.std(ddof=1)),
        ("minimum", values.min()),
        ("maximum", values.max()),
        ("latest_valid_value", values.iloc[-1] if len(values) else float("nan")),
        ("latest_observation_change", latest_change),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value"])


def annual_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly observations into calendar-year statistics."""
    normalized = normalize_series(data)
    if normalized.empty:
        return pd.DataFrame(
            columns=["year", "observations", "valid_values", "missing_values", "mean", "median", "sample_std_dev", "minimum", "maximum", "year_end_value"]
        )

    frame = normalized.copy()
    frame["year"] = pd.to_datetime(frame["date"]).dt.year
    grouped = frame.groupby("year", sort=True)["value"]
    result = grouped.agg(
        observations="size",
        valid_values="count",
        mean="mean",
        median="median",
        sample_std_dev="std",
        minimum="min",
        maximum="max",
    ).reset_index()
    result["missing_values"] = result["observations"] - result["valid_values"]
    year_end = (
        frame.dropna(subset=["value"])
        .groupby("year", sort=True)
        .tail(1)[["year", "value"]]
        .rename(columns={"value": "year_end_value"})
    )
    result = result.merge(year_end, on="year", how="left")
    return result[
        ["year", "observations", "valid_values", "missing_values", "mean", "median", "sample_std_dev", "minimum", "maximum", "year_end_value"]
    ]
