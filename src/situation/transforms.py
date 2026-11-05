"""Put series with different units on one scale: change from a year earlier.

Comparing with the same month a year earlier also removes the usual seasonal
pattern (winter heating, spring hiring), so what remains is this year's
difference from a normal year.
"""

import numpy as np
import pandas as pd

from src.catalog import SeriesSpec

CHANGE_UNITS = {"yoy_pct": "%", "yoy_diff": "ポイント", "as_is": "%"}
CHANGE_LABELS = {"yoy_pct": "前年同月比", "yoy_diff": "前年同月差", "as_is": "前年同月比（公表値）"}


def compare_type(spec: SeriesSpec) -> str:
    """The catalog's ``compare`` setting, or a default chosen from the unit."""
    if spec.compare:
        return spec.compare
    return "yoy_diff" if spec.unit.strip() in {"%", "倍"} else "yoy_pct"


def _shift_months(dates: pd.Series, months: int) -> pd.Series:
    periods = pd.PeriodIndex(pd.to_datetime(dates), freq="M") + months
    return pd.Series(periods.to_timestamp().strftime("%Y-%m-%d"), index=dates.index)


def yoy_changes(observations: pd.DataFrame, specs: list[SeriesSpec]) -> pd.DataFrame:
    """Change from a year earlier for every series.

    Matches each month with the same month a year earlier by date (not by
    position), so a missing month never pairs the wrong months. ``as_is``
    series are copied unchanged.

    Returns ``series_id, date, value, compare, change``.
    """
    frames = []
    for spec in specs:
        one = observations[observations["series_id"] == spec.series_id][["date", "value"]]
        if one.empty:
            continue
        kind = compare_type(spec)
        if kind == "as_is":
            change = one["value"].astype(float)
        else:
            previous = one.assign(date=_shift_months(one["date"], 12)).rename(columns={"value": "prev"})
            merged = one.merge(previous, on="date", how="left")
            if kind == "yoy_pct":
                with np.errstate(divide="ignore", invalid="ignore"):
                    change = (merged["value"] / merged["prev"] - 1) * 100
                change = change.where(merged["prev"].abs() > 0)
            else:
                change = merged["value"] - merged["prev"]
            change.index = one.index
        frames.append(
            pd.DataFrame(
                {"series_id": spec.series_id, "date": one["date"], "value": one["value"],
                 "compare": kind, "change": change.astype(float)}
            )
        )
    columns = ["series_id", "date", "value", "compare", "change"]
    return pd.concat(frames, ignore_index=True)[columns] if frames else pd.DataFrame(columns=columns)


def latest_summary(changes: pd.DataFrame, specs: list[SeriesSpec], history_months: int = 120) -> pd.DataFrame:
    """Latest value and change of each series, and how unusual the change is.

    ``percentile`` is the share of the past ``history_months`` changes (before
    the latest month) that were lower than the latest change: 95 means the
    latest change is higher than 95% of the past ten years.

    Returns one row per series with ``series_id, name, layer, category, unit,
    date, value, compare, change, change_unit, change_3m_ago, percentile, n_history``.
    """
    rows = []
    for spec in specs:
        one = changes[changes["series_id"] == spec.series_id].dropna(subset=["value"])
        if one.empty:
            continue
        last = one.iloc[-1]
        valid = one.dropna(subset=["change"])
        past = valid[valid["date"] < last["date"]].tail(history_months)["change"]
        three_ago = valid[valid["date"] == _shift_months(pd.Series([last["date"]]), -3).iloc[0]]["change"]
        change = last["change"]
        rows.append(
            {
                "series_id": spec.series_id, "name": spec.name, "layer": spec.layer,
                "category": spec.category, "unit": spec.unit, "date": last["date"],
                "value": float(last["value"]), "compare": last["compare"],
                "change": float(change) if pd.notna(change) else np.nan,
                "change_unit": CHANGE_UNITS[last["compare"]],
                "change_3m_ago": float(three_ago.iloc[0]) if len(three_ago) else np.nan,
                "percentile": float((past < change).mean() * 100) if len(past) >= 24 and pd.notna(change) else np.nan,
                "n_history": int(len(past)),
            }
        )
    return pd.DataFrame(rows)
