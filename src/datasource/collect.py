"""Collect every active catalog series into the common long format and save it.

Common format (one row per series and month):
``series_id, date (YYYY-MM-01), value, source, retrieved_at``.

One failing series does not stop the others; each series gets a line in the
returned summary with ``status`` ok or error and a message.
"""

import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.catalog import SeriesSpec
from src.core.exceptions import DataValidationError
from src.datasource.base import DataSource
from src.processing import normalize_series

OBSERVATION_COLUMNS = ["series_id", "date", "value", "source", "retrieved_at"]
SUMMARY_COLUMNS = [
    "series_id", "name", "status", "rows", "first_date", "last_date", "last_value", "message",
]
_SECRET = re.compile(r"(api_key|appId|apikey|token)=[^&\s'\"]+", re.IGNORECASE)


def _redact(text: str) -> str:
    """Hide anything that looks like a key in an error message."""
    return _SECRET.sub(r"\1=***", text)


def to_monthly(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and put every date on the first day of its month.

    Two values in the same month are an error rather than being averaged
    silently; ask the source for monthly data instead.
    """
    normalized = normalize_series(frame)
    if normalized.empty:
        return normalized
    months = pd.PeriodIndex(pd.to_datetime(normalized["date"]), freq="M")
    if months.duplicated().any():
        raise DataValidationError(
            "More than one value in the same month; request monthly data from the source"
        )
    normalized["date"] = months.to_timestamp().strftime("%Y-%m-%d")
    return normalized


def collect(
    specs: list[SeriesSpec],
    sources: dict[str, DataSource],
    repository=None,
    now: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch all active series, save them, and report how each one went.

    Returns ``(observations, summary)``. ``repository`` (optional) receives
    the catalog of active series and all successfully fetched observations.
    """
    retrieved_at = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    frames, rows = [], []
    for spec in specs:
        if not spec.active:
            continue
        summary = {"series_id": spec.series_id, "name": spec.name}
        try:
            source = sources.get(spec.source)
            if source is None:
                raise DataValidationError(
                    f"Unknown source {spec.source!r}. Available: {', '.join(sorted(sources))}"
                )
            data = to_monthly(source.fetch(spec.source_id, dict(spec.options)))
            if data["value"].notna().sum() == 0:
                raise DataValidationError("No values were returned")
        except Exception as exc:  # one series failing must not stop the rest
            rows.append(
                {**summary, "status": "error", "rows": 0, "first_date": None, "last_date": None,
                 "last_value": np.nan, "message": _redact(f"{type(exc).__name__}: {exc}")}
            )
            continue
        valid = data.dropna(subset=["value"])
        frames.append(
            data.assign(series_id=spec.series_id, source=spec.source, retrieved_at=retrieved_at)
        )
        rows.append(
            {**summary, "status": "ok", "rows": int(len(data)), "first_date": data["date"].iloc[0],
             "last_date": valid["date"].iloc[-1], "last_value": float(valid["value"].iloc[-1]),
             "message": ""}
        )

    observations = (
        pd.concat(frames, ignore_index=True)[OBSERVATION_COLUMNS]
        if frames
        else pd.DataFrame(columns=OBSERVATION_COLUMNS)
    )
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    if repository is not None:
        repository.save_catalog([s for s in specs if s.active])
        if not observations.empty:
            repository.save_observations(observations)
    return observations, summary


def series_values(observations: pd.DataFrame, series_id: str) -> pd.DataFrame:
    """``date`` and ``value`` of one series from the long table."""
    one = observations[observations["series_id"] == series_id]
    if one.empty:
        raise DataValidationError(f"Series {series_id} was not collected; check the summary above")
    return one[["date", "value"]].reset_index(drop=True)
