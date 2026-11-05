"""Series catalog: the list of series EFP observes, kept in an editable CSV.

The catalog lives in ``config/series_catalog.csv`` (UTF-8 with BOM, so it
opens correctly in Excel). One row per series:

=============== ==========================================================
column          meaning
=============== ==========================================================
series_id       EFP's own ID (letters, digits, underscore), e.g. USDJPY
active          1 = collect, 0 = skip (e.g. a manual series not filled yet)
layer           世界 / 国内 / 北海道 / 相談 (the layer in the ripple chain)
category        金利 / 為替 / エネルギー / 食料 / 物価 / 雇用 ...
name            display name in Japanese
unit            unit of the values
frequency       M (monthly) or Q (quarterly; stored on the quarter's first month)
source          fred / manual (see ``src/datasource``)
source_id       the ID at the source (FRED series ID, manual file name)
source_options  extra settings as key=value;key=value (optional)
note            free text (optional)
compare         how to compare with a year earlier (optional):
                yoy_pct  = % change from the same month a year earlier
                yoy_diff = difference from a year earlier (for rates: %, 倍)
                as_is    = already a year-on-year change; use as it is
                empty    = chosen from the unit (%, 倍 -> yoy_diff; else yoy_pct)
=============== ==========================================================
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.core.exceptions import ConfigurationError

REQUIRED_COLUMNS = [
    "series_id",
    "active",
    "layer",
    "category",
    "name",
    "unit",
    "frequency",
    "source",
    "source_id",
]
SUPPORTED_FREQUENCIES = {"M", "Q"}  # monthly, quarterly (stored on the first month of the quarter)
COMPARE_TYPES = {"yoy_pct", "yoy_diff", "as_is"}
_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class SeriesSpec:
    """One observed series as described in the catalog."""

    series_id: str
    name: str
    source: str
    source_id: str
    frequency: str = "M"
    unit: str = ""
    layer: str = ""
    category: str = ""
    active: bool = True
    options: dict = field(default_factory=dict)
    note: str = ""
    compare: str = ""


def parse_options(text: str) -> dict[str, str]:
    """Turn ``"frequency=m;aggregation_method=avg"`` into a dict."""
    options: dict[str, str] = {}
    for part in (text or "").split(";"):
        part = part.strip()
        if not part:
            continue
        key, sep, value = part.partition("=")
        if not sep or not key.strip():
            raise ConfigurationError(f"source_options entry {part!r} must look like key=value")
        options[key.strip()] = value.strip()
    return options


def _cell(row: pd.Series, column: str) -> str:
    value = row.get(column, "")
    return "" if pd.isna(value) else str(value).strip()


def load_catalog(path: Path) -> list[SeriesSpec]:
    """Read and validate the catalog CSV. Errors name the CSV line to fix."""
    if not Path(path).exists():
        raise ConfigurationError(f"Series catalog not found: {path}")
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ConfigurationError(f"Series catalog is missing columns: {', '.join(missing)}")

    specs: list[SeriesSpec] = []
    seen: set[str] = set()
    for index, row in frame.iterrows():
        line = index + 2  # header is line 1
        series_id = _cell(row, "series_id")
        if not series_id:
            continue  # blank line
        if not _ID_PATTERN.match(series_id):
            raise ConfigurationError(
                f"Catalog line {line}: series_id {series_id!r} must start with a letter and use only letters, digits and _"
            )
        if series_id in seen:
            raise ConfigurationError(f"Catalog line {line}: series_id {series_id!r} appears twice")
        seen.add(series_id)
        active_text = _cell(row, "active")
        if active_text not in {"0", "1"}:
            raise ConfigurationError(f"Catalog line {line}: active must be 1 or 0")
        frequency = _cell(row, "frequency").upper()
        if frequency not in SUPPORTED_FREQUENCIES:
            raise ConfigurationError(
                f"Catalog line {line}: frequency {frequency!r} is not supported (use M or Q)"
            )
        for column in ("name", "source", "source_id"):
            if not _cell(row, column):
                raise ConfigurationError(f"Catalog line {line}: {column} is empty")
        compare = _cell(row, "compare").lower()
        if compare and compare not in COMPARE_TYPES:
            raise ConfigurationError(
                f"Catalog line {line}: compare must be one of {sorted(COMPARE_TYPES)} or empty"
            )
        try:
            options = parse_options(_cell(row, "source_options"))
        except ConfigurationError as exc:
            raise ConfigurationError(f"Catalog line {line}: {exc}") from None
        specs.append(
            SeriesSpec(
                series_id=series_id,
                name=_cell(row, "name"),
                source=_cell(row, "source").lower(),
                source_id=_cell(row, "source_id"),
                frequency=frequency,
                unit=_cell(row, "unit"),
                layer=_cell(row, "layer"),
                category=_cell(row, "category"),
                active=active_text == "1",
                options=options,
                note=_cell(row, "note"),
                compare=compare,
            )
        )
    return specs


def catalog_frame(specs: list[SeriesSpec]) -> pd.DataFrame:
    """Catalog as a table (for the database and Power BI)."""
    return pd.DataFrame(
        [
            {
                "series_id": s.series_id,
                "name": s.name,
                "unit": s.unit,
                "frequency": s.frequency,
                "layer": s.layer,
                "category": s.category,
                "source": s.source,
                "source_id": s.source_id,
            }
            for s in specs
        ],
        columns=["series_id", "name", "unit", "frequency", "layer", "category", "source", "source_id"],
    )
