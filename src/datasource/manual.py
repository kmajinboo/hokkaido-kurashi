"""Values typed in by hand, for data published only as PDF (e.g. Hokkaido figures).

Each series is one CSV file ``data/manual/<source_id>.csv`` with two columns:

    date,value
    2026-08,1.02
    2026-09,1.01

``date`` may be written as 2026-09, 2026/9, 2026-09-01 or 2026年9月.
Save the file as CSV (UTF-8); Excel's "CSV UTF-8" format works.

For values that change only on certain dates (pension revisions every April,
minimum wage every October), put ``fill=forward`` in the catalog's
source_options and write one row per change. Extra columns (e.g. a source
column for the reference) are allowed and ignored.
"""

import re
from datetime import date
from pathlib import Path

import pandas as pd

from src.core.exceptions import ConfigurationError, DataValidationError
from src.datasource.base import DataSource

_JAPANESE_MONTH = re.compile(r"^\s*(\d{4})\s*年\s*(\d{1,2})\s*月\s*(?:(\d{1,2})\s*日)?\s*$")


def _normalize_date_text(text: str) -> str:
    text = str(text).strip()
    match = _JAPANESE_MONTH.match(text)
    if match:
        year, month, day = match.groups()
        return f"{year}-{int(month):02d}-{int(day or 1):02d}"
    if re.fullmatch(r"\d{4}[-/]\d{1,2}", text):
        year, month = re.split(r"[-/]", text)
        return f"{year}-{int(month):02d}-01"
    return text.replace("/", "-")


class ManualCsvSource(DataSource):
    """Read hand-entered monthly values from ``<directory>/<source_id>.csv``."""

    name = "manual"

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def fetch(self, source_id: str, options: dict[str, str]) -> pd.DataFrame:
        path = self.directory / f"{source_id}.csv"
        if not path.exists():
            raise ConfigurationError(f"Manual data file not found: {path}")
        frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        if not {"date", "value"} <= set(frame.columns):
            raise DataValidationError(f"{path.name} needs the columns date and value")
        frame = frame[frame["date"].str.strip() != ""]
        result = pd.DataFrame(
            {
                "date": frame["date"].map(_normalize_date_text),
                "value": frame["value"].str.replace(",", "", regex=False).str.strip(),
            }
        )
        if options.get("fill", "").lower() == "forward":
            result = self._fill_forward(result, path.name)
        return result

    def _fill_forward(self, frame: pd.DataFrame, file_name: str) -> pd.DataFrame:
        """Values that change on certain dates (e.g. every April or October).

        With ``fill=forward`` in the catalog, write one row per change; each
        value is repeated for every month until the next change, and the last
        one up to the current month.
        """
        values = pd.to_numeric(frame["value"], errors="coerce")
        if values.isna().any() or frame.empty:
            raise DataValidationError(f"{file_name}: every row needs a numeric value for fill=forward")
        months = pd.PeriodIndex(pd.to_datetime(frame["date"]), freq="M")
        if months.duplicated().any():
            raise DataValidationError(f"{file_name}: two changes in the same month")
        steps = pd.Series(values.to_numpy(), index=months).sort_index()
        end = max(steps.index.max(), pd.Period(date.today(), freq="M"))
        full = steps.reindex(pd.period_range(steps.index.min(), end, freq="M")).ffill()
        return pd.DataFrame({"date": full.index.to_timestamp().strftime("%Y-%m-%d"), "value": full.to_numpy()})
