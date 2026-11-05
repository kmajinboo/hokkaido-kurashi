"""北海道消費生活モニター価格動向調査「石油製品の価格推移」(Excel) as a DataSource.

Each month Hokkaido publishes ``04_suiiYYMM.xlsx`` with monthly prices of
kerosene, propane gas and gasoline for the whole of Hokkaido and six areas,
for the current and the previous fiscal year. Save each downloaded file in
``data/manual/hokkaido_monitor/``; all files there are read and combined, and
when two files give a value for the same month, the file whose data reach the
later month wins (so corrected or newer files replace older ones).

Catalog: ``source=hokkaido_monitor`` and ``source_id=<品目>:<地域>``, e.g.
``灯油:全道``, ``プロパンガス:十勝``, ``ガソリン:道北``.

品目: 灯油 (円/1L, 配達価格), プロパンガス (円/5㎥), ガソリン (円/1L, レギュラー)
地域: 全道, 道央, 道南, 道北, オホーツク, 十勝, 釧路・根室
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.core.exceptions import ConfigurationError, DataValidationError
from src.datasource.base import DataSource

PRODUCTS = {"灯油": "灯油", "プロパンガス": "プロパンガス", "ガソリン": "ガソリン"}
AREAS = ["全道", "道央", "道南", "道北", "オホーツク", "十勝", "釧路・根室"]
# matched after NFKC normalisation, which turns （） into ()
_FISCAL_YEAR = re.compile(r"《.*?\((\d{4})年度\)》")


def _norm(value) -> str:
    """NFKC (full/half-width), no spaces: '10　月' -> '10月', 'ｵﾎｰﾂｸ' -> 'オホーツク'."""
    if value is None:
        return ""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value)))


def _month(header) -> int | None:
    match = re.fullmatch(r"(\d{1,2})月", _norm(header))
    if match and 1 <= int(match.group(1)) <= 12:
        return int(match.group(1))
    return None


def read_monitor_file(path: Path) -> pd.DataFrame:
    """All monthly prices in one file: ``product, area, date, value``."""
    try:
        import openpyxl
    except ImportError as exc:
        raise ConfigurationError("openpyxl is missing. Run: python -m pip install openpyxl") from exc
    workbook = openpyxl.load_workbook(path, data_only=True, read_only=True)
    rows = [list(r) for sheet in workbook.worksheets for r in sheet.iter_rows(values_only=True)]
    workbook.close()

    records, product, year, months = [], None, None, None
    for row in rows:
        first = _norm(row[0] if row else "")
        if first.endswith("価格の推移"):
            product = next((p for p in PRODUCTS if p in first), None)
            year = months = None
            continue
        if first.startswith("石油製品価格調査結果"):
            product = None  # the single-month table after the series; not used
            continue
        if product is None:
            continue
        match = _FISCAL_YEAR.search(first)
        if match:
            year, months = int(match.group(1)), None
            continue
        found = {i: _month(v) for i, v in enumerate(row) if _month(v)}
        if len(found) >= 6:
            months = found
            continue
        if year is None or months is None or first not in AREAS:
            continue
        for col, month in months.items():
            value = row[col] if col < len(row) else None
            if isinstance(value, (int, float)):
                calendar_year = year if month >= 4 else year + 1  # fiscal year starts in April
                records.append(
                    {"product": product, "area": first,
                     "date": f"{calendar_year:04d}-{month:02d}-01", "value": float(value)}
                )
    if not records:
        raise DataValidationError(f"{path.name}: no price table found (is this the 石油製品の価格推移 file?)")
    return pd.DataFrame(records)


class HokkaidoMonitorSource(DataSource):
    """Prices from every monitor-survey Excel file in ``directory``."""

    name = "hokkaido_monitor"

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._cache: pd.DataFrame | None = None

    def load_all(self) -> pd.DataFrame:
        if self._cache is None:
            files = sorted(self.directory.glob("*.xlsx")) if self.directory.exists() else []
            files = [f for f in files if not f.name.startswith("~$")]  # Excel lock files
            if not files:
                raise ConfigurationError(f"No monitor-survey Excel files in {self.directory}")
            frames = []
            for path in files:
                frame = read_monitor_file(path)
                frames.append(frame.assign(_latest=frame["date"].max(), _file=path.name))
            combined = pd.concat(frames, ignore_index=True)
            # newer data (later last month, then file name) replaces older values
            combined = combined.sort_values(["_latest", "_file"]).drop_duplicates(
                subset=["product", "area", "date"], keep="last"
            )
            self._cache = combined.sort_values(["product", "area", "date"]).reset_index(drop=True)
        return self._cache

    def fetch(self, source_id: str, options: dict[str, str]) -> pd.DataFrame:
        product, sep, area = _norm(source_id).partition(":")
        if not sep or product not in PRODUCTS or area not in AREAS:
            raise ConfigurationError(
                f"source_id must be <品目>:<地域> with 品目 in {list(PRODUCTS)} and 地域 in {AREAS}; got {source_id!r}"
            )
        data = self.load_all()
        one = data[(data["product"] == product) & (data["area"] == area)]
        return one[["date", "value"]].reset_index(drop=True)
