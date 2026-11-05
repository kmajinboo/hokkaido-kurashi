"""Data sources (FRED, hand-entered CSV, ...) and the collector."""

from src.datasource.base import DataSource
from src.datasource.collect import collect, series_values, to_monthly
from src.datasource.estat import EStatSource
from src.datasource.fred import FredSource
from src.datasource.hokkaido_monitor import HokkaidoMonitorSource
from src.datasource.manual import ManualCsvSource

__all__ = [
    "DataSource", "EStatSource", "FredSource", "HokkaidoMonitorSource", "ManualCsvSource", "collect", "series_values", "to_monthly",
]
