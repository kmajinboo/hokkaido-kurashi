"""FRED as a DataSource (wraps the existing FredClient)."""

import pandas as pd

from src.datasource.base import DataSource


class FredSource(DataSource):
    """Series from FRED. ``options`` are passed to the FRED API as parameters.

    For a daily series, ``frequency=m;aggregation_method=avg`` in the catalog
    asks FRED for monthly averages.
    """

    name = "fred"

    def __init__(self, client=None) -> None:
        # The client is created on first use, so a missing FRED key only
        # matters when a FRED series is actually collected.
        self._client = client

    def _get_client(self):
        if self._client is None:
            from src.api.fred import FredClient

            self._client = FredClient()
        return self._client

    def fetch(self, source_id: str, options: dict[str, str]) -> pd.DataFrame:
        return self._get_client().get_series(source_id, **options)
