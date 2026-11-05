"""DataSource interface: every place EFP gets data from looks the same."""

from abc import ABC, abstractmethod

import pandas as pd


class DataSource(ABC):
    """Fetch one series from one source as ``date`` and ``value`` columns.

    ``source_id`` is the series' ID at the source and ``options`` are the
    extra settings from the catalog's ``source_options`` column. Validation,
    monthly alignment and saving are done by the collector, not here.
    """

    #: Name used in the catalog's ``source`` column (lowercase).
    name: str = ""

    @abstractmethod
    def fetch(self, source_id: str, options: dict[str, str]) -> pd.DataFrame:
        """Return a DataFrame with ``date`` and ``value`` columns."""
        raise NotImplementedError
