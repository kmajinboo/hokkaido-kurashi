"""FRED (Federal Reserve Economic Data) API client."""

from typing import Any

import pandas as pd
import requests

from src.api.base import BaseAPIClient
from src.core.config import API_KEYS
from src.core.exceptions import APIConnectionError, ConfigurationError


class FredClient(BaseAPIClient):
    """FRED observations API client."""

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str | None = None, timeout: float = 30.0) -> None:
        self.api_key = api_key or API_KEYS.get("fred")
        if not self.api_key:
            raise ConfigurationError(
                "FRED_API_KEY is not configured. Add it to the project .env file."
            )
        self.timeout = timeout

    def get_series(self, series_id: str, **kwargs: Any) -> pd.DataFrame:
        """Fetch a FRED series as date/value columns with numeric values."""
        if not series_id or not series_id.strip():
            raise ValueError("series_id must not be empty")

        params = {
            "series_id": series_id.strip(),
            "api_key": self.api_key,
            "file_type": "json",
            **kwargs,
        }
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            observations = response.json()["observations"]
        except (requests.RequestException, ValueError, KeyError) as exc:
            raise APIConnectionError(f"FRED request failed for {series_id!r}") from exc

        frame = pd.DataFrame(observations)
        if frame.empty:
            return pd.DataFrame({"date": pd.Series(dtype="object"), "value": pd.Series(dtype="float64")})
        frame = frame[["date", "value"]].copy()
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        return frame
