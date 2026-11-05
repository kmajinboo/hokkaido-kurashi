"""
fred_api.py

FRED APIから経済指標を取得する
"""

import requests
import pandas as pd

from src.utils.config import FRED_API_KEY


BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


def get_series(series_id):

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json"
    }

    response = requests.get(BASE_URL, params=params)

    response.raise_for_status()

    data = response.json()["observations"]

    df = pd.DataFrame(data)

    df = df[["date", "value"]]

    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df