"""e-Stat (Japanese government statistics) as a DataSource.

Catalog example (one row in ``config/series_catalog.csv``)::

    source=estat
    source_id=<statsDataId>                    e.g. 0004052037
    source_options=cdArea=...;cdCat01=...;cdTab=...

``source_options`` narrow the table down to ONE series (one area, one item,
one kind of figure). If more than one remains, the error lists every
dimension that still has several codes, with their names, so the right
``cd...`` setting can be added. Use ``python estat_tools.py meta <ID>`` to
look up the codes.

The application ID is read from ``ESTAT_APP_ID`` in the project .env file.
"""

import os
import re
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.core import config  # noqa: F401  (loads the project .env)
from src.core.exceptions import APIConnectionError, ConfigurationError, DataValidationError
from src.datasource.base import DataSource

BASE_URL = "https://api.e-stat.go.jp/rest/3.0/app/json"
_MAX_PAGES = 50
_MONTH_NAME = re.compile(r"(\d{4})年\s*(\d{1,2})月")
_RESERVED = {"appid", "statsdataid", "startposition", "limit"}


def as_list(value: Any) -> list:
    """e-Stat returns a dict for one item and a list for several."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def text_of(value: Any) -> str:
    """Text of a field that may be a plain string or {"@code": ..., "$": ...}."""
    if isinstance(value, dict):
        return str(value.get("$", ""))
    return "" if value is None else str(value)


def app_id() -> str:
    key = (os.getenv("ESTAT_APP_ID") or "").strip()
    if not key:
        raise ConfigurationError("ESTAT_APP_ID is not configured in the project .env file.")
    return key


def call(endpoint: str, params: dict[str, Any], timeout: float = 60.0) -> dict:
    """Call one e-Stat API endpoint and return the inner result object.

    Raises APIConnectionError with e-Stat's own message when STATUS is an error.
    The application ID never appears in error messages.
    """
    try:
        response = requests.get(
            f"{BASE_URL}/{endpoint}", params={"appId": app_id(), **params}, timeout=timeout
        )
        response.raise_for_status()
        body = response.json()
    except ConfigurationError:
        raise
    except requests.Timeout as exc:
        raise APIConnectionError(f"e-Stat request timed out ({endpoint}); try again later") from exc
    except requests.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else "?"
        raise APIConnectionError(f"e-Stat returned HTTP {code} ({endpoint}); try again later") from exc
    except requests.ConnectionError as exc:
        raise APIConnectionError(f"Could not connect to e-Stat ({endpoint}); check the network") from exc
    except requests.RequestException as exc:
        raise APIConnectionError(f"e-Stat request failed ({endpoint}): {type(exc).__name__}") from exc
    except ValueError as exc:
        raise APIConnectionError(f"e-Stat returned an unexpected (non-JSON) response ({endpoint})") from exc
    root_key = {"getStatsData": "GET_STATS_DATA", "getStatsList": "GET_STATS_LIST", "getMetaInfo": "GET_META_INFO"}[endpoint]
    root = body.get(root_key, {})
    result = root.get("RESULT", {})
    status = int(result.get("STATUS", -1))
    if status not in (0, 1, 2):
        raise APIConnectionError(f"e-Stat error {status}: {result.get('ERROR_MSG', 'unknown error')}")
    return root


_QUARTER_NAME = re.compile(r"(\d{4})年\s*(\d{1,2})\s*[～〜~\-－ー]\s*(\d{1,2})月期")
_QUARTER_NUMBER = re.compile(r"(\d{4})年?\s*第?\s*([1-4])\s*四半期")


def parse_quarter(code: str, name: str = "") -> str | None:
    """Quarter start as YYYY-MM-01, from e.g. 2025年4～6月期, 2025年第2四半期 or 2025000406."""
    text = name or ""
    match = _QUARTER_NAME.search(text)
    if match:
        year, first, last = (int(g) for g in match.groups())
        if 1 <= first <= 12 and (last - first) % 12 == 2:
            return f"{year:04d}-{first:02d}-01"
    match = _QUARTER_NUMBER.search(text)
    if match:
        year, quarter = int(match.group(1)), int(match.group(2))
        return f"{year:04d}-{3 * quarter - 2:02d}-01"
    code = str(code)
    if re.fullmatch(r"\d{10}", code) and code[4:6] == "00":
        first, last = int(code[6:8]), int(code[8:10])
        if 1 <= first <= 12 and last - first == 2:
            return f"{code[:4]}-{first:02d}-01"
    return None


def parse_month(code: str, name: str = "") -> str | None:
    """Turn an e-Stat time code (or its name, e.g. 2025年1月) into YYYY-MM-01.

    Monthly codes look like 2025000101 (year, 00, month, month). Annual and
    quarterly codes return None.
    """
    match = _MONTH_NAME.search(name or "")
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12:
            return f"{year:04d}-{month:02d}-01"
    code = str(code)
    if re.fullmatch(r"\d{10}", code) and code[4:6] == "00" and code[6:8] == code[8:10]:
        month = int(code[6:8])
        if 1 <= month <= 12:
            return f"{code[:4]}-{month:02d}-01"
    return None


def class_names(statistical_data: dict) -> dict[str, dict]:
    """{dimension id: {"name": ..., "codes": {code: name}}} from CLASS_INF."""
    result = {}
    for obj in as_list(statistical_data.get("CLASS_INF", {}).get("CLASS_OBJ")):
        result[obj.get("@id")] = {
            "name": obj.get("@name", obj.get("@id")),
            "codes": {c.get("@code"): c.get("@name", "") for c in as_list(obj.get("CLASS"))},
        }
    return result


def fetch_values(stats_data_id: str, options: dict[str, str]) -> tuple[list[dict], dict]:
    """All VALUE records for the table and filters (following NEXT_KEY), and the class names."""
    params = {k: v for k, v in options.items() if k.lower() not in _RESERVED}
    params.update(statsDataId=stats_data_id, metaGetFlg="Y", cntGetFlg="N")
    values, classes, start = [], {}, None
    for _ in range(_MAX_PAGES):
        page_params = dict(params) if start is None else {**params, "startPosition": start}
        root = call("getStatsData", page_params)
        data = root.get("STATISTICAL_DATA", {})
        if not classes:
            classes = class_names(data)
        values.extend(as_list(data.get("DATA_INF", {}).get("VALUE")))
        start = data.get("RESULT_INF", {}).get("NEXT_KEY")
        if not start:
            return values, classes
    raise DataValidationError("e-Stat returned too many pages; narrow the table with cd... options")


def values_to_series(values: list[dict], classes: dict) -> pd.DataFrame:
    """Convert VALUE records of ONE series to date/value (monthly only).

    Raises DataValidationError listing every dimension that still has more
    than one code, so the catalog can be narrowed with cd<Dimension>=<code>.
    """
    if not values:
        raise DataValidationError("e-Stat returned no data for these settings")
    frame = pd.DataFrame(values)
    dims = [c for c in frame.columns if c.startswith("@") and c not in ("@time", "@unit")]
    varying = [d for d in dims if frame[d].nunique() > 1]
    if varying:
        lines = []
        for dim in varying:
            key = dim[1:]
            info = classes.get(key, {"name": key, "codes": {}})
            codes = list(dict.fromkeys(frame[dim]))
            shown = ", ".join(f"{c}={info['codes'].get(c, '?')}" for c in codes[:8])
            more = f" ... (all {len(codes)})" if len(codes) > 8 else ""
            setting = "cd" + key[0].upper() + key[1:]
            lines.append(f"{setting}: {info['name']} has {len(codes)} codes: {shown}{more}")
        raise DataValidationError(
            "More than one series matches; add one of these to source_options -> " + " / ".join(lines)
        )
    time_names = classes.get("time", {}).get("codes", {})
    frame["date"] = [parse_month(code, time_names.get(code, "")) for code in frame["@time"]]
    monthly = frame.dropna(subset=["date"])
    if monthly.empty:  # no monthly rows: try quarterly (stored on the quarter's first month)
        frame["date"] = [parse_quarter(code, time_names.get(code, "")) for code in frame["@time"]]
        monthly = frame.dropna(subset=["date"])
    if monthly.empty:
        raise DataValidationError("No monthly or quarterly data in this table (annual only?)")
    numbers = pd.to_numeric(monthly["$"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    return pd.DataFrame({"date": monthly["date"].to_numpy(), "value": numbers.astype(float).to_numpy()})


class EStatSource(DataSource):
    """Series from e-Stat. ``source_id`` is the statsDataId; options narrow it down."""

    name = "estat"

    def fetch(self, source_id: str, options: dict[str, str]) -> pd.DataFrame:
        values, classes = fetch_values(source_id, options)
        return values_to_series(values, classes)
