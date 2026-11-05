"""How long does a change upstream take to reach prices downstream?

For each pair (e.g. crude oil in yen -> kerosene prices) the year-on-year
changes are compared with the upstream series shifted by 0..24 months. The
shift with the highest correlation is the estimated time lag. Changes already
seen upstream but not yet reflected downstream are then used, with a simple
straight-line fit, to show the direction the downstream series may take.

If the best lag equals the upper limit (``at_limit``), the real lag may be
longer; treat it as unreliable.

Caution: a correlation is not a cause. Year-on-year changes move slowly, so
the peak can be broad; read the lag as a rough guide.
"""

import numpy as np
import pandas as pd

from src.catalog import SeriesSpec

# Combined series: world prices converted to yen (what Japan actually pays)
DERIVED = {
    "OIL_DUBAI_JPY": ("OIL_DUBAI", "USDJPY", "円建ての原油価格（ドバイ）", "円/バレル"),
    "LNG_ASIA_JPY": ("LNG_ASIA", "USDJPY", "円建てのLNG価格（アジア）", "円/百万BTU"),
    "FOOD_GLOBAL_JPY": ("FOOD_GLOBAL", "USDJPY", "円建ての国際食料価格指数", "指数×円"),
}

# (upstream, downstream): the ripple chain from the world to Hokkaido
PAIRS = [
    ("OIL_DUBAI_JPY", "CPI_JP_KEROSENE"),
    ("OIL_DUBAI_JPY", "CPI_HKD_OTHER_FUEL"),
    ("OIL_DUBAI_JPY", "KEROSENE_PRICE_HKD"),
    ("LNG_ASIA_JPY", "CPI_JP_ELEC"),
    ("LNG_ASIA_JPY", "CPI_HKD_ELEC"),
    ("FOOD_GLOBAL_JPY", "CPI_JP_FOOD"),
    ("FOOD_GLOBAL_JPY", "CPI_HKD_FOOD"),
    ("USDJPY", "CPI_JP_CORE"),
]


def add_derived(observations: pd.DataFrame, specs: list[SeriesSpec]) -> tuple[pd.DataFrame, list[SeriesSpec]]:
    """Add the yen-converted world prices (dollar price x yen per dollar, same month)."""
    frames, new_specs = [], []
    for derived_id, (dollar_id, fx_id, name, unit) in DERIVED.items():
        dollar = observations[observations["series_id"] == dollar_id][["date", "value"]]
        fx = observations[observations["series_id"] == fx_id][["date", "value"]]
        both = dollar.merge(fx, on="date", suffixes=("_usd", "_fx")).dropna()
        if both.empty:
            continue
        frames.append(pd.DataFrame({"series_id": derived_id, "date": both["date"],
                                    "value": both["value_usd"] * both["value_fx"], "source": "derived",
                                    "retrieved_at": ""}))
        new_specs.append(SeriesSpec(derived_id, name, "derived", f"{dollar_id}*{fx_id}", unit=unit,
                                    layer="世界", category="合成", compare="yoy_pct"))
    if not frames:
        return observations, specs
    return pd.concat([observations, *frames], ignore_index=True), specs + new_specs


def _monthly(changes: pd.DataFrame, series_id: str) -> pd.Series:
    one = changes[changes["series_id"] == series_id].dropna(subset=["change"])
    return pd.Series(one["change"].to_numpy(float), index=pd.PeriodIndex(pd.to_datetime(one["date"]), freq="M"))


def lead_lag(changes: pd.DataFrame, upstream: str, downstream: str, max_lag: int = 24,
             window_months: int = 240, min_n: int = 60) -> dict:
    """Correlation for each lag, the best lag, and the outlook it implies.

    Uses the latest ``window_months`` months of the downstream series. Pairs
    with fewer than ``min_n`` common months are reported as too short.
    """
    up, down = _monthly(changes, upstream), _monthly(changes, downstream)
    result = {"upstream": upstream, "downstream": downstream}
    if up.empty or down.empty:
        return {**result, "status": "データなし"}
    down = down[down.index > down.index.max() - window_months]
    correlations = {}
    for lag in range(max_lag + 1):
        shifted = up.copy()
        shifted.index = shifted.index + lag  # upstream value of `lag` months earlier
        both = pd.concat([shifted.rename("up"), down.rename("down")], axis=1, join="inner")
        if len(both) >= min_n and both["up"].std() > 0 and both["down"].std() > 0:
            correlations[lag] = (float(both["up"].corr(both["down"])), len(both))
    if not correlations:
        return {**result, "status": f"期間が短い（共通の月が{min_n}か月未満）"}
    best_lag = max(correlations, key=lambda k: correlations[k][0])
    corr, n = correlations[best_lag]

    # straight-line fit down(t) = a + b * up(t - lag), then use upstream values already known
    shifted = up.copy()
    shifted.index = shifted.index + best_lag
    both = pd.concat([shifted.rename("up"), down.rename("down")], axis=1, join="inner")
    slope, intercept = np.polyfit(both["up"], both["down"], 1)
    last_down = down.index.max()
    outlook = [
        {"date": str(month.to_timestamp().date()), "expected_change": float(intercept + slope * shifted[month])}
        for month in shifted.index
        if month > last_down
    ]
    return {**result, "status": "ok", "best_lag": best_lag, "at_limit": best_lag == max_lag,
            "correlation": corr, "n": n,
            "slope": float(slope), "intercept": float(intercept),
            "down_latest_date": str(last_down.to_timestamp().date()), "down_latest_change": float(down.iloc[-1]),
            "up_latest_change": float(up.iloc[-1]), "outlook": outlook,
            "correlations": {k: v[0] for k, v in correlations.items()}}


def lead_lag_table(changes: pd.DataFrame, names: dict[str, str], pairs=PAIRS, **kwargs) -> pd.DataFrame:
    """One row per pair with the best lag, correlation and the outlook."""
    rows = []
    for upstream, downstream in pairs:
        r = lead_lag(changes, upstream, downstream, **kwargs)
        outlook = r.get("outlook", [])
        rows.append({
            "upstream": upstream, "upstream_name": names.get(upstream, upstream),
            "downstream": downstream, "downstream_name": names.get(downstream, downstream),
            "status": r["status"], "best_lag_months": r.get("best_lag"), "at_limit": bool(r.get("at_limit", False)),
            "correlation": r.get("correlation"),
            "n_months": r.get("n"), "down_latest_date": r.get("down_latest_date"),
            "down_latest_change": r.get("down_latest_change"), "up_latest_change": r.get("up_latest_change"),
            "outlook_until": outlook[-1]["date"] if outlook else None,
            "outlook_change": outlook[-1]["expected_change"] if outlook else None,
        })
    return pd.DataFrame(rows)
