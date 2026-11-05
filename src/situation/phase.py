"""Judge the current situation from several indicators and combine the judgments.

Each viewpoint (物価, 食料, 光熱, 賃金と物価, 雇用, 家計の支出, ...) is judged
from the latest change from a year earlier, using the thresholds below. They
are rules of thumb, not official criteria; edit ``THRESHOLDS`` to adjust.
Hokkaido series are used when available, otherwise the national ones.

The combination rules turn the judgments into short points to keep in mind
in consultations. They describe possibilities, never causes.
"""

import numpy as np
import pandas as pd

# (lower bound, label) from the highest band down; the first band whose lower
# bound is <= the value applies. Units: % for yoy_pct, points for yoy_diff.
THRESHOLDS = {
    "price": [(3.0, "高い上昇"), (2.0, "上昇"), (0.5, "緩やかな上昇"), (-0.5, "横ばい"), (-np.inf, "下落")],
    "food": [(5.0, "大幅な上昇"), (2.0, "上昇"), (-2.0, "横ばい"), (-np.inf, "下落")],
    "energy": [(10.0, "大幅な上昇"), (3.0, "上昇"), (-3.0, "横ばい"), (-np.inf, "下落")],
    "yen": [(5.0, "円安が進んでいる"), (-5.0, "大きな変化なし"), (-np.inf, "円高が進んでいる")],
    "real_wage": [(1.0, "収入が物価を上回っている"), (-1.0, "収入と物価がほぼ同じ"), (-np.inf, "収入が物価に追いついていない")],
    # cumulative gap over several years: small yearly shortfalls add up
    "cumulative": [(2.0, "収入が物価を上回っている"), (-2.0, "収入と物価がほぼ同じ"), (-np.inf, "収入が物価に追いついていない")],
    "jobs_ratio": [(0.05, "改善"), (-0.05, "横ばい"), (-np.inf, "悪化")],
    "unemployment": [(0.2, "悪化"), (-0.2, "横ばい"), (-np.inf, "改善")],
    "spending": [(2.0, "増加"), (-2.0, "横ばい"), (-np.inf, "減少")],
    "sentiment": [(2.0, "改善"), (-2.0, "横ばい"), (-np.inf, "悪化")],
}

# viewpoint, threshold set, candidate series (first one available is used)
VIEWPOINTS = [
    ("物価全体", "price", ["CPI_HKD_CORE", "CPI_JP_CORE"]),
    ("食料", "food", ["CPI_HKD_FOOD", "CPI_JP_FOOD"]),
    ("医療費（保健医療）", "price", ["CPI_HKD_HEALTH"]),
    ("たばこ", "price", ["CPI_HKD_TOBACCO"]),
    ("酒類", "price", ["CPI_HKD_ALCOHOL"]),
    ("灯油など（他の光熱）", "energy", ["CPI_HKD_OTHER_FUEL", "CPI_JP_KEROSENE"]),
    ("灯油の店頭価格", "energy", ["KEROSENE_PRICE_HKD"]),
    ("電気代", "energy", ["CPI_HKD_ELEC", "CPI_JP_ELEC"]),
    ("為替", "yen", ["USDJPY"]),
    ("円建ての原油価格", "energy", ["OIL_DUBAI_JPY"]),
    ("有効求人倍率", "jobs_ratio", ["HKD_JOBS_RATIO", "JOBS_RATIO_JP"]),
    ("完全失業率", "unemployment", ["UNEMP_JP"]),
    ("家計の支出", "spending", ["HH_CONSUMPTION_WORKER_HKD", "HH_SPEND_JP"]),
    ("家計の支出（収入の低い世帯）", "spending", ["HH_CONSUMPTION_Q1_JP"]),
    ("家計の支出（単身の勤労者世帯・北海道東北）", "spending", ["SINGLE_WORKER_CONSUMPTION_HT"]),
    ("家計の支出（単身世帯全体・北海道東北）", "spending", ["SINGLE_CONSUMPTION_HT"]),
    ("暮らし向きの見通し", "sentiment", ["CONSUMER_CONF_JP"]),
]
RISING = {"高い上昇", "上昇", "大幅な上昇"}
# The wage series covers manufacturing employees only; the label says so.
WAGE_ASPECT = "賃金（製造業）と物価"
INCOME_ASPECT = "手取り収入（勤労者世帯の平均）と物価"
LOW_INCOME_ASPECT = "手取り収入（収入の低い勤労者世帯）と物価"
PENSION_ASPECT = "年金と物価"
PENSION_CUMULATIVE_ASPECT = "年金と物価（2021年度からの累積）"
MIN_WAGE_CUMULATIVE_ASPECT = "最低賃金と物価（2021年10月からの累積）"
SINGLE_ASPECT = "手取り収入（単身の勤労者世帯・北海道東北）と物価"
MIN_WAGE_ASPECT = "最低賃金と物価"
# aspect -> (take-home income, prices) candidates, first available used
INCOME_VIEWS = [
    (INCOME_ASPECT, [("HH_DISPOSABLE_HKD", "CPI_HKD_ALL"), ("HH_DISPOSABLE_JP", "CPI_JP_ALL")]),
    # lowest 20% of working households by annual income (national only)
    (LOW_INCOME_ASPECT, [("HH_DISPOSABLE_Q1_JP", "CPI_JP_ALL")]),
    # outside the household survey's working households
    (PENSION_ASPECT, [("PENSION_INDEX_JP", "CPI_HKD_ALL"), ("PENSION_INDEX_JP", "CPI_JP_ALL")]),
    (MIN_WAGE_ASPECT, [("HKD_MIN_WAGE", "CPI_HKD_ALL"), ("HKD_MIN_WAGE", "CPI_JP_ALL")]),
    # one-person working households, Hokkaido and Tohoku combined (quarterly)
    (SINGLE_ASPECT, [("SINGLE_WORKER_DISPOSABLE_HT", "CPI_HKD_ALL"), ("SINGLE_WORKER_DISPOSABLE_HT", "CPI_JP_ALL")]),
]
# aspect -> (income series, price candidates): change since the income series' first month
CUMULATIVE_VIEWS = [
    (PENSION_CUMULATIVE_ASPECT, "PENSION_INDEX_JP", ["CPI_HKD_ALL", "CPI_JP_ALL"]),
    (MIN_WAGE_CUMULATIVE_ASPECT, "HKD_MIN_WAGE", ["CPI_HKD_ALL", "CPI_JP_ALL"]),
]


def classify(value: float, kind: str) -> str:
    if pd.isna(value):
        return "判定できない"
    for lower, label in THRESHOLDS[kind]:
        if value >= lower:
            return label
    return THRESHOLDS[kind][-1][1]


def _is_quarterly(dates: pd.Series) -> bool:
    """True when the dates are about three months apart (a quarterly series)."""
    if len(dates) < 3:
        return False
    periods = pd.PeriodIndex(pd.to_datetime(dates), freq="M").asi8
    return bool(np.median(np.diff(np.sort(periods))) >= 3)


def _price_change_for_quarters(prices: pd.DataFrame, quarter_dates: pd.Series) -> pd.Series:
    """Average monthly price change over the three months of each quarter."""
    lookup = dict(zip(prices["date"], prices["change"]))
    out = []
    for d in quarter_dates:
        start = pd.Period(d, freq="M")
        values = [lookup.get((start + k).to_timestamp().strftime("%Y-%m-%d")) for k in range(3)]
        values = [v for v in values if v is not None and pd.notna(v)]
        out.append(float(np.mean(values)) if len(values) == 3 else np.nan)
    return pd.Series(out, index=quarter_dates.index)


def _latest_change(changes: pd.DataFrame, series_id: str, months: int = 1) -> tuple[str | None, float]:
    one = changes[changes["series_id"] == series_id].dropna(subset=["change"])
    if one.empty:
        return None, np.nan
    tail = one.tail(months)
    return one["date"].iloc[-1], float(tail["change"].mean())


def judge(changes: pd.DataFrame, names: dict[str, str]) -> pd.DataFrame:
    """One row per viewpoint: the series used, its latest change and the judgment."""
    rows = []
    for aspect, kind, candidates in VIEWPOINTS:
        series_id = next((s for s in candidates if not changes[(changes["series_id"] == s) & changes["change"].notna()].empty), None)
        if series_id is None:
            continue
        quarterly = _is_quarterly(changes[changes["series_id"] == series_id]["date"])
        # monthly spending is noisy: use a 3-month average (a quarterly series already is one)
        months = 3 if kind == "spending" and not quarterly else 1
        date, value = _latest_change(changes, series_id, months)
        unit = "ポイント" if changes[changes["series_id"] == series_id]["compare"].iloc[0] == "yoy_diff" else "%"
        rows.append({"aspect": aspect, "series_id": series_id, "name": names.get(series_id, series_id),
                     "date": date, "change": value, "change_unit": unit,
                     "basis": "最新の四半期" if quarterly else ("直近3か月の平均" if months == 3 else "最新月"),
                     "judgment": classify(value, kind)})

    # 賃金と物価: nominal wage growth minus consumer price growth, same month
    wage = changes[changes["series_id"] == "WAGE_MFG_JP"].dropna(subset=["change"])
    prices = changes[changes["series_id"] == "CPI_JP_ALL"].dropna(subset=["change"])
    both = wage.merge(prices, on="date", suffixes=("_w", "_p"))
    if not both.empty:
        last = both.iloc[-1]
        gap = float(last["change_w"] - last["change_p"])
        rows.append({"aspect": WAGE_ASPECT, "series_id": "WAGE_MFG_JP-CPI_JP_ALL",
                     "name": "賃金（製造業・名目）の前年比 − 物価（総合）の前年比。非正規・パート・年金の世帯は含まない", "date": last["date"],
                     "change": gap, "change_unit": "ポイント", "basis": "最新月（両方そろう月）",
                     "judgment": classify(gap, "real_wage")})

    # 手取り収入と物価: household disposable income growth minus price growth,
    # 3-month averages over the latest months where both exist (survey data are noisy)
    for aspect, pairs in INCOME_VIEWS:
        for income_id, price_id in pairs:
            income = changes[changes["series_id"] == income_id].dropna(subset=["change"])
            prices = changes[changes["series_id"] == price_id].dropna(subset=["change"])
            if income.empty or prices.empty:
                continue
            if _is_quarterly(income["date"]):
                # quarterly income vs the average price change over the same three months
                q = income.assign(change_p=_price_change_for_quarters(prices, income["date"])).dropna(subset=["change_p"])
                if q.empty:
                    continue
                last = q.iloc[-1]
                gap, last_date, basis = float(last["change"] - last["change_p"]), last["date"], "最新の四半期"
            else:
                both = income.merge(prices, on="date", suffixes=("_i", "_p")).tail(3)
                if len(both) < 3:
                    continue
                gap = float(both["change_i"].mean() - both["change_p"].mean())
                last_date, basis = both["date"].iloc[-1], "直近3か月の平均"
            rows.append({"aspect": aspect, "series_id": f"{income_id}-{price_id}",
                         "name": f"{names.get(income_id, income_id)}の前年比 − {names.get(price_id, price_id)}の前年比",
                         "date": last_date, "change": gap, "change_unit": "ポイント",
                         "basis": basis, "judgment": classify(gap, "real_wage")})
            break
    rows.extend(_cumulative_rows(changes, names))
    return pd.DataFrame(rows, columns=["aspect", "series_id", "name", "date", "change", "change_unit", "basis", "judgment"])


def _cumulative_rows(changes: pd.DataFrame, names: dict[str, str]) -> list[dict]:
    """Income vs prices over the whole period since the income series starts.

    Compares the rise of each level from the first month of the income series
    to the latest month both series have. Small yearly shortfalls that stay
    inside the one-year band add up here.
    """
    rows = []
    for aspect, income_id, price_ids in CUMULATIVE_VIEWS:
        income = changes[changes["series_id"] == income_id].dropna(subset=["value"]).set_index("date")["value"]
        if income.empty:
            continue
        start = income.index.min()
        for price_id in price_ids:
            prices = changes[changes["series_id"] == price_id].dropna(subset=["value"]).set_index("date")["value"]
            common = sorted(set(income.index) & set(prices.index))
            if start not in prices.index or len(common) < 13:
                continue
            last = common[-1]
            income_rise = (income[last] / income[start] - 1) * 100
            price_rise = (prices[last] / prices[start] - 1) * 100
            gap = float(income_rise - price_rise)
            rows.append({"aspect": aspect, "series_id": f"{income_id}-{price_id}",
                         "name": (f"{names.get(income_id, income_id)}の上がり幅（{income_rise:+.1f}%） − "
                                  f"{names.get(price_id, price_id)}の上がり幅（{price_rise:+.1f}%）"),
                         "date": last, "change": gap, "change_unit": "ポイント",
                         "basis": f"{start[:7]}から{last[:7]}まで", "judgment": classify(gap, "cumulative")})
            break
    return rows


def combine(judgments: pd.DataFrame) -> list[str]:
    """Points to keep in mind, from combinations of the judgments."""
    j = dict(zip(judgments["aspect"], judgments["judgment"]))
    points = []
    price_up = j.get("物価全体") in RISING
    energy_up = any(j.get(k) in RISING for k in ("灯油など（他の光熱）", "灯油の店頭価格", "電気代"))
    # take-home income of working households is closer to consultation clients than
    # the manufacturing wage, so it is used whenever it is available
    # Prefer the low-income households, closest to consultation clients, then the average.
    income = j.get(LOW_INCOME_ASPECT, j.get(INCOME_ASPECT, j.get(WAGE_ASPECT)))
    if price_up and "収入が物価に追いついていない" in (income, j.get(SINGLE_ASPECT)):
        points.append("物価の上昇に収入が追いついておらず、実質的な収入が目減りしている可能性があります。"
                      "働いている世帯でも家計が苦しくなりやすい状況です。")
    if "収入が物価に追いついていない" in (j.get(PENSION_ASPECT), j.get(PENSION_CUMULATIVE_ASPECT)):
        points.append("年金の伸びが物価の上昇に追いついておらず、年金で暮らす世帯では実質的な目減りが続いている可能性があります。"
                      "1年ごとの差は小さくても、数年で積み重なります。本人の努力では補いにくい部分です。")
    if j.get(MIN_WAGE_ASPECT) == "収入が物価を上回っている":
        points.append("最低賃金は物価以上に上がっています。ただし、働ける時間や仕事の安定は別に確認が必要です。")
    if energy_up:
        points.append("光熱費（灯油・電気など）の負担が前年より重くなっています。"
                      "光熱費の増加は、使い方よりも値上がりによる部分が大きい可能性があります。")
    rising_personal = [a for a in ("医療費（保健医療）", "たばこ", "酒類") if j.get(a) in RISING]
    if rising_personal:
        points.append(f"{'・'.join(rising_personal)}の値段が上がっています。{'これらの' if len(rising_personal) > 1 else 'その'}支出の増加には、使う量だけでなく"
                      f"値上がりによる部分が含まれている可能性があります。")
    if j.get("食料") in RISING:
        points.append("食料品の値上がりが続いています。食費の増加は、本人の工夫だけでは抑えにくい部分です。")
    if price_up and "減少" in (j.get("家計の支出"), j.get("家計の支出（収入の低い世帯）"), j.get("家計の支出（単身の勤労者世帯・北海道東北）"), j.get("家計の支出（単身世帯全体・北海道東北）")):
        points.append("物価が上がる中で家計の支出が減っており、生活を切り詰めている世帯が増えている可能性があります。")
    if "悪化" in (j.get("有効求人倍率"), j.get("完全失業率")):
        points.append("雇用の状況が悪くなりつつあります。就労の支援では、求人の状況も踏まえた計画が必要です。")
    if j.get("為替") == "円安が進んでいる" and j.get("円建ての原油価格") in RISING:
        points.append("円安と原油高が重なり、輸入するエネルギーや食料の値上がりが、これから暮らしに届く可能性があります。")
    if j.get("暮らし向きの見通し") == "悪化":
        points.append("人々の暮らし向きの見通しが悪くなっています。")
    if not points:
        points.append("目安の範囲では、大きな変化の組み合わせは見られません。")
    return points
