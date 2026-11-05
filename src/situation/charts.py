"""Static charts (SVG) for the reports and the public site.

Each chart is drawn from the same tables the report uses (``changes`` and
the timeline), so the pictures and the numbers always agree. Text in the
SVG files stays as text, so the browser draws the Japanese with its own font.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.situation.events import _missing  # noqa: E402

INK, GRID = "#1f2a36", "#d8dde3"
BLUE, RED, GREEN, GOLD = "#2f6690", "#b5462f", "#3d7a4a", "#b8860b"
BAND = "#9fb7c9"
YEARS_SHOWN = 10

JAPANESE_FONTS = ["Yu Gothic", "Meiryo", "BIZ UDGothic", "MS Gothic", "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic"]


def _japanese_font() -> list[str]:
    """The first Japanese font installed on this computer (avoids font warnings)."""
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    return [name for name in JAPANESE_FONTS if name in installed][:1] + ["sans-serif"]


plt.rcParams.update({
    "font.family": _japanese_font(),
    "svg.fonttype": "none",
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "figure.dpi": 100,
})

# file name, title shown in the report, one-line caption
CHARTS = {
    "income_gradation": ("勤労者世帯の年収階級別：手取りと物価（全国、2021年10月＝100）",
                         "勤労者世帯を年収の低い順に5つに分け、手取り（可処分所得）の12か月平均を物価と並べたもの。"
                         "物価の線より下にある階級は、手取りが物価に追いついていない。"
                         "かっこ内は各階級の平均の年収（税・社会保険料を引く前）と手取り（月額）。どちらも直近12か月の平均。"),
    "income_gap": ("所得の層ごとの収入と物価の伸びの差（2021年10月〜）",
                   "棒が左（マイナス）にある階級ほど、手取りの伸びが物価に届いていない。どの所得の層まで物価高の影響が及んでいるかを見る。"
                   "低所得は、毎月の家計の統計がないため、主な収入源である最低賃金（フルタイム換算）と基礎年金の伸びで示す。"
                   "中所得・高所得は二人以上の勤労者世帯の年収5階級で、かっこ内は各階級の平均の年収と手取り（月額）。"),
    "oil_kerosene": ("円建ての原油価格と北海道の灯油など（前年比）",
                     "世界の原油価格の動きが、おおむね2か月遅れて北海道の灯油などの物価に届く様子。"),
    "pension_prices": ("年金と北海道の物価（2021年4月＝100）",
                       "年金の改定は物価の上がり方に追いつかず、差が積み重なっている。"),
    "fuel_subsidies": ("北海道の灯油などの物価と、燃料の補助の期間",
                       "帯は国の燃料補助の期間。補助がある間は物価が抑えられている。"),
    "electricity": ("北海道の電気代の物価と、補助・料金改定",
                    "帯は国の電気・ガス料金の値引き、縦線は北海道電力の料金改定。"),
    "yoy_heatmap": ("主な指標の前年比の推移（直近36か月）",
                    "赤いほど前年より上昇、青いほど低下。どの時期に何が重なったかを見る。"),
    "lead_lag": ("世界の動きが暮らしに届くまでの時間差",
                 "上流の前年比をずらしたとき、下流と最もよく重なったずれと相関（過去20年）。"),
}


def _series(changes: pd.DataFrame, series_id: str, column: str = "value") -> pd.Series:
    rows = changes[changes["series_id"] == series_id].dropna(subset=[column])
    if rows.empty:
        return pd.Series(dtype=float)
    index = pd.to_datetime(rows["date"])
    return pd.Series(rows[column].to_numpy(dtype=float), index=index).sort_index()


def _recent(series: pd.Series, years: int = YEARS_SHOWN) -> pd.Series:
    if series.empty:
        return series
    return series[series.index >= series.index.max() - pd.DateOffset(years=years)]


def _bands(ax, events: pd.DataFrame, series_ids: set[str], start, end, label: str) -> None:
    """Shade subsidy periods of events linked to these series (overlapping programs merged)."""
    if events is None or events.empty:
        return
    spans = []
    for _, e in events.iterrows():
        if "補助" not in e["kind"] or not set(e["series"]) & series_ids:
            continue
        left = max(e["start"].to_timestamp(), pd.Timestamp(start))
        right = min(pd.Timestamp(end) if _missing(e["end"]) else (e["end"] + 1).to_timestamp(), pd.Timestamp(end))
        if left < right:
            spans.append([left, right])
    merged = []
    for left, right in sorted(spans):
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    for i, (left, right) in enumerate(merged):
        ax.axvspan(left, right, color=BAND, alpha=0.35, lw=0, label=label if i == 0 else None)


def _event_lines(ax, events: pd.DataFrame, series_ids: set[str], start, end, keyword: str, label: str) -> None:
    if events is None or events.empty:
        return
    first = True
    for _, e in events.iterrows():
        if "補助" in e["kind"] or keyword not in e["title"] or not set(e["series"]) & series_ids:
            continue
        when = e["start"].to_timestamp()
        if pd.Timestamp(start) <= when <= pd.Timestamp(end):
            ax.axvline(when, color=RED, lw=1, ls="--", label=label if first else None)
            first = False


def _title(fig, text: str) -> None:
    """Title at the left edge of the whole figure, so long axis labels never push it off the page."""
    fig.suptitle(text, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)


def _legend(ax, handles=None, labels=None) -> None:
    """Legend under the plot, so it never covers the lines."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=3, fontsize=8)


def _finish(fig, ax, path: Path, source: str, bottom: float = 0.04) -> Path:
    fig.text(0.01, 0.01, f"出典：{source}", fontsize=8, color="#5a6570")
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


def chart_oil_kerosene(changes, path: Path):
    oil = _recent(_series(changes, "OIL_DUBAI_JPY", "change"))
    fuel = _recent(_series(changes, "CPI_HKD_OTHER_FUEL", "change"))
    if oil.empty or fuel.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _title(fig, CHARTS["oil_kerosene"][0])
    ax.plot(fuel.index, fuel.values, color=RED, lw=1.8, label="北海道 他の光熱（灯油など）（左目盛）")
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_ylabel("前年比（%）")
    ax2 = ax.twinx()
    ax2.plot(oil.index, oil.values, color=BLUE, lw=1.2, alpha=0.8, label="円建ての原油価格 ドバイ（右目盛）")
    ax2.set_ylabel("前年比（%）")
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    lines = ax.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    _legend(ax, lines[0] + lines2[0], lines[1] + lines2[1])
    return _finish(fig, ax, path, "総務省「消費者物価指数」、IMF・FRED（原油価格、為替）")


def chart_pension_prices(changes, events, path: Path):
    pension = _series(changes, "PENSION_INDEX_JP")
    prices = _series(changes, "CPI_HKD_ALL")
    base = pd.Timestamp("2021-04-01")
    if pension.empty or prices.empty or base not in pension.index or base not in prices.index:
        return None
    pension = pension[pension.index >= base] / pension[base] * 100
    prices = prices[prices.index >= base] / prices[base] * 100
    common = pension.index.intersection(prices.index)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _title(fig, CHARTS["pension_prices"][0])
    ax.plot(prices.index, prices.values, color=RED, lw=1.8, label="北海道 消費者物価指数 総合")
    ax.step(pension.index, pension.values, where="post", color=BLUE, lw=1.8, label="年金額（基礎年金）")
    ax.fill_between(common, pension[common].values, prices[common].values,
                    where=prices[common].values > pension[common].values, color=RED, alpha=0.12, lw=0,
                    label="物価が年金を上回った部分")
    if len(common):
        last = common[-1]
        gap = pension[last] - prices[last]
        ax.annotate(f"{last:%Y年%m月}\n差 {gap:+.1f}", xy=(last, (pension[last] + prices[last]) / 2),
                    xytext=(8, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
    ax.set_ylabel("2021年4月＝100")
    _legend(ax)
    return _finish(fig, ax, path, "総務省「消費者物価指数」、厚生労働省（年金額の改定率）")


QUINTILES = [("HH_DISPOSABLE_Q1_JP", "Ⅰ"), ("HH_DISPOSABLE_Q2_JP", "Ⅱ"), ("HH_DISPOSABLE_Q3_JP", "Ⅲ"),
             ("HH_DISPOSABLE_Q4_JP", "Ⅳ"), ("HH_DISPOSABLE_Q5_JP", "Ⅴ")]
QUINTILE_COLORS = ["#8c2d04", "#cc4c02", "#ec7014", "#7a9cbf", "#2f6690"]
INCOME_BASE = pd.Timestamp("2021-10-01")


def _indexed_12m(changes, series_id: str) -> pd.Series:
    """12-month average, indexed to INCOME_BASE = 100 (empty if the base month is missing)."""
    s = _series(changes, series_id).rolling(12).mean().dropna()
    if s.empty or INCOME_BASE not in s.index:
        return pd.Series(dtype=float)
    return s[s.index >= INCOME_BASE] / s[INCOME_BASE] * 100


def _monthly_level(changes, series_id: str) -> float | None:
    """Latest 12-month average of the monthly amount (yen)."""
    s = _series(changes, series_id).rolling(12).mean().dropna()
    return float(s.iloc[-1]) if not s.empty else None


def income_class(annual: float | None, ref: dict[str, str]) -> str:
    """低所得／中所得／高所得 (目安): 低所得 up to the common 300万円 line,
    中所得 up to twice the median, 高所得 above."""
    try:
        low = float(ref["low_income_line"])
        high = float(ref["median"]) * float(ref.get("middle_high_ratio", 2.0))
    except (KeyError, ValueError):
        return ""
    if annual is None:
        return ""
    if annual <= low:
        return "低所得"
    return "中所得" if annual <= high else "高所得"


def quintile_label(changes, series_id: str, label: str) -> str:
    """e.g. 'Ⅰ 中所得（年収 約415万円・手取り 月約32万円）'."""
    parts = []
    annual = _monthly_level(changes, series_id.replace("HH_DISPOSABLE_", "HH_ANNUAL_INCOME_"))
    if annual is not None and annual < 10:
        annual = None  # not a 万円 amount (test data)
    klass = income_class(annual, income_reference())
    if annual:
        parts.append(f"年収 約{annual:.0f}万円")
    level = _monthly_level(changes, series_id)
    if level and level >= 10000:  # yen (skips test data that is not a yen amount)
        parts.append(f"手取り 月約{level / 10000:.0f}万円")
    head = f"{label} {klass}" if klass else label
    return f"{head}（{'・'.join(parts)}）" if parts else head


def _indexed_point(changes, series_id: str) -> pd.Series:
    """Monthly values indexed to INCOME_BASE = 100 (no averaging)."""
    s = _series(changes, series_id).dropna()
    if s.empty or INCOME_BASE not in s.index:
        return pd.Series(dtype=float)
    return s[s.index >= INCOME_BASE] / s[INCOME_BASE] * 100


def income_gaps(changes) -> pd.DataFrame:
    """Per quintile: rise of take-home pay and of prices (both 12-month averages) since INCOME_BASE."""
    prices = _indexed_12m(changes, "CPI_JP_ALL")
    rows = []
    for (sid, label), color in zip(QUINTILES, QUINTILE_COLORS):
        income = _indexed_12m(changes, sid)
        common = income.index.intersection(prices.index)
        if common.empty:
            continue
        last = common[-1]
        rows.append({"series_id": sid, "label": quintile_label(changes, sid, label), "color": color, "date": last,
                     "income_rise": income[last] - 100, "price_rise": prices[last] - 100,
                     "gap": income[last] - prices[last]})
    return pd.DataFrame(rows)


def chart_income_gradation(changes, path: Path):
    prices = _indexed_12m(changes, "CPI_JP_ALL")
    lines = [(sid, quintile_label(changes, sid, label), color, _indexed_12m(changes, sid))
             for (sid, label), color in zip(QUINTILES, QUINTILE_COLORS)]
    lines = [line for line in lines if not line[3].empty]
    if prices.empty or len(lines) < 2:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.4))
    _title(fig, CHARTS["income_gradation"][0])
    ax.plot(prices.index, prices.values, color=INK, lw=2.4, ls="--", label="全国 消費者物価指数 総合（12か月平均）")
    for sid, label, color, s in lines:
        ax.plot(s.index, s.values, color=color, lw=1.6, label=f"手取り {label}")
        ax.annotate(f"{s.iloc[-1]:.1f}", xy=(s.index[-1], s.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=color, va="center")
    ax.axhline(100, color=INK, lw=0.6)
    ax.set_ylabel("2021年10月＝100（12か月平均）")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2, fontsize=8)
    return _finish(fig, ax, path, "総務省「家計調査」（勤労者世帯・年間収入五分位）、「消費者物価指数」")


REFERENCE_PATH = Path(__file__).resolve().parents[2] / "config" / "income_reference.csv"


def income_reference(path: Path = REFERENCE_PATH) -> dict[str, str]:
    """{key: value} from config/income_reference.csv (empty if missing)."""
    if not Path(path).exists():
        return {}
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    return dict(zip(frame["key"], frame["value"]))


LOW_COLOR = "#5c1a00"


def low_income_gaps(changes, ref: dict[str, str]) -> pd.DataFrame:
    """Low income has no household survey series, so its main income sources stand in:
    the Hokkaido minimum wage (full-time equivalent) and the basic pension, against prices."""
    rows = []
    for sid, price_id, label in (
        ("HKD_MIN_WAGE", "CPI_HKD_ALL", "最低賃金で働く人"),
        ("PENSION_INDEX_JP", "CPI_JP_ALL", "基礎年金だけで暮らす人"),
    ):
        # these change once a year (step series that start in 2021), so compare the latest month
        # with the base month directly instead of 12-month averages
        income, prices = _indexed_point(changes, sid), _indexed_point(changes, price_id)
        common = income.index.intersection(prices.index)
        if common.empty:
            continue
        last = common[-1]
        detail = ""
        try:
            if sid == "HKD_MIN_WAGE":
                wage = _series(changes, sid).dropna().iloc[-1]
                detail = f"（北海道 時給{wage:,.0f}円、フルタイムで年約{wage * float(ref['fulltime_hours']) / 10000:.0f}万円）"
            else:
                detail = f"（満額 年約{float(ref['basic_pension_monthly']) * 12 / 10000:.0f}万円）"
        except (KeyError, ValueError, IndexError):
            pass
        rows.append({"series_id": sid, "label": f"低所得 {label}{detail}", "color": LOW_COLOR, "date": last,
                     "income_rise": income[last] - 100, "price_rise": prices[last] - 100,
                     "gap": income[last] - prices[last], "income_word": "収入"})
    return pd.DataFrame(rows)


def chart_income_gap(changes, path: Path):
    ref = income_reference()
    gaps = income_gaps(changes)
    if len(gaps) < 2:
        return None
    gaps["income_word"] = "手取り"
    gaps = pd.concat([low_income_gaps(changes, ref), gaps], ignore_index=True)
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(gaps) + 2.4))
    _title(fig, CHARTS["income_gap"][0])
    y = np.arange(len(gaps))
    ax.barh(y, gaps["gap"], color=gaps["color"])
    span = max(abs(gaps["gap"]).max(), 1) * 1.6
    for i, r in zip(y, gaps.to_dict("records")):
        # the value next to the bar end, the breakdown in a column on the right
        ax.text(r["gap"] + (0.08 if r["gap"] >= 0 else -0.08) * span, i, f"{r['gap']:+.1f}",
                va="center", ha="left" if r["gap"] >= 0 else "right", fontsize=9, color=INK, fontweight="bold")
        ax.text(1.02, i, f"{r['income_word']}{r['income_rise']:+.1f}%\n物価{r['price_rise']:+.1f}%",
                transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8, color="#5a6570")
    ax.axvline(0, color=INK, lw=0.8)
    n_low = int((gaps["color"] == LOW_COLOR).sum())
    if n_low:
        ax.axhline(n_low - 0.5, color=INK, lw=0.6, ls=":")
    ax.set_yticks(y, gaps["label"])
    ax.invert_yaxis()
    ax.set_xlim(-span * 1.25, span * 1.25)
    ax.set_xlabel(f"ポイント（{gaps['date'].iloc[-1]:%Y年%m月}までの12か月平均で比較）")
    if ref:
        fig.text(0.01, 0.045,
                 f"所得の物差し（{ref['year']}年、国民生活基礎調査）：世帯所得の中央値 {ref['median']}万円、"
                 f"貧困線 {ref['poverty_line']}万円（等価可処分所得）、所得200万円未満の世帯 {ref['under_200']}%。\n"
                 f"区分の目安：低所得＝年収{ref.get('low_income_line', '300')}万円以下"
                 f"（住民税非課税の目安は単身{ref.get('nontax_single', '')}万・夫婦{ref.get('nontax_couple', '')}万・"
                 f"3人約{ref.get('nontax_3', '')}万・4人約{ref.get('nontax_4', '')}万円）、\n"
                 f"中所得＝中央値の{float(ref.get('middle_high_ratio', 2.0)):.0f}倍（約{float(ref['median']) * float(ref.get('middle_high_ratio', 2.0)):.0f}万円）まで、"
                 "高所得＝それより上。低所得の世帯の手取りは毎月の統計がないため、主な収入源（最低賃金・基礎年金）の伸びで示す。",
                 fontsize=8, color=INK)
    return _finish(fig, ax, path, "総務省「家計調査」「消費者物価指数」、厚生労働省「国民生活基礎調査」・年金額、北海道労働局（最低賃金）",
                   bottom=0.14 if ref else 0.04)


def chart_fuel_subsidies(changes, events, path: Path):
    fuel = _series(changes, "CPI_HKD_OTHER_FUEL")
    fuel = fuel[fuel.index >= pd.Timestamp("2019-01-01")]
    if fuel.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _title(fig, CHARTS["fuel_subsidies"][0])
    _bands(ax, events, {"CPI_HKD_OTHER_FUEL", "KEROSENE_PRICE_HKD"}, fuel.index.min(), fuel.index.max(), "国の燃料補助の期間")
    ax.plot(fuel.index, fuel.values, color=RED, lw=1.8, label="北海道 他の光熱（灯油など）")
    ax.set_ylabel("指数（2025年＝100）")
    _legend(ax)
    return _finish(fig, ax, path, "総務省「消費者物価指数」、資源エネルギー庁（補助の期間）")


def chart_electricity(changes, events, path: Path):
    elec = _series(changes, "CPI_HKD_ELEC")
    elec = elec[elec.index >= pd.Timestamp("2019-01-01")]
    if elec.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _title(fig, CHARTS["electricity"][0])
    _bands(ax, events, {"CPI_HKD_ELEC"}, elec.index.min(), elec.index.max(), "国の電気・ガス料金の値引き")
    _event_lines(ax, events, {"CPI_HKD_ELEC"}, elec.index.min(), elec.index.max(), "北海道電力", "北海道電力の料金改定")
    ax.plot(elec.index, elec.values, color=GOLD, lw=1.8, label="北海道 電気代")
    ax.set_ylabel("指数（2025年＝100）")
    _legend(ax)
    return _finish(fig, ax, path, "総務省「消費者物価指数」、資源エネルギー庁・北海道電力（料金）")


HEATMAP_ROWS = [
    ("CPI_HKD_CORE", "物価（生鮮除く）"), ("CPI_HKD_FOOD", "食料"), ("CPI_HKD_OTHER_FUEL", "灯油など"),
    ("CPI_HKD_ELEC", "電気代"), ("CPI_HKD_HEALTH", "保健医療"), ("CPI_HKD_TOBACCO", "たばこ"),
    ("HKD_MIN_WAGE", "最低賃金"), ("PENSION_INDEX_JP", "年金額"), ("USDJPY", "為替（円安＝＋）"),
    ("OIL_DUBAI_JPY", "円建て原油"),
]


def chart_yoy_heatmap(changes, path: Path):
    rows, labels = [], []
    end = pd.to_datetime(changes["date"]).max()
    months = pd.date_range(end - pd.DateOffset(months=35), end, freq="MS")
    for sid, label in HEATMAP_ROWS:
        s = _series(changes, sid, "change")
        if s.empty:
            continue
        rows.append(s.reindex(months).to_numpy())
        labels.append(label)
    if not rows:
        return None
    data = np.array(rows, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.4))
    _title(fig, CHARTS["yoy_heatmap"][0])
    limit = 10.0
    image = ax.imshow(np.clip(data, -limit, limit), aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_yticks(range(len(labels)), labels)
    ticks = [i for i, m in enumerate(months) if m.month in (1, 7)]
    ax.set_xticks(ticks, [f"{months[i]:%Y-%m}" for i in ticks], fontsize=8)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01)
    bar.set_label("前年比（%、±10で頭打ち）", fontsize=8)
    return _finish(fig, ax, path, "総務省、厚生労働省、北海道労働局、FRED ほか（前年同月比）")


def chart_lead_lag(lags: pd.DataFrame, path: Path):
    ok = lags[lags["status"] == "ok"] if not lags.empty else lags
    if ok.empty:
        return None
    labels = [f"{u} → {d}" for u, d in zip(ok["upstream_name"], ok["downstream_name"])]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(ok) + 1.3))
    _title(fig, CHARTS["lead_lag"][0])
    y = np.arange(len(ok))
    ax.barh(y, ok["correlation"], color=BLUE, alpha=0.8)
    for i, (corr, lag) in enumerate(zip(ok["correlation"], ok["best_lag_months"])):
        ax.text(corr + 0.01, i, f"{int(lag)}か月（相関 {corr:.2f}）", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("相関（1に近いほどよく重なる）")
    return _finish(fig, ax, path, "本サイトの計算（前年比どうしの相関。原因を示すものではない）")


def draw_all(changes: pd.DataFrame, events: pd.DataFrame, lags: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    """Draw every chart that has data. Returns {name: path}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    drawn = {
        "income_gradation": chart_income_gradation(changes, out_dir / "income_gradation.svg"),
        "income_gap": chart_income_gap(changes, out_dir / "income_gap.svg"),
        "oil_kerosene": chart_oil_kerosene(changes, out_dir / "oil_kerosene.svg"),
        "pension_prices": chart_pension_prices(changes, events, out_dir / "pension_prices.svg"),
        "fuel_subsidies": chart_fuel_subsidies(changes, events, out_dir / "fuel_subsidies.svg"),
        "electricity": chart_electricity(changes, events, out_dir / "electricity.svg"),
        "yoy_heatmap": chart_yoy_heatmap(changes, out_dir / "yoy_heatmap.svg"),
        "lead_lag": chart_lead_lag(lags, out_dir / "lead_lag.svg"),
    }
    return {name: p for name, p in drawn.items() if p is not None}


def charts_section(drawn: dict[str, Path], relative_dir: str) -> list[str]:
    """Markdown lines that embed the charts (paths relative to the report)."""
    if not drawn:
        return []
    lines = ["", "## グラフ", ""]
    for name, (title, caption) in CHARTS.items():
        if name in drawn:
            lines += [f"### {title}", "", f"![{title}]({relative_dir}/{name}.svg)", "", caption, ""]
    return lines
