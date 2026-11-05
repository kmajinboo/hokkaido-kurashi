"""Timeline of policies and events (制度・出来事の年表).

Three editable tables in ``config/`` (UTF-8 with BOM, opens in Excel):

- ``events.csv``        national and Hokkaido events (subsidies, tax and
                        pension changes, benefits, big economic events)
- ``events_local.csv``  events in the areas we cover (local points, vouchers)
- ``fukushi_touyu.csv`` 福祉灯油 by municipality and fiscal year

They are used three ways:

1. Notes next to judgments: a subsidy that lowers prices, or its end, moves
   the year-on-year change. ``note_for`` says so, so a calm price is not
   read as a calm market.
2. A report section listing what started or ended recently and what comes
   in the next three months.
3. In the heating season, what each municipality's 福祉灯油 is worth at
   today's kerosene price (and a year ago).
"""

from datetime import date
from pathlib import Path

import pandas as pd

RECENT_MONTHS = 12
ENDED_MONTHS = 6  # list ended events only this far back, to keep the table short
AHEAD_MONTHS = 3
HEATING_MONTHS = {10, 11, 12, 1, 2, 3, 4}
# kerosene price series by 振興局 (道北 covers 上川・留萌 in the monitor survey)
KEROSENE_BY_AREA = {"十勝": "KEROSENE_PRICE_TOKACHI", "上川": "KEROSENE_PRICE_DOHOKU", "留萌": "KEROSENE_PRICE_DOHOKU"}
KEROSENE_DEFAULT = "KEROSENE_PRICE_HKD"


def _read(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    frame.columns = [c.strip() for c in frame.columns]
    return frame


def _month(text: str) -> pd.Period | None:
    """'2026-03', '2026-3', '2026' (whole year -> January) or '' -> Period or None."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        if len(text) == 4 and text.isdigit():
            return pd.Period(f"{text}-01", freq="M")
        return pd.Period(text[:7].replace("/", "-"), freq="M")
    except (ValueError, TypeError):
        return None


def short_title(text: str) -> str:
    """The part of a description before the first bracket or colon."""
    for mark in ("（", "：", "("):
        if mark in text:
            text = text.split(mark)[0]
    return text.strip()


def load_events(config_dir: Path) -> pd.DataFrame:
    """National/Hokkaido and local events as one table.

    Columns: id, start, end (Period or None), area, kind, target, title,
    series (list of series_id), source, confirmed (bool), memo, scope.
    """
    rows = []
    for r in _read(Path(config_dir) / "events.csv").to_dict("records"):
        rows.append({"id": r.get("ID", ""), "start": _month(r.get("開始", "")), "end": _month(r.get("終了", "")),
                     "area": r.get("地域", ""), "kind": r.get("種類", ""), "target": r.get("対象", ""),
                     "title": r.get("内容", ""), "series": [s for s in r.get("関係する指標", "").split(";") if s],
                     "source": r.get("出典", ""), "confirmed": r.get("確認状況", "").startswith("出典あり"),
                     "memo": r.get("メモ", ""), "scope": "国・北海道"})
    for r in _read(Path(config_dir) / "events_local.csv").to_dict("records"):
        if not r.get("開始"):
            continue  # reference rows without a period
        amount = f"{int(float(r['金額'])):,}{r.get('単位', '')}" if r.get("金額", "").replace(".", "").isdigit() else ""
        title = f"{r.get('市町村', '')} {r.get('名称', '')}" + (f"（{amount}）" if amount else "")
        rows.append({"id": r.get("ID", ""), "start": _month(r.get("開始", "")), "end": _month(r.get("終了", "")),
                     "area": f"{r.get('振興局', '')}・{r.get('市町村', '')}", "kind": r.get("種類", ""),
                     "target": r.get("対象", ""), "title": title, "series": [],
                     "source": r.get("出典", ""), "confirmed": r.get("出典", "").startswith("http"),
                     "memo": r.get("メモ", ""), "scope": "担当地域"})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame[frame["start"].notna()].reset_index(drop=True)


def _missing(period) -> bool:
    return period is None or pd.isna(period)


def _active(event, month: pd.Period) -> bool:
    end = event["end"]
    return event["start"] <= month and (_missing(end) or month <= end)


def note_for(events: pd.DataFrame, series_ids: list[str], latest: str) -> str:
    """Events that move the year-on-year change of these series at the latest month.

    Subsidies are judged together: what matters is whether the latest month
    and the same month a year earlier had any subsidy, not which program.
    """
    if events.empty or not latest:
        return ""
    now = pd.Period(str(latest)[:7], freq="M")
    before = now - 12
    related = [e for _, e in events.iterrows()
               if set(e["series"]) & set(series_ids) and "給付" not in e["kind"]]  # benefits do not move prices
    label = lambda e: short_title(e["title"]) + ("" if e["confirmed"] else "（要確認）")  # noqa: E731
    subsidies = [e for e in related if "補助" in e["kind"]]
    sub_now = [e for e in subsidies if _active(e, now)]
    sub_before = [e for e in subsidies if _active(e, before)]
    notes = []
    if sub_now:
        names = "・".join(dict.fromkeys(label(e) for e in sub_now))
        since = min(e["start"] for e in sub_now).strftime("%Y-%m")
        if sub_before:
            notes.append(f"{names}で価格が抑えられている（{since}〜）。前年同月も補助あり（額は異なる場合がある）")
        else:
            notes.append(f"{names}で価格が抑えられている（{since}〜）。前年同月は補助なしのため、前年比は低めに出る")
    elif sub_before:
        names = "・".join(dict.fromkeys(label(e) for e in sub_before))
        ended = max(e["end"] for e in sub_before if not _missing(e["end"])).strftime("%Y-%m")
        notes.append(f"{names}が終わった（{ended}）ため、前年比は高めに出る")
    for e in related:
        if "補助" not in e["kind"] and before < e["start"] <= now:
            notes.append(f"{label(e)}（{e['start'].strftime('%Y-%m')}）を含む")
    return "。".join(dict.fromkeys(notes))


def add_notes(table: pd.DataFrame, events: pd.DataFrame, id_column: str = "series_id") -> pd.DataFrame:
    """Add a 'note' column to judgments or the latest-value table."""
    table = table.copy()
    table["note"] = [note_for(events, str(sid).split("-"), str(d)) for sid, d in zip(table[id_column], table["date"])]
    return table


def recent_and_upcoming(events: pd.DataFrame, today: date) -> pd.DataFrame:
    """Events that started or ended in the last 12 months, or start/end within 3 months."""
    if events.empty:
        return pd.DataFrame(columns=["when", "status", "scope", "area", "title", "target", "confirmed"])
    now = pd.Period(today, freq="M")
    rows = []
    for _, e in events.iterrows():
        start, end = e["start"], e["end"]
        if now < start <= now + AHEAD_MONTHS:
            status, when = "これから始まる", start
        elif not _missing(end) and now <= end <= now + AHEAD_MONTHS and start <= now:
            status, when = "実施中（まもなく終了）", end
        elif now - RECENT_MONTHS < start <= now and (_missing(end) or end >= now):
            status, when = "最近始まった・実施中", start
        elif not _missing(end) and now - ENDED_MONTHS < end < now:
            status, when = "最近終わった", end
        elif now - RECENT_MONTHS < start <= now:
            status, when = "最近あった", start
        else:
            continue
        rows.append({"when": when, "status": status, "scope": e["scope"], "area": e["area"],
                     "title": e["title"], "target": e["target"], "confirmed": e["confirmed"],
                     "period": f"{start.strftime('%Y-%m')}〜{'' if _missing(end) else end.strftime('%Y-%m')}"})
    order = {"これから始まる": 0, "実施中（まもなく終了）": 1, "最近始まった・実施中": 2, "最近あった": 3, "最近終わった": 4}
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["status", "when"], key=lambda s: s.map(order) if s.name == "status" else s).reset_index(drop=True)


def _fiscal_year(day: date) -> int:
    return day.year if day.month >= 4 else day.year - 1


def _latest_and_year_ago(changes: pd.DataFrame, series_id: str) -> tuple[str, float, float] | None:
    rows = changes[(changes["series_id"] == series_id)].dropna(subset=["value"]).sort_values("date")
    if rows.empty:
        return None
    last = rows.iloc[-1]
    ago_date = (pd.Period(str(last["date"])[:7], freq="M") - 12).strftime("%Y-%m-01")
    ago = rows[rows["date"] == ago_date]
    return str(last["date"])[:7], float(last["value"]), (float(ago["value"].iloc[0]) if not ago.empty else float("nan"))


def touyu_table(config_dir: Path, changes: pd.DataFrame, today: date) -> pd.DataFrame:
    """福祉灯油 that is (or was last) in effect, valued at today's and last year's kerosene price."""
    frame = _read(Path(config_dir) / "fukushi_touyu.csv")
    if frame.empty:
        return pd.DataFrame()
    fy = _fiscal_year(today)
    rows = []
    for r in frame.to_dict("records"):
        start = r.get("開始年度", "")
        if not start.isdigit() or int(start) > fy:
            continue
        end = r.get("終了年度", "")
        if end.isdigit() and int(end) < fy - 1:
            continue  # ended before last winter
        liters = pd.to_numeric(r.get("量_L", ""), errors="coerce")
        yen = pd.to_numeric(r.get("金額_円", ""), errors="coerce")
        if pd.isna(liters) and pd.isna(yen):
            continue
        price = _latest_and_year_ago(changes, KEROSENE_BY_AREA.get(r.get("振興局", ""), KEROSENE_DEFAULT)) \
            or _latest_and_year_ago(changes, KEROSENE_DEFAULT)
        if price is None:
            continue
        month, now_price, ago_price = price
        if not pd.isna(liters):
            given = f"灯油{liters:.0f}L"
            now_value, ago_value = f"約{liters * now_price:,.0f}円分", (f"約{liters * ago_price:,.0f}円分" if ago_price == ago_price else "-")
        else:
            given = f"{yen:,.0f}円"
            now_value, ago_value = f"約{yen / now_price:.0f}L分", (f"約{yen / ago_price:.0f}L分" if ago_price == ago_price else "-")
        latest_known = int(end) if end.isdigit() else max(int(start), 0)
        rows.append({"area": r.get("振興局", ""), "town": r.get("市町村", ""), "given": given,
                     "method": r.get("支給方法", ""), "price_month": month, "price": now_price,
                     "now_value": now_value, "year_ago_value": ago_value,
                     "known_year": f"{start}年度〜" + (f"{end}年度" if end else ""),
                     "this_year_unknown": not r.get("確認済み") and int(start) < fy})
    return pd.DataFrame(rows)


def events_section(events: pd.DataFrame, recent: pd.DataFrame, touyu: pd.DataFrame, today: date) -> list[str]:
    """Markdown lines for the report."""
    if events.empty and touyu.empty:
        return []
    lines = ["", "## 最近と、これからの制度・出来事", "",
             f"直近{RECENT_MONTHS}か月に始まったもの、{ENDED_MONTHS}か月以内に終わったもの、これから{AHEAD_MONTHS}か月の予定です。"
             "（要確認）は出典で確かめていないものです。年表は config/events.csv と events_local.csv で直せます。", ""]
    if recent.empty:
        lines.append("- （該当なし）")
    else:
        lines += ["| 状況 | 期間 | 範囲 | 内容 | 対象 |", "|---|---|---|---|---|"]
        for r in recent.itertuples():
            mark = "" if r.confirmed else "（要確認）"
            lines.append(f"| {r.status} | {r.period} | {r.area} | {r.title}{mark} | {r.target} |")
    if today.month in HEATING_MONTHS and not touyu.empty:
        lines += ["", "### 福祉灯油の目安（今の灯油価格で換算）", "",
                  "北海道消費生活モニターの灯油価格（配達、十勝は十勝、上川・留萌は道北）で換算しています。"
                  "現物（L）の支給は値上がりしても量が変わらず、現金の支給は値上がりの分だけ買える量が減ります。", "",
                  "| 振興局 | 市町村 | 支給 | 今の価格で | 1年前の価格で | 内容が分かっている年度 |", "|---|---|---|---|---|---|"]
        for r in touyu.itertuples():
            unknown = "（今年度は未確認）" if r.this_year_unknown else ""
            lines.append(f"| {r.area} | {r.town} | {r.given} | {r.now_value}（{r.price:.0f}円/L、{r.price_month}） | "
                         f"{r.year_ago_value} | {r.known_year}{unknown} |")
    return lines


def events_for_powerbi(events: pd.DataFrame) -> pd.DataFrame:
    """Flat table for chart annotations in Power BI."""
    if events.empty:
        return pd.DataFrame(columns=["id", "start", "end", "scope", "area", "kind", "target", "title", "series_ids", "confirmed"])
    return pd.DataFrame({
        "id": events["id"],
        "start": [p.strftime("%Y-%m-01") for p in events["start"]],
        "end": ["" if _missing(p) else p.strftime("%Y-%m-01") for p in events["end"]],
        "scope": events["scope"], "area": events["area"], "kind": events["kind"], "target": events["target"],
        "title": events["title"], "series_ids": [";".join(s) for s in events["series"]],
        "confirmed": events["confirmed"].map({True: "出典あり", False: "要確認"}),
    })
