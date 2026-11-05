"""Run the combination analysis and write the monthly 「暮らしを取り巻く状況」 outputs."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.catalog import SeriesSpec
from src.situation.lead_lag import add_derived, lead_lag_table
from src.situation.events import add_notes, events_for_powerbi, events_section, load_events, recent_and_upcoming, touyu_table
from src.situation.outside import load_reference, outside_point, outside_section
from src.situation.phase import combine, judge
from src.situation.transforms import CHANGE_LABELS, latest_summary, yoy_changes


def season_notes(path: Path, months: list[int]) -> pd.DataFrame:
    """Notes from ``config/season_notes.csv`` that cover any of ``months``.

    A row covers month_from..month_to and may wrap over the new year (10..4).
    """
    if not Path(path).exists():
        return pd.DataFrame(columns=["month_from", "month_to", "topic", "note"])
    notes = pd.read_csv(path, encoding="utf-8-sig", dtype={"topic": str, "note": str})

    def covers(row) -> bool:
        start, end = int(row["month_from"]), int(row["month_to"])
        span = range(start, end + 1) if start <= end else [*range(start, 13), *range(1, end + 1)]
        return any(m in span for m in months)

    return notes[notes.apply(covers, axis=1)].reset_index(drop=True)


def fmt_change(value: float, unit: str) -> str:
    """+3.1% / -4.5% / 0.0ポイント (no "-0.0")."""
    if pd.isna(value):
        return "-"
    rounded = round(float(value), 1) + 0.0  # + 0.0 turns -0.0 into 0.0
    sign = "+" if rounded > 0 else ""
    return f"{sign}{rounded:.1f}{unit}"


_fmt = fmt_change


def build_report(latest: pd.DataFrame, judgments: pd.DataFrame, points: list[str],
                 lags: pd.DataFrame, notes: pd.DataFrame, today: date, outside: list[str] | None = None,
                 timeline: list[str] | None = None, charts: list[str] | None = None) -> str:
    """Markdown report (no AI): situation, points to keep in mind, lags, season."""
    lines = [f"# 暮らしを取り巻く状況（{today:%Y年%m月%d日}作成）", "",
             "経済指標の前年同月との比較から、生活相談の背景にありうる外的な一因を整理したものです。"
             "判定の基準は目安で、個々の相談者の原因を示すものではありません。", "",
             "## 確認ポイント", ""]
    lines += [f"- {p}" for p in points]
    lines += ["", "## 観点ごとの判定", "", "「注記」は、補助や制度の変更で前年との比較が動いているものです（config/events.csv の年表から）。", "",
              "| 観点 | 使った指標 | 時点 | 前年との比較 | 判定 | 注記 |", "|---|---|---|---|---|---|"]
    for r in judgments.itertuples():
        note = getattr(r, "note", "") or ""
        lines.append(f"| {r.aspect} | {r.name} | {str(r.date)[:7]}（{r.basis}） | {_fmt(r.change, r.change_unit)} | {r.judgment} | {note} |")
    lines += charts or []
    lines += outside or []
    lines += timeline or []
    lines += ["", "## 世界の動きが暮らしに届くまでの時間差（目安）", "",
              "上流の前年比を0〜24か月ずらし、下流の前年比と最もよく重なるずれを過去20年で調べたものです。"
              "相関は原因を示すものではありません。", "",
              "| 上流 | 下流 | 時間差 | 相関 | 下流の最新 | 見通し（直線の当てはめ） |", "|---|---|---|---|---|---|"]
    for r in lags.itertuples():
        if r.status != "ok":
            lines.append(f"| {r.upstream_name} | {r.downstream_name} | - | - | - | {r.status} |")
            continue
        outlook = (f"{str(r.outlook_until)[:7]}ごろ {_fmt(r.outlook_change, '%')}"
                   if r.outlook_until else "（時間差0か月のため見通しなし）")
        lag = f"{int(r.best_lag_months)}か月" + ("以上（探した範囲の上限。参考程度）" if r.at_limit else "")
        lines.append(f"| {r.upstream_name} | {r.downstream_name} | {lag} | {r.correlation:.2f} | "
                     f"{str(r.down_latest_date)[:7]} {_fmt(r.down_latest_change, '%')} | {outlook} |")
    lines += ["", "## 季節の確認ポイント（今月・来月）", ""]
    lines += [f"- **{r.topic}**：{r.note}" for r in notes.itertuples()] or ["- （該当なし）"]
    lines += ["", "## 指標の一覧（最新値と前年との比較）", "",
              "「過去10年での位置」は、最新の変化が過去10年の変化の何％より大きいかを示します（95なら、過去10年でも大きい上昇）。", "",
              "| 層 | 指標 | 時点 | 値 | 前年との比較 | 過去10年での位置 | 注記 |", "|---|---|---|---|---|---|---|"]
    for r in latest.itertuples():
        position = "-" if pd.isna(r.percentile) else f"{r.percentile:.0f}"
        note = getattr(r, "note", "") or ""
        lines.append(f"| {r.layer} | {r.name} | {str(r.date)[:7]} | {r.value:,.2f} {r.unit} | "
                     f"{_fmt(r.change, r.change_unit)}（{CHANGE_LABELS[r.compare]}） | {position} | {note} |")
    return "\n".join(lines) + "\n"


def _with_extra_points(points: list[str], outside_extra: str | None, notes_all: list[str]) -> list[str]:
    points = list(points)
    if outside_extra:  # right after the pension point, or first if there is none
        at = next((i + 1 for i, p in enumerate(points) if p.startswith("年金")), 0)
        points.insert(at, outside_extra)
    if any(("抑えられている" in n or "終わった" in n) for n in notes_all):
        points.append("燃料や光熱費の一部は補助で価格が抑えられているか、補助が終わったため前年比が高めに出ています。補助が縮小・終了すると、"
                      "前年比が急に上がることがあります（「観点ごとの判定」の注記を参照）。")
    return points


def analyse(observations: pd.DataFrame, specs: list[SeriesSpec], analysis_dir: Path, powerbi_dir: Path,
            report_dir: Path, season_path: Path, today: date | None = None,
            outside_path: Path | None = None, public_report_dir: Path | None = None) -> dict:
    """Run every step, write the files and return the main tables.

    Two reports are written: the internal one (``report_dir``) and a public one
    (``public_report_dir``, default ``<report_dir>/../public/reports``) without
    the parts that stay internal: local events, 福祉灯油 and the 計算例.
    """
    today = today or date.today()
    active = [s for s in specs if s.active]
    observations, active = add_derived(observations, active)
    names = {s.series_id: s.name for s in active}
    changes = yoy_changes(observations, active)
    latest = latest_summary(changes, active)
    judgments = judge(changes, names)
    base_points = combine(judgments)
    lags = lead_lag_table(changes, names)
    months = [today.month, today.month % 12 + 1]
    notes = season_notes(season_path, months)
    config_dir = Path(season_path).parent
    reference = load_reference(outside_path or config_dir / "households_outside.csv")
    events = load_events(config_dir)
    judgments = add_notes(judgments, events)
    latest = add_notes(latest, events)
    notes_all = list(judgments["note"]) + list(latest["note"])
    touyu = touyu_table(config_dir, changes, today)

    # internal report
    outside = outside_section(changes, reference)
    points = _with_extra_points(base_points, outside_point(changes, reference), notes_all)
    recent = recent_and_upcoming(events, today)
    timeline = events_section(events, recent, touyu, today)

    # public report: national/Hokkaido events only, no 福祉灯油, no 計算例
    public_events = events[events["scope"] != "担当地域"] if not events.empty else events
    public_outside = outside_section(changes, reference, public=True)
    public_points = _with_extra_points(base_points, outside_point(changes, reference, public=True), notes_all)
    public_timeline = events_section(public_events, recent_and_upcoming(public_events, today), pd.DataFrame(), today)

    out = Path(analysis_dir) / "situation"
    out.mkdir(parents=True, exist_ok=True)
    changes.to_csv(out / "changes.csv", index=False, encoding="utf-8-sig")
    latest.to_csv(out / "latest.csv", index=False, encoding="utf-8-sig")
    judgments.to_csv(out / "judgments.csv", index=False, encoding="utf-8-sig")
    lags.to_csv(out / "lead_lag.csv", index=False, encoding="utf-8-sig")

    meta = pd.DataFrame([{"series_id": s.series_id, "name": s.name, "layer": s.layer, "category": s.category} for s in active])
    bi = changes.merge(meta, on="series_id", how="left")
    bi["change_label"] = bi["compare"].map(CHANGE_LABELS)
    Path(powerbi_dir).mkdir(parents=True, exist_ok=True)
    bi[["date", "series_id", "name", "layer", "category", "value", "change", "change_label"]].to_csv(
        Path(powerbi_dir) / "indicators_yoy.csv", index=False, encoding="utf-8-sig")
    events_for_powerbi(events).to_csv(Path(powerbi_dir) / "events.csv", index=False, encoding="utf-8-sig")

    import shutil

    chart_dir = Path(report_dir).parent / "charts"
    public_dir = Path(public_report_dir) if public_report_dir else Path(report_dir).parent / "public" / "reports"
    try:  # matplotlib is needed only for the charts; the reports work without it
        from src.situation.charts import charts_section, draw_all

        drawn = draw_all(changes, public_events, lags, chart_dir)
    except Exception as exc:  # a chart must never stop the report
        print(f"グラフを作れませんでした: {type(exc).__name__}: {exc}")
        drawn, charts_section = {}, (lambda *_: [])
    public_chart_dir = public_dir.parent / "charts"
    public_chart_dir.mkdir(parents=True, exist_ok=True)
    for chart in drawn.values():
        shutil.copy2(chart, public_chart_dir / chart.name)
    chart_lines = charts_section(drawn, "../charts")

    name = f"situation_{today:%Y%m%d}.md"
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    report_path = Path(report_dir) / name
    report_path.write_text(build_report(latest, judgments, points, lags, notes, today, outside, timeline, chart_lines),
                           encoding="utf-8")
    public_dir.mkdir(parents=True, exist_ok=True)
    public_path = public_dir / name
    public_path.write_text(build_report(latest, judgments, public_points, lags, notes, today, public_outside, public_timeline,
                                           chart_lines),
                           encoding="utf-8")
    return {"changes": changes, "latest": latest, "judgments": judgments, "points": points,
            "lags": lags, "notes": notes, "outside": outside, "events": events, "recent": recent, "touyu": touyu,
            "report_path": report_path, "public_report_path": public_path, "charts": drawn}
