"""Pages of the public site that are made from the data (used by publish.py).

The fixed pages (方法, ご意見をいただきたい点, ...) live in ``site/docs``.
This module writes the pages that change every month:

- ``index.md``           latest 確認ポイント and three main charts
- ``reports/index.md``   list of monthly reports
- ``charts.md``          every chart with its caption
- ``timeline.md``        national and Hokkaido events (config/events.csv)
and fills ``{{THRESHOLDS}}`` in methods.md and ``{{SOURCES}}`` in sources.md.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.catalog import load_catalog
from src.situation.charts import CHARTS
from src.situation.phase import THRESHOLDS, VIEWPOINTS

MAIN_CHARTS = ["income_gap", "income_gradation", "oil_kerosene", "fuel_subsidies"]
SOURCE_NAMES = {"estat": "e-Stat", "fred": "FRED", "manual": "手入力（発表資料）",
                "hokkaido_monitor": "北海道 消費生活モニター価格動向調査"}


def _report_date(path: Path) -> str:
    match = re.search(r"(\d{4})(\d{2})(\d{2})", path.name)
    return f"{match.group(1)}年{int(match.group(2))}月{int(match.group(3))}日" if match else path.stem


def _section(text: str, heading: str) -> list[str]:
    lines, inside = [], False
    for line in text.splitlines():
        if line.startswith("## "):
            inside = line.strip() == f"## {heading}"
            continue
        if inside:
            lines.append(line)
    return [line for line in lines if line.strip()]


def index_page(reports: list[Path], charts: set[str]) -> str:
    lines = ["# 暮らしを取り巻く状況（北海道）", "",
             "経済指標と制度の動きを毎月同じ方法で整理し、生活困窮の相談の背景にありうる"
             "**外的な一因**を見えるようにする試みです。判定の基準は目安で、個々の相談者の原因を示すものではありません。", ""]
    if reports:
        latest = reports[0]
        points = _section(latest.read_text(encoding="utf-8"), "確認ポイント")
        lines += [f"## 最新の確認ポイント（{_report_date(latest)}）", ""] + points
        lines += ["", f"[月次レポートの全文を読む](reports/{latest.stem}.md)", ""]
    shown = [name for name in MAIN_CHARTS if name in charts]
    if shown:
        lines += ["## 主なグラフ", ""]
        for name in shown:
            title, caption = CHARTS[name]
            lines += [f"![{title}](charts/{name}.svg)", "", caption, ""]
        lines += ["[すべてのグラフを見る](charts.md)", ""]
    lines += ["## このサイトの見方", "",
              "- **月次レポート**：毎月の判定、確認ポイント、年表、指標の一覧",
              "- **方法**：判定の基準と計算の仕方。限界もここに書いています",
              "- **ご意見をいただきたい点**：研究者の方に特に見ていただきたい論点", ""]
    return "\n".join(lines) + "\n"


def reports_page(reports: list[Path]) -> str:
    lines = ["# 月次レポート", "", "新しい順です。各レポートは、その日に取得した最新のデータで作っています。", ""]
    lines += [f"- [{_report_date(p)}]({p.name})" for p in reports] or ["- （まだありません）"]
    return "\n".join(lines) + "\n"


def charts_page(charts: set[str]) -> str:
    lines = ["# グラフで見る", "", "グラフは月次レポートと同じデータから毎月作り直しています。"
             "帯や縦線は「制度・出来事の年表」から自動で描いています。", ""]
    for name, (title, caption) in CHARTS.items():
        if name in charts:
            lines += [f"## {title}", "", f"![{title}](charts/{name}.svg)", "", caption, ""]
    return "\n".join(lines) + "\n"


def timeline_page(events_csv: Path) -> str:
    lines = ["# 制度・出来事の年表", "",
             "国と北海道の補助・税・料金改定・給付などです。新しい順。"
             "「要確認」は、日付や金額を出典でまだ確かめていないものです。", "",
             "| 開始 | 終了 | 地域 | 種類 | 内容 | 対象 | 確認 | 出典 |", "|---|---|---|---|---|---|---|---|"]
    if not events_csv.exists():
        return "\n".join(lines[:4]) + "\n（年表はまだありません）\n"
    frame = pd.read_csv(events_csv, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    frame = frame.sort_values("開始", ascending=False)
    for r in frame.to_dict("records"):
        source = r.get("出典", "")
        if source.startswith("http"):
            link = f"[リンク]({source})"
        elif "ユーザー提供" in source:
            link = "提供を受けた資料（" + ("内閣官房の公表資料の写し" if "内閣官房" in source else "原典を確認中") + "）"
        elif source.startswith("data/manual"):
            link = "厚生労働省・北海道労働局の発表"
        else:
            link = source
        status = "出典あり" if r.get("確認状況", "").startswith("出典あり") else "要確認"
        cells = [r.get("開始", ""), r.get("終了", ""), r.get("地域", ""), r.get("種類", ""),
                 r.get("内容", ""), r.get("対象", ""), status, link]
        lines.append("| " + " | ".join(c.replace("|", "／") for c in cells) + " |")
    return "\n".join(lines) + "\n"


def thresholds_table() -> str:
    kinds = {}
    for aspect, kind, _ in VIEWPOINTS:
        kinds.setdefault(kind, []).append(aspect)
    kinds.setdefault("real_wage", []).append("収入と物価（1年）")
    kinds.setdefault("cumulative", []).append("年金と物価（累積）")
    unit = {"jobs_ratio": "ポイント", "unemployment": "ポイント", "real_wage": "ポイント",
            "cumulative": "ポイント", "sentiment": "ポイント"}
    lines = ["| 観点 | 区分（左の値以上でその区分） |", "|---|---|"]
    for kind, aspects in kinds.items():
        bands = THRESHOLDS.get(kind)
        if not bands:
            continue
        u = unit.get(kind, "%")
        text = "、".join(f"{label}（{low:+g}{u}以上）" if np.isfinite(low) else f"{label}（それ未満）" for low, label in bands)
        lines.append(f"| {'・'.join(aspects)} | {text} |")
    return "\n".join(lines)


def sources_table(catalog_csv: Path) -> str:
    specs = [s for s in load_catalog(catalog_csv) if s.active]
    lines = ["| 層 | 系列 | 出典 | 統計表ID・系列ID | 頻度 |", "|---|---|---|---|---|"]
    for s in specs:
        source = SOURCE_NAMES.get(s.source, s.source)
        sid = s.source_id if s.source in ("estat", "fred") else ""
        lines.append(f"| {s.layer} | {s.name} | {source} | {sid} | {'月次' if s.frequency == 'M' else '四半期'} |")
    return "\n".join(lines)


def write_site_pages(target: Path, base: Path) -> list[str]:
    """Write the generated pages into ``target/docs``. Returns their paths."""
    docs = Path(target) / "docs"
    reports = sorted((docs / "reports").glob("situation_*.md"), reverse=True)
    charts = {p.stem for p in (docs / "charts").glob("*.svg")}
    pages = {
        "index.md": index_page(reports, charts),
        "reports/index.md": reports_page(reports),
        "charts.md": charts_page(charts),
        "timeline.md": timeline_page(Path(base) / "config" / "events.csv"),
    }
    written = []
    for rel, text in pages.items():
        path = docs / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(f"docs/{rel}")
    for rel, placeholder, filler in (("methods.md", "{{THRESHOLDS}}", thresholds_table),
                                     ("sources.md", "{{SOURCES}}", lambda: sources_table(Path(base) / "config" / "series_catalog.csv"))):
        path = docs / rel
        if path.exists() and placeholder in path.read_text(encoding="utf-8"):
            path.write_text(path.read_text(encoding="utf-8").replace(placeholder, filler()), encoding="utf-8")
            written.append(f"docs/{rel}（表を記入）")
    return written
