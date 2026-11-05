"""Find e-Stat tables and their codes, and test a catalog setting before adding it.

Usage (PowerShell, in C:\\EconomicForecast with .venv active):

    python estat_tools.py search 消費者物価指数 2025年基準
    python estat_tools.py search 一般職業紹介状況 --limit 30
    python estat_tools.py meta 0004052037 --find 札幌 --find 総合
    python estat_tools.py test 0004052037 cdArea=XXXXX cdCat01=XXXX cdTab=X
    python estat_tools.py test 0004052037 cdArea=XXXXX --save

``search`` lists tables (statsDataId and title). ``meta`` lists the codes of
each dimension (area, items, kind of figure); ``--find`` shows only names
containing the given words. ``test`` fetches with the given settings exactly
as the catalog would and shows the latest values, or explains which setting
is still missing. ``--save`` also writes a small sample of the raw response to
data/estat_samples/ (without the application ID) to share for checking.
"""

import argparse
import json
import sys
from pathlib import Path

from src.core.config import DATA_DIR
from src.core.exceptions import EFPError
from src.datasource.estat import as_list, call, fetch_values, text_of, values_to_series


def _redact(obj):
    if isinstance(obj, dict):
        return {k: _redact(v) for k, v in obj.items() if "appid" not in k.lower()}
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


def cmd_search(args) -> None:
    params = {"searchWord": " ".join(args.words), "limit": args.limit}
    if args.code:
        params["statsCode"] = args.code
    root = call("getStatsList", params)
    info = root.get("DATALIST_INF", {})
    tables = as_list(info.get("TABLE_INF"))
    print(f"該当 {info.get('NUMBER', len(tables))} 件（表示 {len(tables)} 件）")
    for t in tables:
        title = text_of(t.get("TITLE")) or text_of(t.get("STATISTICS_NAME"))
        print(
            f"{t.get('@id')}  {text_of(t.get('STAT_NAME'))} / {text_of(t.get('STATISTICS_NAME'))}\n"
            f"    表題: {title}  周期: {text_of(t.get('CYCLE'))}  調査年月: {text_of(t.get('SURVEY_DATE'))}"
            f"  公開: {text_of(t.get('OPEN_DATE'))}"
        )


def cmd_meta(args) -> None:
    root = call("getMetaInfo", {"statsDataId": args.stats_data_id})
    table = root.get("METADATA_INF", {}).get("TABLE_INF", {})
    print(f"{args.stats_data_id}  {text_of(table.get('STATISTICS_NAME'))}  {text_of(table.get('TITLE'))}")
    for obj in as_list(root.get("METADATA_INF", {}).get("CLASS_INF", {}).get("CLASS_OBJ")):
        dim = obj.get("@id", "")
        setting = "cd" + dim[:1].upper() + dim[1:]
        classes = as_list(obj.get("CLASS"))
        print(f"\n[{setting}] {obj.get('@name')}（{len(classes)}件）")
        if dim == "time":
            if classes:
                print(f"    {classes[0].get('@code')} {classes[0].get('@name')} 〜 "
                      f"{classes[-1].get('@code')} {classes[-1].get('@name')}")
            continue
        shown = [c for c in classes if not args.find or any(w in c.get("@name", "") for w in args.find)]
        for c in shown[: args.limit]:
            unit = f"  単位: {c.get('@unit')}" if c.get("@unit") else ""
            level = f"  階層{c.get('@level')}" if c.get("@level") else ""
            print(f"    {c.get('@code')}  {c.get('@name')}{level}{unit}")
        if len(shown) > args.limit:
            print(f"    ...（ほか {len(shown) - args.limit} 件。--find で絞り込めます）")


def cmd_test(args) -> None:
    options = {}
    for item in args.settings:
        key, sep, value = item.partition("=")
        if not sep:
            sys.exit(f"設定は key=value の形で書いてください: {item}")
        options[key] = value
    if args.save:
        root = call("getStatsData", {**options, "statsDataId": args.stats_data_id, "limit": 20})
        out_dir = DATA_DIR / "estat_samples"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{args.stats_data_id}.json"
        path.write_text(json.dumps(_redact(root), ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"応答の見本を保存しました: {path}")
    values, classes = fetch_values(args.stats_data_id, options)
    series = values_to_series(values, classes)
    valid = series.dropna(subset=["value"])
    print(f"取得できました: {len(series)}か月分（{series['date'].iloc[0][:7]} 〜 {series['date'].iloc[-1][:7]}）")
    print("直近13か月:")
    for row in valid.tail(13).itertuples():
        print(f"    {row.date[:7]}  {row.value:g}")
    joined = ";".join(f"{k}={v}" for k, v in options.items())
    print("\nカタログ（config/series_catalog.csv）に書く値:")
    print(f"    source=estat  source_id={args.stats_data_id}  source_options={joined}")


def main() -> None:
    parser = argparse.ArgumentParser(description="e-Statの統計表と分類コードを調べる")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("search", help="統計表を検索する")
    p.add_argument("words", nargs="+")
    p.add_argument("--code", help="政府統計コード（例: 00200573 消費者物価指数）")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_search)
    p = sub.add_parser("meta", help="統計表の分類コードを表示する")
    p.add_argument("stats_data_id")
    p.add_argument("--find", action="append", help="名前に含む語で絞り込む（複数可）")
    p.add_argument("--limit", type=int, default=40)
    p.set_defaults(func=cmd_meta)
    p = sub.add_parser("test", help="設定どおりに取得できるか試す")
    p.add_argument("stats_data_id")
    p.add_argument("settings", nargs="*", help="cdArea=... cdCat01=... など")
    p.add_argument("--save", action="store_true", help="応答の見本を data/estat_samples に保存")
    p.set_defaults(func=cmd_test)
    args = parser.parse_args()
    try:
        args.func(args)
    except EFPError as exc:
        sys.exit(f"エラー: {exc}")


if __name__ == "__main__":
    main()
