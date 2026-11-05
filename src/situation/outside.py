"""Households outside the monthly statistics: single pensioners.

The monthly statistics (家計調査 etc.) mostly describe working households.
Single pensioners, a large group among people who come for consultation, are
described only by occasional surveys. This module keeps a few figures from
the latest one (年金制度基礎調査 2022, in ``config/households_outside.csv``)
and moves them forward to now with the pension index and Hokkaido prices.

The figures are national; Hokkaido adds heating costs on top.
"""

from pathlib import Path

import pandas as pd

PENSION_ID = "PENSION_INDEX_JP"
PRICE_ID = "CPI_HKD_ALL"


def load_reference(path: Path) -> dict[str, dict]:
    """{key: {"value", "unit", "label", "source"}} from the reference CSV (empty if missing)."""
    if not Path(path).exists():
        return {}
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    return {r["key"]: r for r in frame.to_dict("records")}


def _level(changes: pd.DataFrame, series_id: str) -> pd.Series:
    rows = changes[changes["series_id"] == series_id].dropna(subset=["value"])
    return rows.set_index("date")["value"].sort_index()


def since_survey(changes: pd.DataFrame, survey_month: str) -> dict | None:
    """Pension and Hokkaido price rise from the survey month to the latest month both have."""
    pension, prices = _level(changes, PENSION_ID), _level(changes, PRICE_ID)
    start = f"{survey_month}-01"
    common = sorted(set(pension.index) & set(prices.index))
    if start not in pension.index or start not in prices.index or not common or common[-1] <= start:
        return None
    last = common[-1]
    return {"start": start, "last": last,
            "pension_rise": (pension[last] / pension[start] - 1) * 100,
            "price_rise": (prices[last] / prices[start] - 1) * 100}


def outside_section(changes: pd.DataFrame, reference: dict[str, dict], public: bool = False) -> list[str]:
    """Markdown lines for the report (empty when the reference file is missing).

    ``public=True`` leaves out the 計算例 (it uses figures seen in our own
    consultations, which stay internal) and keeps only the survey figures.
    """
    if not reference:
        return []
    v = lambda key: reference[key]["value"]  # noqa: E731
    lines = ["", "## 統計の外にいる世帯（年金で暮らす単身世帯）", "",
             f"毎月の統計は主に働いている世帯の数字です。ここでは{reference['survey_month']['source']}の"
             f"全国の数字を基準に、その後の年金と北海道の物価の動きを当てはめています。", "",
             f"**基準（{v('survey_month')}時点、単身の受給者 約{v('single_total')}万人）**", "",
             f"- 公的年金が年100万円未満：{v('low_pension_share')}%。そのうち生活保護を受けているのは"
             f"{v('low_pension_assistance_share')}%で、残りの多くは保護を受けずに暮らしています",
             f"- 支出が収入を上回っていることが明らかな人：約{v('deficit_share')}%（少なめの見積もり）",
             f"- 貯蓄が100万円未満：{v('low_savings_share')}%",
             f"- 賃貸住宅に住む人：{v('renter_share')}%（家賃の値上がりの影響を受けやすい）",
             f"- 働いていない人のうち「働くことができない」：{v('unable_to_work_share')}%"
             f"（配偶者のいない受給者。働いている人の{v('must_work_share')}%は「生活にどうしても必要だから」）",
             f"- 支出（月額）の中央値：{v('spending_median')}万円"]
    moved = since_survey(changes, v("survey_month"))
    if moved:
        spending = float(v("spending_median")) * 10000
        extra_cost = spending * moved["price_rise"] / 100
        lines += ["", f"**今の状況（{moved['start'][:7]}から{moved['last'][:7]}まで）**", "",
                  f"- 年金は{moved['pension_rise']:+.1f}%、北海道の物価は{moved['price_rise']:+.1f}%"
                  f"（差 {moved['pension_rise'] - moved['price_rise']:+.1f}ポイント）",
                  f"- 調査時に月{spending / 10000:.1f}万円（中央値）で暮らしていた人が同じ暮らしを続けると、"
                  f"今は月に約{extra_cost:,.0f}円多くかかる計算です"]
    if moved and not public and "example_pension" in reference:
        pension_month = float(v("example_pension")) * 10000 / 12
        example_spending = float(v("example_spending")) * 10000
        example_cost = example_spending * moved["price_rise"] / 100
        extra_pension = pension_month * moved["pension_rise"] / 100
        lines += [f"- 計算例：年金が年{v('example_pension')}万円（月約{pension_month:,.0f}円）で、"
                  f"月{example_spending / 10000:.1f}万円で暮らしていた人（{reference['example_spending']['label']}）は、"
                  f"支出が月に約{example_cost:,.0f}円増える一方、年金の増え方は月に約{extra_pension:,.0f}円です"]
        before = example_spending - pension_month
        now = example_spending + example_cost - pension_month - extra_pension
        if before > 0 or now > 0:
            lines.append(f"- 毎月の不足は、調査時の約{before:,.0f}円から今は約{now:,.0f}円になっている計算です")
        if now > 0 and "example_savings" in reference:
            savings = float(v("example_savings")) * 10000
            lines.append(f"- この不足を貯蓄で埋めると、貯蓄{v('example_savings')}万円は約{savings / (now * 12):.1f}年でなくなります"
                         f"（{reference['example_savings']['source']}）")
    if public:
        lines += ["", "全国の数字です。北海道ではこれに冬の暖房費（灯油）の上がり方が加わります。"
                  "実際の家計には、医療費など個人の事情も重なります。この計算は、そのうち物価と年金のずれによる部分（外的な一因）だけを示します。"]
        return lines
    lines += ["", "全国の数字です。北海道ではこれに冬の暖房費（灯油）の上がり方が加わります。"
              "計算例は一部の相談者に見られる目安を使ったもので、相談者全体を代表するものではありません。"
              "実際の不足には、病気（医療費）や嗜好品（酒・たばこ）など個人の事情も重なります。"
              "この計算は、そのうち物価と年金のずれによる部分（外的な一因）だけを示します。"]
    return lines


def outside_point(changes: pd.DataFrame, reference: dict[str, dict], public: bool = False) -> str | None:
    """One 確認ポイント when pensions have fallen behind Hokkaido prices since the survey."""
    if not reference:
        return None
    moved = since_survey(changes, reference["survey_month"]["value"])
    if not moved or moved["pension_rise"] >= moved["price_rise"]:
        return None
    text = (f"年金で暮らす単身世帯では、{moved['start'][:4]}年からの年金の伸び（{moved['pension_rise']:+.1f}%）が"
            f"北海道の物価（{moved['price_rise']:+.1f}%）に届いていません。")
    if public or "example_pension" not in reference:
        return text + "（「統計の外にいる世帯」の節を参照）"
    pension_month = float(reference["example_pension"]["value"]) * 10000 / 12
    spending = float(reference["example_spending"]["value"]) * 10000
    before = spending - pension_month
    now = spending * (1 + moved["price_rise"] / 100) - pension_month * (1 + moved["pension_rise"] / 100)
    if before > 0 and now > before:
        text += (f"計算例として、調査時に月{before / 10000:.1f}万円足りなかった人は、物価と年金のずれだけで"
                 f"今は月約{now / 10000:.1f}万円の不足になります（「統計の外にいる世帯」の節を参照）。")
    return text
