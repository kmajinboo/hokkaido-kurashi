"""Create a one-off Japanese FEDFUNDS report using the configured AI provider."""

import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.ai.report import generate_report
from src.core.config import OUTPUT_DIR, PROCESSED_DIR, RAW_DIR
from src.forecasting import describe_backtest


def build_context(analysis_dir: Path) -> dict:
    """Collect the figures the AI may use, with the conditions of each evaluation."""
    descriptive = pd.read_csv(analysis_dir / "FEDFUNDS_descriptive_stats.csv")
    annual = pd.read_csv(analysis_dir / "FEDFUNDS_annual_summary.csv")
    metrics = pd.read_csv(analysis_dir / "FEDFUNDS_backtest_metrics.csv")
    raw_path = RAW_DIR / "fred" / "FEDFUNDS.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path.name} がありません。先に python main.py を実行してください。")
    recent = pd.read_csv(raw_path).dropna(subset=["value"]).tail(13)
    required = {
        name: analysis_dir / f"FEDFUNDS_{name}.csv"
        for name in (
            "rolling_predictions",
            "rolling_metrics",
            "model_comparison",
            "forecast_intervals",
            "interval_coverage",
        )
    }
    missing = [path.name for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{', '.join(missing)} がありません。先に python main.py を実行してください。"
        )
    rolling_predictions = pd.read_csv(required["rolling_predictions"])
    rolling = pd.read_csv(required["rolling_metrics"])
    comparison = pd.read_csv(required["model_comparison"])
    intervals = pd.read_csv(required["forecast_intervals"])
    coverage = pd.read_csv(required["interval_coverage"])

    return {
        "series": "FEDFUNDS",
        "descriptive_statistics": dict(zip(descriptive["metric"], descriptive["value"])),
        "recent_annual_summary": json.loads(annual.tail(10).to_json(orient="records")),
        "recent_actual_values": {
            "description": "直近13か月の実績（月次平均、%）。最後の行が最新の実績。",
            "values": json.loads(recent.to_json(orient="records")),
        },
        "one_step_backtest": {
            "description": (
                "各月について、その月より前の実績だけで1か月先を予測したときの誤差。"
                "12か月先予測の精度ではない。"
            ),
            "forecast_step_months": 1,
            "test_periods": int(metrics["observations"].max()),
            "metrics": json.loads(metrics.to_json(orient="records")),
        },
        "rolling_multistep_backtest": {
            "description": (
                "複数の予測起点から、それぞれ12か月先までを一度に予測したときの誤差。"
                "stepは予測起点から何か月先かを表す。"
                "mae_ratio_to_naiveは同じstepでのNaiveとのMAE比で、1未満ならNaiveより正確。"
                "予測値は0未満にならないよう0で下限を設けている（naive・seasonal_naive以外）。"
            ),
            "conditions": describe_backtest(rolling_predictions),
            "metrics_by_step": json.loads(rolling.to_json(orient="records")),
        },
        "model_comparison_test": {
            "description": (
                "ローリング評価の誤差について、各モデルとNaiveの差を検定した結果"
                "（Diebold-Mariano検定、絶対誤差、HLN補正）。"
                "mean_loss_diffが負ならNaiveより誤差が小さい。"
                "p_valueが0.05未満なら差が偶然とは考えにくい。p_valueが空の行は誤差が同一で検定できない。"
            ),
            "results": json.loads(comparison.to_json(orient="records")),
        },
        "forecast_with_intervals": {
            "description": (
                "基準月の翌月から12か月先までの予測値と予測の幅（lower〜upper）。"
                "幅は、過去の複数の予測起点で同じモデルが同じ月数先に出した誤差から作った目安。"
                "ruleは幅の作り方で、coverage_checkのadoptedがtrueの行と同じもの。"
                "n_errorsは使った誤差の件数。"
                "将来の誤差が過去と同程度であることを前提にしている。確率を保証するものではない。"
                "lowerは0未満にならないようにしている。"
            ),
            "forecasts": json.loads(intervals.to_json(orient="records")),
            "coverage_check": {
                "description": (
                    "幅の作り方（rule）ごとに、過去の各予測起点でその時点で分かっていた誤差だけを使って幅を作り、"
                    "実績が幅に入った割合（coverage）を同じ検証期間で比べた結果。"
                    "symmetric=trueは上下対称の幅、error_originsは使った予測起点の数、"
                    "exclude_belowはその水準未満の時期（ゼロ金利期など）の誤差を除いたことを表す。"
                    "mean_widthは幅の平均（ポイント）。adoptedがtrueの作り方を将来予測に使っている。"
                    "coverageがinterval_levelに近いほど幅の大きさが適切。"
                ),
                "results": json.loads(coverage.to_json(orient="records")),
            },
        },
    }


def main() -> None:
    analysis_dir = PROCESSED_DIR / "fred"
    report = generate_report(build_context(analysis_dir))
    output_dir: Path = OUTPUT_DIR / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"FEDFUNDS_AI_report_{date.today():%Y%m%d}.md"
    output_path.write_text(report + "\n", encoding="utf-8")
    print(f"AIレポート保存先: {output_path}")


if __name__ == "__main__":
    main()
