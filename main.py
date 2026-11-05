"""Collect all catalog series, then analyse and forecast FEDFUNDS."""

import pandas as pd

from src.catalog import catalog_frame, load_catalog
from src.core.config import BASE_DIR, DATA_DIR, FEATURE_DIR, OUTPUT_DIR, POWERBI_DIR, PROCESSED_DIR, RAW_DIR
from src.core.logger import get_logger
from src.database import PostgresRepository, save_series
from src.datasource import EStatSource, FredSource, HokkaidoMonitorSource, ManualCsvSource, collect, series_values
from src.features import build_lag_features
from src.processing import normalize_series
from src.analytics import annual_summary, descriptive_statistics
from src.forecasting import (
    available_models,
    compare_models,
    compare_to_baseline,
    evaluate_baselines,
    forecast_baselines,
    IntervalRule,
    forecast_with_intervals,
    get_forecaster,
    realtime_coverage,
    rolling_backtest,
    rolling_metrics,
)
from src.visualization import build_bi_forecast, build_bi_series
from src.visualization.forecast_chart import build_history_forecast
from src.situation import analyse
from src.situation.run import fmt_change


logger = get_logger(__name__)

# 1か月先予測を直近何か月分評価するか（evaluate_baselines）
ONE_STEP_TEST_PERIODS = 60
# ローリング評価: 何か月先までを、直近何回の予測起点から評価するか
ROLLING_HORIZON = 12
ROLLING_ORIGINS = 120
# 将来予測を予測の幅付きで出すモデルと、幅の確率（0.8 = 80%）
INTERVAL_MODELS = ["naive", "arima"]
INTERVAL_LEVEL = 0.8
# Power BIの「実績＋予測」グラフに載せる過去の実績の月数（60 = 5年）
CHART_HISTORY_MONTHS = 60
# 予測の幅の作り方の候補。同じ検証期間で当たり具合を比べ、ADOPTED_RULE を将来予測に使う。
#   symmetric: True=上下対称の幅 / False=過去の誤差の上下の偏りをそのまま使う
#   n_origins: 直近何回の予測起点の誤差を使うか
#   exclude_below: この水準未満の時期（ゼロ金利期など）の誤差を除く
INTERVAL_RULES = {
    "asym_120": IntervalRule(symmetric=False, n_origins=120),
    "asym_240": IntervalRule(symmetric=False, n_origins=240),
    "sym_120": IntervalRule(symmetric=True, n_origins=120),
    "sym_240": IntervalRule(symmetric=True, n_origins=240),
    "sym_240_ex0.5": IntervalRule(symmetric=True, n_origins=240, exclude_below=0.5),
}
# 2026-10-29 実データの比較: sym_240 が naive・arima とも79%で目標80%に最も近く、幅も狭い
ADOPTED_RULE = "sym_240"
# 当たり具合の検証: 直近何回の予測起点で確かめるか、どの水準未満の時期を検証から除くか
COVERAGE_TEST_ORIGINS = 120
COVERAGE_TEST_EXCLUDE_BELOW = 0.5
# 観測する系列の一覧と、手入力データの置き場所
CATALOG_PATH = BASE_DIR / "config" / "series_catalog.csv"
MANUAL_DIR = DATA_DIR / "manual"
# 季節の確認ポイント（編集できます）
SEASON_NOTES_PATH = BASE_DIR / "config" / "season_notes.csv"


def collect_all_series():
    """Collect every active catalog series into PostgreSQL and Power BI files."""
    specs = load_catalog(CATALOG_PATH)
    sources = {
        "fred": FredSource(),
        "manual": ManualCsvSource(MANUAL_DIR),
        "estat": EStatSource(),
        "hokkaido_monitor": HokkaidoMonitorSource(MANUAL_DIR / "hokkaido_monitor"),
    }
    observations, summary = collect(specs, sources, PostgresRepository())
    active = [s for s in specs if s.active]

    # Power BI: one table with names and layers, ready for a slicer on name
    POWERBI_DIR.mkdir(parents=True, exist_ok=True)
    indicators = observations.merge(catalog_frame(active), on=["series_id", "source"], how="left")
    indicators[["date", "series_id", "name", "unit", "layer", "category", "value"]].to_csv(
        POWERBI_DIR / "indicators.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(PROCESSED_DIR / "collection_summary.csv", index=False, encoding="utf-8-sig")

    print(f"系列の取得結果（{len(summary)}系列、一覧: {CATALOG_PATH.name}）:")
    for row in summary.itertuples():
        if row.status == "ok":
            print(f"  OK  {row.series_id:<14}{row.name}  最新 {row.last_date[:7]} = {row.last_value:g}（{row.rows}件）")
        else:
            print(f"  NG  {row.series_id:<14}{row.name}  {row.message}")
    skipped = [s.series_id for s in specs if not s.active]
    if skipped:
        print(f"  （取得しない設定 active=0: {', '.join(skipped)}）")
    return observations, specs


def show_situation(observations: pd.DataFrame, specs) -> None:
    """Combination analysis: judgments, points to keep in mind, time lags, season."""
    result = analyse(observations, specs, PROCESSED_DIR, POWERBI_DIR, OUTPUT_DIR / "reports", SEASON_NOTES_PATH)
    print("\n暮らしを取り巻く状況（前年同月との比較。判定の基準は目安）:")
    for row in result["judgments"].itertuples():
        print(f"  {row.aspect:<14}{fmt_change(row.change, row.change_unit)}（{str(row.date)[:7]}） → {row.judgment}")
    print("確認ポイント:")
    for point in result["points"]:
        print(f"  ・{point}")
    lags = result["lags"][result["lags"]["status"] == "ok"]
    if not lags.empty:
        print("世界の動きが届くまでの時間差（目安）:")
        for row in lags.itertuples():
            limit = "以上（上限。参考程度）" if row.at_limit else ""
            print(f"  {row.upstream_name} → {row.downstream_name}: {int(row.best_lag_months)}か月{limit}（相関 {row.correlation:.2f}）")
    print(f"レポート（社内用）: {result['report_path']}")
    print(f"レポート（公開用）: {result['public_report_path']}\n")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    observations, specs = collect_all_series()
    show_situation(observations, specs)

    output_dir = RAW_DIR / "fred"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "FEDFUNDS.csv"
    data = normalize_series(series_values(observations, "FEDFUNDS"))
    data.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Saved %s observations to %s", len(data), output_path)

    feature_dir = FEATURE_DIR / "fred"
    feature_dir.mkdir(parents=True, exist_ok=True)
    feature_path = feature_dir / "FEDFUNDS_features.csv"
    features = build_lag_features(data)
    features.to_csv(feature_path, index=False, encoding="utf-8")
    logger.info("Saved %s feature rows to %s", len(features), feature_path)

    analysis_dir = PROCESSED_DIR / "fred"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    descriptive_path = analysis_dir / "FEDFUNDS_descriptive_stats.csv"
    annual_path = analysis_dir / "FEDFUNDS_annual_summary.csv"
    descriptive_statistics(data).to_csv(descriptive_path, index=False, encoding="utf-8")
    annual_summary(data).to_csv(annual_path, index=False, encoding="utf-8")
    logger.info("Saved descriptive and annual statistics to %s", analysis_dir)

    # 1か月先予測の評価（各月をその月より前の実績だけで予測）
    backtest_predictions, backtest_metrics = evaluate_baselines(
        data, test_periods=ONE_STEP_TEST_PERIODS
    )
    forecast_path = analysis_dir / "FEDFUNDS_baseline_forecast.csv"
    backtest_path = analysis_dir / "FEDFUNDS_backtest_predictions.csv"
    metrics_path = analysis_dir / "FEDFUNDS_backtest_metrics.csv"
    forecasts = forecast_baselines(data, horizon=12)
    forecasts.to_csv(forecast_path, index=False, encoding="utf-8")
    backtest_predictions.to_csv(backtest_path, index=False, encoding="utf-8")
    backtest_metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    logger.info("Saved baseline forecasts and backtest results to %s", analysis_dir)

    # 12か月先までの予測を複数の予測起点から評価（ローリング・マルチステップ）
    rolling_predictions = pd.concat(
        [
            rolling_backtest(
                data,
                get_forecaster(name),
                horizon=ROLLING_HORIZON,
                n_origins=ROLLING_ORIGINS,
            )
            for name in available_models()
        ],
        ignore_index=True,
    )
    rolling_by_step = compare_to_baseline(rolling_metrics(rolling_predictions), baseline="naive")
    rolling_predictions_path = analysis_dir / "FEDFUNDS_rolling_predictions.csv"
    rolling_metrics_path = analysis_dir / "FEDFUNDS_rolling_metrics.csv"
    rolling_predictions.to_csv(rolling_predictions_path, index=False, encoding="utf-8")
    rolling_by_step.to_csv(rolling_metrics_path, index=False, encoding="utf-8")
    logger.info("Saved rolling backtest results to %s", analysis_dir)

    # Naiveとの差が偶然でないかの検定（Diebold-Mariano検定）
    comparison = compare_models(rolling_predictions, baseline="naive")
    comparison_path = analysis_dir / "FEDFUNDS_model_comparison.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8")

    # 将来12か月の予測と、過去の誤差にもとづく予測の幅。
    # 当たり具合の検証には「検証期間＋幅に使う最長の期間」分の過去の予測が必要
    # （系列が短い場合は、取れる分だけにする。各起点に最低60か月の学習期間を残す）
    longest_window = max(rule.n_origins or 0 for rule in INTERVAL_RULES.values())
    interval_origins = min(COVERAGE_TEST_ORIGINS + longest_window, len(data) - ROLLING_HORIZON - 60 + 1)
    if interval_origins < COVERAGE_TEST_ORIGINS + longest_window:
        logger.warning(
            "Series is short: interval backtest uses %s origins instead of %s",
            interval_origins,
            COVERAGE_TEST_ORIGINS + longest_window,
        )
    interval_predictions = pd.concat(
        [
            rolling_backtest(
                data,
                get_forecaster(name),
                horizon=ROLLING_HORIZON,
                n_origins=interval_origins,
            )
            for name in INTERVAL_MODELS
        ],
        ignore_index=True,
    )
    coverage = pd.concat(
        [
            realtime_coverage(
                interval_predictions,
                data,
                rule,
                level=INTERVAL_LEVEL,
                test_origins=COVERAGE_TEST_ORIGINS,
                test_exclude_below=COVERAGE_TEST_EXCLUDE_BELOW,
            ).assign(rule=name, adopted=name == ADOPTED_RULE)
            for name, rule in INTERVAL_RULES.items()
        ],
        ignore_index=True,
    )
    interval_forecasts = pd.concat(
        [
            forecast_with_intervals(
                data,
                get_forecaster(name),
                interval_predictions,
                horizon=ROLLING_HORIZON,
                level=INTERVAL_LEVEL,
                rule=INTERVAL_RULES[ADOPTED_RULE],
            ).assign(rule=ADOPTED_RULE)
            for name in INTERVAL_MODELS
        ],
        ignore_index=True,
    )
    interval_path = analysis_dir / "FEDFUNDS_forecast_intervals.csv"
    coverage_path = analysis_dir / "FEDFUNDS_interval_coverage.csv"
    interval_forecasts.to_csv(interval_path, index=False, encoding="utf-8")
    coverage.to_csv(coverage_path, index=False, encoding="utf-8")
    logger.info("Saved forecast intervals and model comparison to %s", analysis_dir)

    POWERBI_DIR.mkdir(parents=True, exist_ok=True)
    bi_series_path = POWERBI_DIR / "fred_fedfunds.csv"
    bi_forecast_path = POWERBI_DIR / "fred_fedfunds_forecast.csv"
    bi_interval_path = POWERBI_DIR / "fred_fedfunds_forecast_intervals.csv"
    build_bi_series(data).to_csv(bi_series_path, index=False, encoding="utf-8")
    build_bi_forecast(forecasts).to_csv(bi_forecast_path, index=False, encoding="utf-8")
    interval_forecasts[["date", "model", "step", "forecast", "lower", "upper", "interval_level"]].to_csv(
        bi_interval_path, index=False, encoding="utf-8"
    )
    bi_history_forecast_path = POWERBI_DIR / "fred_fedfunds_history_forecast.csv"
    build_history_forecast(data, interval_forecasts, history_months=CHART_HISTORY_MONTHS).to_csv(
        bi_history_forecast_path, index=False, encoding="utf-8"
    )
    logger.info("Saved Power BI datasets to %s", POWERBI_DIR)

    saved = save_series(data)
    logger.info("Upserted %s observations into PostgreSQL table fred_fedfunds", saved)
    print(f"取得件数: {len(data)}")
    print(f"CSV保存先: {output_path}")
    print(f"特徴量CSV保存先: {feature_path}")
    print(f"統計分析保存先: {analysis_dir}")
    print(f"予測・検証結果保存先: {analysis_dir}")
    print(f"Power BI用CSV保存先: {POWERBI_DIR}")
    print(f"PostgreSQL保存件数: {saved}（fred_fedfunds）")
    print(
        f"ローリング評価（直近{ROLLING_ORIGINS}起点、最大{ROLLING_HORIZON}か月先）の"
        "MAE（かっこ内はNaive比。1未満ならNaiveより正確）:"
    )
    shown_steps = [1, 3, 6, ROLLING_HORIZON]
    print("  " + f"{'model':<16}" + "".join(f"{f'{s}か月先':>18}" for s in shown_steps))
    for model in available_models():
        rows = rolling_by_step[rolling_by_step["model"] == model].set_index("step")
        cells = "".join(
            f"{rows.at[s, 'mae']:>10.4f} ({rows.at[s, 'mae_ratio_to_naive']:.2f})" for s in shown_steps
        )
        print(f"  {model:<16}{cells}")

    level_pct = round(INTERVAL_LEVEL * 100)
    print("Naiveとの差の検定（p値。0.05未満なら偶然とは考えにくい）:")
    print("  " + f"{'model':<16}" + "".join(f"{f'{s}か月先':>12}" for s in shown_steps))
    for model in comparison["model"].drop_duplicates():
        rows = comparison[comparison["model"] == model].set_index("step")
        cells = "".join(
            f"{'-':>12}" if pd.isna(rows.at[s, "p_value"]) else f"{rows.at[s, 'p_value']:>12.3f}"
            for s in shown_steps
        )
        print(f"  {model:<16}{cells}")

    print(
        f"予測の幅の作り方の比較（直近{COVERAGE_TEST_ORIGINS}起点のうち水準{COVERAGE_TEST_EXCLUDE_BELOW}以上の時期で、"
        f"その時点で分かっていた誤差だけで幅を作った場合に実績が入った割合。目標 {level_pct}%）:"
    )
    print("  " + f"{'rule':<16}" + "".join(f"{m:>22}" for m in INTERVAL_MODELS))
    for name in INTERVAL_RULES:
        rows = coverage[coverage["rule"] == name].set_index("model")
        cells = "".join(
            f"{rows.at[m, 'coverage']:>10.0%}（幅平均{rows.at[m, 'mean_width']:.2f}）" for m in INTERVAL_MODELS
        )
        mark = " ←採用" if name == ADOPTED_RULE else ""
        print(f"  {name:<16}{cells}{mark}")
    test_count = coverage.iloc[0]
    print(f"  （検証した予測: 1モデルあたり{test_count.test_forecasts}件、誤差不足で除外 {test_count.skipped}件）")

    print(
        f"将来予測と{level_pct}%の予測の幅（基準月 {interval_forecasts['base_date'].iloc[0]}、"
        f"作り方 {ADOPTED_RULE}）:"
    )
    for model in INTERVAL_MODELS:
        rows = interval_forecasts[interval_forecasts["model"] == model].set_index("step")
        cells = "  ".join(
            f"{rows.at[s, 'date'][:7]} {rows.at[s, 'forecast']:.2f} [{rows.at[s, 'lower']:.2f}–{rows.at[s, 'upper']:.2f}]"
            for s in shown_steps
        )
        print(f"  {model:<8}{cells}")
    if not data.empty:
        print(f"最新データ: {data['date'].iloc[-1]} = {data['value'].iloc[-1]}")


if __name__ == "__main__":
    main()
