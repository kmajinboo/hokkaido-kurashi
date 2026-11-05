"""Check whether a model's accuracy differs from a baseline by more than chance.

Uses the Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample
correction. Errors of h-month-ahead forecasts from consecutive origins
overlap, so the variance includes autocovariances up to lag h-1.
"""

import numpy as np
import pandas as pd

from src.core.exceptions import ConfigurationError, DataValidationError

_LOSSES = {
    "absolute": np.abs,
    "squared": np.square,
}


def _long_run_variance(d: np.ndarray, h: int) -> float:
    """Variance of the mean of ``d`` allowing autocorrelation up to lag h-1."""
    n = len(d)
    centered = d - d.mean()
    gammas = [float(centered @ centered) / n]
    for k in range(1, h):
        gammas.append(float(centered[k:] @ centered[:-k]) / n)
    variance = (gammas[0] + 2 * sum(gammas[1:])) / n
    if variance <= 0 and h > 1:
        # Fall back to Bartlett weights, which keep the estimate non-negative.
        variance = (gammas[0] + 2 * sum((1 - k / h) * g for k, g in enumerate(gammas[1:], 1))) / n
    return variance


def diebold_mariano(
    predictions: pd.DataFrame, model: str, baseline: str = "naive", loss: str = "absolute"
) -> pd.DataFrame:
    """Compare ``model`` with ``baseline`` at every forecast step.

    ``mean_loss_diff`` is the model's average loss minus the baseline's;
    negative means the model was more accurate. ``p_value`` is two-sided; a
    small value (for example below 0.05) means the difference is unlikely to
    be chance. ``p_value`` is empty when the two models' errors are identical.
    """
    try:
        from scipy import stats
    except ImportError as exc:
        raise ConfigurationError("scipy is missing. Run: python -m pip install statsmodels") from exc
    if loss not in _LOSSES:
        raise DataValidationError(f"loss must be one of {sorted(_LOSSES)}")

    keys = ["origin_date", "target_date", "step"]
    valid = predictions.dropna(subset=["actual", "forecast"])
    left = valid[valid["model"] == model][keys + ["actual", "forecast"]]
    right = valid[valid["model"] == baseline][keys + ["forecast"]]
    if left.empty or right.empty:
        raise DataValidationError(f"Backtest results for {model!r} and {baseline!r} are both required")
    paired = left.merge(right, on=keys, suffixes=("", "_baseline")).sort_values(["step", "origin_date"])

    rows = []
    for step, group in paired.groupby("step", sort=True):
        h = int(step)
        loss_fn = _LOSSES[loss]
        d = (
            loss_fn(group["actual"] - group["forecast"])
            - loss_fn(group["actual"] - group["forecast_baseline"])
        ).to_numpy(dtype=float)
        n = len(d)
        mean_diff = float(d.mean()) if n else float("nan")
        stat = p_value = float("nan")
        if n > h + 1:
            variance = _long_run_variance(d, h)
            if variance > 0:
                correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
                stat = float(mean_diff / np.sqrt(variance) * correction)
                p_value = float(2 * stats.t.sf(abs(stat), df=n - 1))
        rows.append(
            {
                "model": model,
                "baseline": baseline,
                "step": h,
                "n": n,
                "loss": loss,
                "mean_loss_diff": mean_diff,
                "dm_stat": stat,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def compare_models(predictions: pd.DataFrame, baseline: str = "naive", loss: str = "absolute") -> pd.DataFrame:
    """Run ``diebold_mariano`` for every model other than ``baseline``."""
    models = [m for m in predictions["model"].drop_duplicates() if m != baseline]
    if not models:
        raise DataValidationError("No models to compare with the baseline")
    return pd.concat(
        [diebold_mariano(predictions, m, baseline, loss) for m in models], ignore_index=True
    )
