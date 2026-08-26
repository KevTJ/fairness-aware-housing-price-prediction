"""Predictive-performance and calibration evaluation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return core regression metrics on the original dollar scale."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if np.any(y_true <= 0):
        raise ValueError("MAPE requires strictly positive observed sale prices.")

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "MAPE": np.mean(np.abs((y_pred - y_true) / y_true)) * 100,
        "R2": r2_score(y_true, y_pred),
    }


def prediction_diagnostics(y_true, y_pred) -> pd.DataFrame:
    """Return observation-level residual and percentage-error diagnostics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if np.any(y_true <= 0):
        raise ValueError("Percentage diagnostics require positive observed prices.")

    out = pd.DataFrame({"actual": y_true, "predicted": y_pred})
    out["residual"] = out["predicted"] - out["actual"]
    out["absolute_error"] = out["residual"].abs()
    out["signed_pct_error"] = out["residual"] / out["actual"] * 100
    out["absolute_pct_error"] = out["signed_pct_error"].abs()
    out["prediction_ratio"] = out["predicted"] / out["actual"]
    return out


def price_decile_calibration(y_true, y_pred, q: int = 10) -> pd.DataFrame:
    """Summarize prediction calibration across observed-price quantiles."""
    diagnostics = prediction_diagnostics(y_true, y_pred)
    diagnostics["price_group"] = pd.qcut(
        diagnostics["actual"],
        q=q,
        duplicates="drop",
    )

    return (
        diagnostics.groupby("price_group", observed=True)
        .agg(
            n=("actual", "size"),
            median_actual=("actual", "median"),
            median_predicted=("predicted", "median"),
            median_signed_pct_error=("signed_pct_error", "median"),
            median_absolute_pct_error=("absolute_pct_error", "median"),
            median_prediction_ratio=("prediction_ratio", "median"),
        )
        .reset_index()
    )


def metrics_table(results: dict[str, tuple]) -> pd.DataFrame:
    """Build a model-comparison table from {name: (y_true, y_pred)} mappings."""
    rows = []
    for model_name, (y_true, y_pred) in results.items():
        row = {"Model": model_name, **regression_metrics(y_true, y_pred)}
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")
