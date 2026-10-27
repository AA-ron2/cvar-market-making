"""Leakage-safe, deterministic model-selection rules."""

from __future__ import annotations

import math

import pandas as pd


def require_b2_alpha(alpha: float) -> float:
    """A B2 drawdown weight is finite and positive (F-05). alpha = 0 is the B1+ control (D-18), never B2."""

    value = float(alpha)
    if not (math.isfinite(value) and value > 0):
        raise ValueError(
            f"B2 needs a finite alpha > 0 (F-05), got {alpha!r}. alpha = 0 is the B1+ control (D-18): "
            "train_formula_b2.py --phase final --control b1plus"
        )
    return value


def choose_b2_alpha(summary: pd.DataFrame) -> float:
    """Select B2 alpha from validation summaries only.

    With positive mean validation Sharpe, minimize mean MaxDD, then maximize
    Sharpe, then prefer smaller alpha.  Otherwise maximize Sharpe, minimize
    MaxDD, and prefer smaller alpha. Every alpha must be a B2 alpha (> 0), so
    the B1+ control can never be a candidate (D-18).
    """

    required = {"alpha", "mean_sharpe", "mean_maxdd"}
    if not required.issubset(summary.columns):
        raise ValueError(f"B2 summary missing columns: {sorted(required - set(summary.columns))}")
    for alpha in summary["alpha"]:
        require_b2_alpha(alpha)
    positive = summary[summary["mean_sharpe"] > 0]
    if len(positive):
        ordered = positive.sort_values(
            ["mean_maxdd", "mean_sharpe", "alpha"], ascending=[True, False, True]
        )
    else:
        ordered = summary.sort_values(
            ["mean_sharpe", "mean_maxdd", "alpha"], ascending=[False, True, True]
        )
    return float(ordered.iloc[0]["alpha"])
