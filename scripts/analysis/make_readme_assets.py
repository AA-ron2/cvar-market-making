"""The README's results table and figure, from results/final/.

    python scripts/analysis/make_readme_assets.py [--figure docs/figures/cvar_by_model.png]

Prints the README results table in Markdown (the summary over the 23 test days and the two primary contrasts) and
writes the figure: the CVaR0.2 of the seed-averaged daily maximum drawdown per model, with the 23 daily values as
points. Each model's CVaR is recomputed from its daily values with the thesis code (empirical_ru_cvar) and must equal
the summary table's.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cvar_mm.gym.cvar_lagrangian import B3_TAIL_MASS, empirical_ru_cvar  # noqa: E402

MAIN = REPO / "results" / "final" / "main"
FIGURE = REPO / "docs" / "figures" / "cvar_by_model.png"
# (label, summary-table row, per-day file, colour). Colours: the thesis's model palette (Okabe-Ito based); the B1+
# control, added after the palette, takes its secondary-comparison sky blue.
MODELS = [
    ("B0", "B0", "b0_test_results.csv", "#333333"),
    ("B1", "B1", "b1_test_results.csv", "#0072B2"),
    ("B1⁺ (control)", "B1PLUS", "b1plus_test_results.csv", "#56B4E9"),
    ("B2", "B2", "b2_test_results.csv", "#E69F00"),
    ("B3", "B3", "b3_test_results.csv", "#009E73"),
]
TEXT, MUTED, GRID = "#111827", "#6B7280", "#E5E7EB"


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def daily_maxdd(name: str) -> np.ndarray:
    rows = sorted(read_rows(MAIN / name), key=lambda row: row["Day"])
    if len(rows) != 23:
        raise ValueError(f"{name}: expected 23 test days, found {len(rows)}")
    return np.array([float(row["Max DD"]) for row in rows])


def results_table() -> str:
    summary = {row["Model"]: row for row in read_rows(MAIN / "overall_summary_table.csv")}
    columns = [("Final P&L (USDT)", "Final PnL Mean", 3), ("Sharpe (60 s)", "Sharpe Mean", 3),
               ("Sortino", "Sortino Mean", 3), ("MaxDD mean (USDT)", "Max DD Mean", 4),
               ("CVaR₀.₂ of daily MaxDD (USDT)", "Max DD CVaR0.2 (days)", 4), ("P&L-to-MAP", "P&L-to-MAP Mean", 3),
               ("Mean \\|q\\| (DOGE)", "Mean |q| Mean", 2)]
    lines = ["| Model | " + " | ".join(title for title, _, _ in columns) + " |",
             "|---|" + "---|" * len(columns)]
    for label, key, name, _ in MODELS:
        row = summary[key]
        recomputed = empirical_ru_cvar(daily_maxdd(name), rho=B3_TAIL_MASS).value
        if not math.isclose(recomputed, float(row["Max DD CVaR0.2 (days)"]), rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{key}: CVaR from the daily values {recomputed} differs from the summary table")
        lines.append(f"| {label} | " + " | ".join(f"{float(row[field]):.{digits}f}" for _, field, digits in columns) + " |")
    lines += ["", "Means over the 23 seed-averaged test days (each day: 2 fill seeds × 20 paths, deterministic actions); "
              "the CVaR₀.₂ is",
              "the mean of the worst 20 % of those days (0.2 × 23 = 4.6 days: the worst four and 0.6 of the fifth).",
              "",
              "| Primary contrast (CVaR₀.₂ of daily MaxDD) | Δ (USDT) | p | Holm-adjusted p |", "|---|---|---|---|"]
    primary = [row for row in read_rows(MAIN / "paired_day_inference_primary.csv") if row["family"] == "primary"]
    for reference in ("B1", "B2"):
        row = next(r for r in primary if r["reference"] == reference and r["challenger"] == "B3")
        lines.append(f"| B3 − {reference} | {float(row['estimate']):+.4f} | {float(row['p_value']):.3f} | "
                     f"{float(row['holm_p']):.3f} |".replace("-0.", "−0."))
    lines += ["", "p: within-day label-swap randomization test, 10,000 draws; Holm over the two primary contrasts. "
              "Neither result is", "evidence for a difference, nor for equivalence."]
    return "\n".join(lines)


def figure(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")
    offsets = np.linspace(-0.24, 0.24, 23)  # the days left to right, Jan 7 to Jan 29
    for i, (label, _, name, colour) in enumerate(MODELS):
        values = daily_maxdd(name)
        estimate = empirical_ru_cvar(values, rho=B3_TAIL_MASS)
        tail = values >= estimate.nu
        ax.scatter(i + offsets[~tail], values[~tail], s=26, color=colour, alpha=0.45, edgecolors="#FFFFFF",
                   linewidths=0.8, zorder=3)
        ax.scatter(i + offsets[tail], values[tail], s=26, color=colour, edgecolors="#FFFFFF", linewidths=0.8, zorder=3)
        ax.plot([i - 0.3, i + 0.3], [estimate.value] * 2, color=colour, linewidth=2.5, solid_capstyle="round",
                zorder=4)
        ax.annotate(f"{estimate.value:.4f}", (i, estimate.value), xytext=(0, 4), textcoords="offset points",
                    va="bottom", ha="center", fontsize=9, color=TEXT, zorder=5)
    ax.set_xticks(range(len(MODELS)), [label for label, *_ in MODELS], color=TEXT)
    ax.set_xlim(-0.6, len(MODELS) - 0.4)
    ax.set_ylim(0, None)
    ax.set_ylabel("Daily maximum drawdown (USDT)", color=TEXT)
    ax.set_title("CVaR₀.₂ of the daily maximum drawdown on the 23 test days (Jan 7–29, 2025)", color=TEXT,
                 fontsize=11, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=TEXT)
    handles = [Line2D([], [], marker="o", linestyle="", color=MUTED, alpha=0.45, label="daily value"),
               Line2D([], [], marker="o", linestyle="", color=MUTED, label="day in the worst 20 %"),
               Line2D([], [], color=MUTED, linewidth=2.5, label="CVaR₀.₂ (value above)")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.5, labelcolor=TEXT)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="#FFFFFF", metadata={"Software": None})
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--figure", type=Path, default=FIGURE)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # the table holds subscripts and minus signs
    print(results_table())
    figure(args.figure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
