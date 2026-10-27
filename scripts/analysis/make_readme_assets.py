"""The README's results table and figures, from results/final/.

    python scripts/analysis/make_readme_assets.py [--out-dir docs/figures]

Prints the README results table in Markdown (the summary over the 23 test days and the two primary contrasts) and
writes two figures, each in a light and a dark version (the README picks one with the reader's GitHub theme):

- results.png / results-dark.png: the CVaR0.2 of the seed-averaged daily maximum drawdown per model, with the 23
  daily values as points, and each model's CVaR0.2 against its mean Sharpe ratio;
- b3_training.png / b3_training-dark.png: B3's constraint statistic (as a multiple of the threshold d) and its
  Lagrange multiplier over the 10 training updates, for the three dual step sizes of the eta profiling.

Each model's CVaR is recomputed from its daily values with the thesis code (empirical_ru_cvar) and must equal the
summary table's. docs/figures/overview.svg (the schematic at the top of the README) is drawn by hand, not here.
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

FINAL = REPO / "results" / "final"
MAIN = FINAL / "main"
B3_DUAL = FINAL / "profiling" / "b3_dual"
OUT_DIR = REPO / "docs" / "figures"

# (label, summary-table row, per-day file, light colour, dark colour), in the thesis order B0 | B1 | B2 | B3 with
# the B1+ control last. Colours: the thesis's model palette (Okabe-Ito based; B0 is the neutral baseline ink), with
# dark-mode steps checked against GitHub's dark surface. Identity is always also given by a text label.
MODELS = [
    ("B0", "B0", "b0_test_results.csv", "#333333", "#C9D1D9"),
    ("B1", "B1", "b1_test_results.csv", "#0072B2", "#3F7FE8"),
    ("B2", "B2", "b2_test_results.csv", "#E69F00", "#BF8300"),
    ("B3", "B3", "b3_test_results.csv", "#009E73", "#109C76"),
    ("B1⁺ (control)", "B1PLUS", "b1plus_test_results.csv", "#56B4E9", "#3E9BDF"),
]
THEMES = {
    "light": {"text": "#1F2328", "muted": "#59636E", "grid": "#E4E8EC", "axis": "#9AA4AE", "risk": "#D55E00",
              "ring": "#FFFFFF", "other": "#AEB6BF", "suffix": ""},
    "dark": {"text": "#E6EDF3", "muted": "#9198A1", "grid": "#262C35", "axis": "#4A535D", "risk": "#F07A3A",
             "ring": "#0D1117", "other": "#5A636D", "suffix": "-dark"},
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def daily_maxdd(name: str) -> np.ndarray:
    rows = sorted(read_rows(MAIN / name), key=lambda row: row["Day"])
    if len(rows) != 23:
        raise ValueError(f"{name}: expected 23 test days, found {len(rows)}")
    return np.array([float(row["Max DD"]) for row in rows])


def summary_rows() -> dict[str, dict[str, str]]:
    return {row["Model"]: row for row in read_rows(MAIN / "overall_summary_table.csv")}


def checked_cvar(key: str, name: str, summary: dict[str, dict[str, str]]):
    estimate = empirical_ru_cvar(daily_maxdd(name), rho=B3_TAIL_MASS)
    if not math.isclose(estimate.value, float(summary[key]["Max DD CVaR0.2 (days)"]), rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{key}: CVaR from the daily values {estimate.value} differs from the summary table")
    return estimate


def results_table() -> str:
    summary = summary_rows()
    columns = [("Sharpe (60 s)", "Sharpe Mean", 3), ("CVaR₀.₂ of daily MaxDD", "Max DD CVaR0.2 (days)", 4),
               ("Mean daily MaxDD", "Max DD Mean", 4), ("Final P&L", "Final PnL Mean", 3),
               ("Mean \\|q\\| (DOGE)", "Mean |q| Mean", 2)]
    lines = ["| Model | " + " | ".join(title for title, _, _ in columns) + " |",
             "|---|" + "---:|" * len(columns)]
    for label, key, name, _, _ in MODELS:
        checked_cvar(key, name, summary)
        row = summary[key]
        cells = [f"{float(row[field]):.{digits}f}" for _, field, digits in columns]
        if key == "B0":
            cells = [f"**{cell}**" for cell in cells[:2]] + cells[2:]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "Means over the 23 seed-averaged test days (each day: 2 fill seeds × 20 paths, deterministic actions); "
              "drawdowns and P&L in USDT.",
              "The CVaR₀.₂ is the mean of the worst 20 % of the days (0.2 × 23 = 4.6 days: the worst four and 0.6 of "
              "the fifth).",
              "",
              "| Primary contrast (CVaR₀.₂ of daily MaxDD) | Δ (USDT) | p | Holm-adjusted p |", "|---|---:|---:|---:|"]
    primary = [row for row in read_rows(MAIN / "paired_day_inference_primary.csv") if row["family"] == "primary"]
    for reference in ("B1", "B2"):
        row = next(r for r in primary if r["reference"] == reference and r["challenger"] == "B3")
        lines.append(f"| B3 − {reference} | {float(row['estimate']):+.4f} | {float(row['p_value']):.3f} | "
                     f"{float(row['holm_p']):.3f} |".replace("-0.", "−0."))
    lines += ["", "p: within-day label-swap randomization test, 10,000 draws; Holm over the two primary contrasts. "
              "Neither result is", "evidence for a difference, nor for equivalence."]
    return "\n".join(lines)


def _style_axis(ax, theme: dict[str, str]) -> None:
    ax.grid(axis="y", color=theme["grid"], linewidth=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
    ax.tick_params(colors=theme["axis"], labelcolor=theme["muted"], labelsize=10.5)
    ax.set_facecolor("none")


def _title(ax, title: str, subtitle: str, theme: dict[str, str]) -> None:
    ax.text(0, 1.13, title, transform=ax.transAxes, fontsize=13.5, fontweight="bold", color=theme["text"],
            ha="left", va="bottom")
    ax.text(0, 1.04, subtitle, transform=ax.transAxes, fontsize=10.5, color=theme["muted"], ha="left", va="bottom")


def results_figure(mode: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme = THEMES[mode]
    colour = 3 if mode == "light" else 4
    summary = summary_rows()
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.6), dpi=200, gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_alpha(0)

    offsets = np.linspace(-0.22, 0.22, 23)  # the 23 days left to right, Jan 7 to Jan 29
    for i, model in enumerate(MODELS):
        label, key, name, c = model[0], model[1], model[2], model[colour]
        values = daily_maxdd(name)
        estimate = checked_cvar(key, name, summary)
        tail = values >= estimate.nu
        left.scatter(i + offsets[~tail], values[~tail], s=30, color=c, alpha=0.38, edgecolors=theme["ring"],
                     linewidths=1.0, zorder=3)
        left.scatter(i + offsets[tail], values[tail], s=34, color=c, edgecolors=theme["ring"], linewidths=1.0,
                     zorder=3)
        left.plot([i - 0.25, i + 0.25], [estimate.value] * 2, color=c, linewidth=3.2, solid_capstyle="round",
                  zorder=4)
        left.annotate(f"{estimate.value:.3f}", (i, estimate.value), xytext=(0, 6), textcoords="offset points",
                      va="bottom", ha="center", fontsize=10, color=theme["text"], zorder=5,
                      bbox={"boxstyle": "round,pad=0.15", "facecolor": theme["ring"], "edgecolor": "none",
                            "alpha": 0.85})
    left.set_xticks(range(len(MODELS)), [m[0].replace(" (control)", "\n(control)") for m in MODELS])
    left.set_xlim(-0.55, len(MODELS) - 0.25)
    left.set_ylim(0, 0.37)
    left.set_ylabel("Daily maximum drawdown (USDT)", color=theme["muted"], fontsize=10.5)
    _style_axis(left, theme)
    left.tick_params(axis="x", labelcolor=theme["text"], labelsize=11.5, length=0)
    _title(left, "Tail risk on the 23 test days",
           "Dots: one test day each (solid: the worst 20 %). Bar: CVaR₀.₂. Lower is better.", theme)

    for model in MODELS:
        label, key, c = model[0], model[1], model[colour]
        cvar = float(summary[key]["Max DD CVaR0.2 (days)"])
        sharpe = float(summary[key]["Sharpe Mean"])
        right.scatter([cvar], [sharpe], s=140, color=c, edgecolors=theme["ring"], linewidths=2.0, zorder=4)
        # Label positions (data coordinates): the four PPO models sit close together, so their labels fan out
        # with thin leader lines.
        place = {"B0": (0.062, 0.413), "B1": (0.118, 0.265), "B2": (0.098, 0.215), "B3": (0.150, 0.300),
                 "B1PLUS": (0.150, 0.045)}[key]
        leader = None if key == "B0" else {"arrowstyle": "-", "color": theme["axis"], "linewidth": 0.9,
                                            "shrinkA": 2, "shrinkB": 7}
        right.annotate(label, (cvar, sharpe), xytext=place, textcoords="data", fontsize=11, color=theme["text"],
                       ha="left", va="center", arrowprops=leader, zorder=5)
    right.set_xlim(0, 0.2)
    right.set_ylim(0, 0.46)
    right.set_xlabel("CVaR₀.₂ of the daily maximum drawdown (USDT)", color=theme["muted"], fontsize=10.5)
    right.set_ylabel("Mean Sharpe ratio (60 s)", color=theme["muted"], fontsize=10.5)
    _style_axis(right, theme)
    right.grid(axis="x", color=theme["grid"], linewidth=0.9)
    right.annotate("", xy=(0.012, 0.44), xytext=(0.052, 0.36),
                   arrowprops={"arrowstyle": "-|>", "color": theme["muted"], "linewidth": 1.2})
    right.text(0.056, 0.355, "better: less tail risk,\nhigher Sharpe", fontsize=9.5, color=theme["muted"],
               ha="left", va="top")
    _title(right, "Return against tail risk", "One point per model, over the 23 test days.", theme)

    fig.tight_layout(w_pad=4.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, transparent=True, metadata={"Software": None})
    plt.close(fig)


def b3_training_figure(mode: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme = THEMES[mode]
    b3 = MODELS[3][3] if mode == "light" else MODELS[3][4]
    runs = {}
    for eta in ("0.5", "2.0", "8.0"):
        rows = read_rows(B3_DUAL / f"history_eta{eta}.csv")
        if [int(r["update"]) for r in rows] != list(range(1, 11)):
            raise ValueError(f"history_eta{eta}.csv: expected updates 1-10")
        runs[eta] = (np.array([float(r["ru_cvar"]) / float(r["threshold_d"]) for r in rows]),
                     np.array([float(r["lambda"]) for r in rows]))
    updates = np.arange(1, 11)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 3.8), dpi=200)
    fig.patch.set_alpha(0)
    for eta, (ratio, lam) in runs.items():
        selected = eta == "8.0"
        style = {"color": b3 if selected else theme["other"], "linewidth": 2.4 if selected else 1.6,
                 "marker": "o" if selected else None, "markersize": 6, "markeredgecolor": theme["ring"],
                 "markeredgewidth": 1.2, "zorder": 4 if selected else 3}
        name = f"η = {eta.rstrip('0').rstrip('.')}" + (" (the final B3)" if selected else "")
        # End labels; on the left the three runs end close together, so the labels are spaced apart.
        nudge = {"0.5": 9, "2.0": -3, "8.0": -6}[eta]
        for ax, series, dy in ((left, ratio, nudge), (right, lam, 0)):
            ax.plot(updates, series, **style)
            ax.annotate(name, (10, series[-1]), xytext=(8, dy), textcoords="offset points", va="center",
                        fontsize=10, color=theme["text"] if selected else theme["muted"])
    left.axhline(1.0, color=theme["risk"], linewidth=1.6, linestyle=(0, (5, 3)), zorder=2)
    left.text(1, 1.012, "threshold d: the constraint holds below this line", color=theme["risk"], fontsize=10,
              va="bottom", ha="left")
    left.set_ylim(0.9, 1.45)
    left.set_ylabel("Ĥ / d", color=theme["muted"], fontsize=10.5)
    right.set_yscale("log")
    right.set_yticks([0.2, 0.5, 1, 2, 5, 10, 20], ["0.2", "0.5", "1", "2", "5", "10", "20"])
    right.minorticks_off()
    right.set_ylabel("λ (log scale)", color=theme["muted"], fontsize=10.5)
    for ax in (left, right):
        ax.set_xlim(0.6, 12.6)
        ax.set_xticks(updates)
        ax.set_xlabel("Training update", color=theme["muted"], fontsize=10.5)
        _style_axis(ax, theme)
        ax.spines["bottom"].set_bounds(1, 10)
    _title(left, "B3's tail-risk statistic during training",
           "RU-CVaR₀.₂ of the full-day max drawdown on Jan 1–6, as a multiple of d.", theme)
    _title(right, "Its Lagrange multiplier",
           "λ grows every update, yet the statistic stays above d.", theme)
    fig.tight_layout(w_pad=4.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, transparent=True, metadata={"Software": None})
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # the table holds subscripts and minus signs
    print(results_table())
    for mode, theme in THEMES.items():
        results_figure(mode, args.out_dir / f"results{theme['suffix']}.png")
        b3_training_figure(mode, args.out_dir / f"b3_training{theme['suffix']}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
