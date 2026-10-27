"""Size of the primary CVaR tests at n = 23 days under 8 null designs (DECISIONS D-14a; MODEL_SPEC §10).

The designs cross a cross-day AR(1) coefficient phi in {0, 0.5}, a within-day model correlation r in {0.5, 0.9}
and a lognormal sigma in {0.5, 1.0}. Model m in {A, B} on day d = 1..23 has X_d^m = exp(sigma (D_d + e_d^m)),
where D is a stationary Gaussian AR(1) with unit variance and (e_d^A, e_d^B) is bivariate normal with unit
variances and correlation r, independent across days. The two models are exchangeable within each day, so the
D-14a null holds and the population CVaR difference is 0.

Each replication runs, with the production code of cvar_mm/gym/statistical_inference.py, the two tests the thesis
reports on the CVaR_0.2 difference of the 23 values: the within-day label-swap randomization test (D-14a) and the
centred paired circular block bootstrap (block 3) with the interval that inverts it (D-14). The randomization test
rejects at p <= 0.05, the Monte Carlo test's level-0.05 rule; the bootstrap at p < 0.05, the test its interval
inverts (D-14), so its rejection rate is 1 minus the interval's coverage of 0. Each rate comes with its Monte Carlo
standard error sqrt(rate (1 - rate) / replications).

A numpy simulation with no market data: the only project code it runs is the two tests it measures. Every
replication draws from its own seeds, (SEED, design, replication), so the results do not depend on --workers or
--chunk.

    python scripts/analysis/cvar_test_size_study.py      # 2,000 replications; 2,000 + 2,000 draws each
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cvar_mm.gym.cvar_lagrangian import B3_TAIL_MASS  # noqa: E402
from cvar_mm.gym.statistical_inference import (  # noqa: E402
    paired_block_bootstrap_cvar_difference,
    paired_randomization_test_cvar,
)

PHIS = (0.0, 0.5)
CORRELATIONS = (0.5, 0.9)
SIGMAS = (0.5, 1.0)
DESIGNS = tuple((phi, correlation, sigma) for phi in PHIS for correlation in CORRELATIONS for sigma in SIGMAS)
DAYS = 23
ALPHA = 0.05
BLOCK_LENGTH = 3  # the D-14 primary block
SEED = 42
REPLICATIONS = 2_000
RANDOMIZATION_DRAWS = 2_000
BOOTSTRAP_DRAWS = 2_000
OUTPUT_DIR = REPO / "results" / "final" / "inference_size_study"
COLUMNS = (
    "design", "phi", "within_day_correlation", "sigma", "days", "rho", "alpha", "replications",
    "randomization_draws", "bootstrap_draws", "bootstrap_block_length", "seed",
    "randomization_rejection_rate", "randomization_rejection_mcse",
    "bootstrap_rejection_rate", "bootstrap_rejection_mcse",
    "bootstrap_ci_coverage", "bootstrap_ci_coverage_mcse",
)


def simulate_pair(rng: np.random.Generator, phi: float, correlation: float, sigma: float, days: int = DAYS):
    """One null sample: the two models' daily values, each of shape (days,), exchangeable within each day."""

    shocks = rng.standard_normal(days)
    day_effect = np.empty(days)
    day_effect[0] = shocks[0]
    for index in range(1, days):
        day_effect[index] = phi * day_effect[index - 1] + math.sqrt(1.0 - phi**2) * shocks[index]
    first, second = rng.standard_normal((2, days))
    error_a = first
    error_b = correlation * first + math.sqrt(1.0 - correlation**2) * second
    return np.exp(sigma * (day_effect + error_a)), np.exp(sigma * (day_effect + error_b))


def replication(design: int, index: int, draws: int, bootstrap_draws: int) -> tuple[float, float, float, float]:
    """Both tests on one null sample: (randomization p, bootstrap p, bootstrap CI low, bootstrap CI high)."""

    phi, correlation, sigma = DESIGNS[design]
    data_seeds, test_seeds = np.random.SeedSequence([SEED, design, index]).spawn(2)
    reference, challenger = simulate_pair(np.random.default_rng(data_seeds), phi, correlation, sigma)
    test_seed = int(test_seeds.generate_state(1)[0])
    randomization = paired_randomization_test_cvar(
        challenger, reference, rho=B3_TAIL_MASS, draws=draws, seed=test_seed
    )
    bootstrap = paired_block_bootstrap_cvar_difference(
        reference[:, None], challenger[:, None], rho=B3_TAIL_MASS, block_length=BLOCK_LENGTH,
        repetitions=bootstrap_draws, seed=test_seed,
    )
    return randomization["p_value"], bootstrap["p_value"], bootstrap["ci_low"], bootstrap["ci_high"]


def run_chunk(task: tuple[int, int, int, int, int]) -> list[tuple[float, float, float, float]]:
    design, start, stop, draws, bootstrap_draws = task
    return [replication(design, index, draws, bootstrap_draws) for index in range(start, stop)]


def rate(flags: np.ndarray) -> tuple[float, float]:
    """A proportion and its Monte Carlo standard error."""

    value = float(np.mean(flags))
    return value, math.sqrt(value * (1.0 - value) / len(flags))


def size_study(
    replications: int = REPLICATIONS,
    draws: int = RANDOMIZATION_DRAWS,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    *,
    workers: int = 1,
    chunk: int = 50,
) -> list[dict]:
    """One row per design: both tests' rejection rates at ALPHA and the bootstrap CI's coverage of 0."""

    if min(replications, draws, bootstrap_draws, workers, chunk) < 1:
        raise ValueError("replications, draws, bootstrap_draws, workers and chunk must be positive")
    tasks = [
        (design, start, min(start + chunk, replications), draws, bootstrap_draws)
        for design in range(len(DESIGNS)) for start in range(0, replications, chunk)
    ]
    if workers == 1:
        chunks = [run_chunk(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            chunks = list(pool.map(run_chunk, tasks))  # in task order, whatever the scheduling
    outcomes = {design: [] for design in range(len(DESIGNS))}
    for task, rows in zip(tasks, chunks):
        outcomes[task[0]].extend(rows)
    table = []
    for design, (phi, correlation, sigma) in enumerate(DESIGNS):
        randomization_p, bootstrap_p, low, high = np.asarray(outcomes[design], dtype=float).T
        randomization, randomization_se = rate(randomization_p <= ALPHA)  # the Monte Carlo test's level-alpha rule
        bootstrap, bootstrap_se = rate(bootstrap_p < ALPHA)  # D-14: the test its CI inverts, so 1 - coverage
        coverage, coverage_se = rate((low <= 0.0) & (0.0 <= high))
        table.append(dict(zip(COLUMNS, (
            design + 1, phi, correlation, sigma, DAYS, B3_TAIL_MASS, ALPHA, replications,
            draws, bootstrap_draws, BLOCK_LENGTH, SEED,
            randomization, randomization_se, bootstrap, bootstrap_se, coverage, coverage_se,
        ))))
    return table


def booktabs_table(table: list[dict]) -> str:
    """The thesis appendix table (booktabs, LaTeX), rates with their Monte Carlo SE in parentheses."""

    first = table[0]

    def cell(row: dict, value: str, mcse: str) -> str:
        return f"{row[value]:.3f} ({row[mcse]:.3f})"

    lines = [
        "% Generated by scripts/analysis/cvar_test_size_study.py (DECISIONS D-14a); do not edit by hand.",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Size of the primary tests of $\Delta\,\mathrm{CVaR}_{0.2}$ of daily maximum drawdown under eight "
        r"null designs: rejection rates at the nominal 5\,\% level of the within-day label-swap randomization test "
        rf"({first['randomization_draws']:,} draws) and of the centred paired circular block bootstrap "
        rf"(block {first['bootstrap_block_length']}, {first['bootstrap_draws']:,} draws), and the coverage of the "
        r"bootstrap's unadjusted 95\,\% interval. $X_d^m = \exp(\sigma(D_d + e_d^m))$ with a cross-day AR(1) day "
        r"effect $D$ (coefficient $\phi$, unit variance) and within-day errors of correlation $r$; "
        rf"$n = {first['days']}$ days, $\rho = {first['rho']}$, {first['replications']:,} replications per design. "
        r"Monte Carlo standard errors in parentheses.}",
        r"  \label{tab:cvar-test-size-study}",
        r"  \begin{tabular}{rrrccc}",
        r"    \toprule",
        r"    $\phi$ & $r$ & $\sigma$ & Randomization rejection & Bootstrap rejection & Bootstrap CI coverage \\",
        r"    \midrule",
    ]
    for row in table:
        lines.append(
            f"    {row['phi']:.1f} & {row['within_day_correlation']:.1f} & {row['sigma']:.1f} & "
            f"{cell(row, 'randomization_rejection_rate', 'randomization_rejection_mcse')} & "
            f"{cell(row, 'bootstrap_rejection_rate', 'bootstrap_rejection_mcse')} & "
            f"{cell(row, 'bootstrap_ci_coverage', 'bootstrap_ci_coverage_mcse')} \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def git_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "uncommitted_code": git(
            "status", "--porcelain", "--", "scripts/analysis/cvar_test_size_study.py", "cvar_mm/gym/statistical_inference.py"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Size study of the D-14a randomization test and the D-14 bootstrap.")
    parser.add_argument("--replications", type=int, default=REPLICATIONS)
    parser.add_argument("--draws", type=int, default=RANDOMIZATION_DRAWS, help="randomization draws per replication")
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--chunk", type=int, default=50, help="replications per task")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    started = time.perf_counter()
    table = size_study(args.replications, args.draws, args.bootstrap_draws, workers=args.workers, chunk=args.chunk)
    elapsed = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "size_study.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({name: repr(value) if isinstance(value, float) else value for name, value in row.items()}
                         for row in table)
    (args.output_dir / "size_study_table.tex").write_text(booktabs_table(table), encoding="utf-8", newline="\n")
    metadata = {
        "script": "scripts/analysis/cvar_test_size_study.py",
        "decision": "D-14a size study",
        "designs": [dict(zip(("phi", "within_day_correlation", "sigma"), design)) for design in DESIGNS],
        "days": DAYS, "rho": B3_TAIL_MASS, "alpha": ALPHA, "seed": SEED,
        "replications": args.replications, "randomization_draws": args.draws, "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_block_length": BLOCK_LENGTH,
        "rejection_rule": "randomization: p <= alpha (Phipson & Smyth 2010); bootstrap: p < alpha, the D-14 test "
                          "its CI inverts, so its rejection rate is 1 - coverage; the CI covers when "
                          "ci_low <= 0 <= ci_high. With 2,000 draws p is a multiple of 1/2001, and the two rules "
                          "coincide",
        "workers": args.workers, "chunk": args.chunk, "wall_seconds": round(elapsed, 1),
        "python_version": platform.python_version(), "numpy_version": np.__version__, "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **git_state(),
    }
    (args.output_dir / "size_study_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")
    worst = max(row["randomization_rejection_rate"] for row in table)
    print(f"Wrote {args.output_dir} in {elapsed:.0f} s; largest randomization rejection rate {worst:.4f}")


if __name__ == "__main__":
    main()
