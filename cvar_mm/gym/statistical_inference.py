"""Paired inference with market day as the sampling unit (DECISIONS D-14, D-14a, D-16, D-18; MODEL_SPEC §10)."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from cvar_mm.gym.cvar_lagrangian import B3_TAIL_MASS, empirical_ru_cvar, rockafellar_uryasev_value

# Pre-specified families (DECISIONS D-14, D-14a). A contrast is (reference, challenger); its effect is
# challenger - reference.
PRIMARY_CONTRASTS = (("B1", "B3"), ("B2", "B3"))
SECONDARY_CONTRASTS = (("B1", "B0"), ("B1", "B2"), ("B1", "B3"), ("B2", "B3"))
SECONDARY_METRICS = (
    "Sharpe", "Sortino", "Max DD", "P&L-to-MAP", "Final PnL", "Mean |q|", "Near Cap Fraction",
)
# Descriptive rows, outside both Holm families and not multiplicity-adjusted: B3 vs B0 on the primary statistic
# (D-16); B3 vs B1+ and B1+ vs B1 on the primary statistic and on the secondary metrics (D-18).
PRIMARY_DESCRIPTIVE_CONTRASTS = (("B0", "B3"), ("B1PLUS", "B3"), ("B1", "B1PLUS"))
SECONDARY_DESCRIPTIVE_CONTRASTS = (("B1PLUS", "B3"), ("B1", "B1PLUS"))
PRIMARY_METRIC = "Max DD"
RANDOMIZATION_DRAWS = 10_000
EXACT_RANDOMIZATION_MAX_DAYS = 16
# The `test` column: what produced each row's p.
DAY_SWAP_TEST = "within-day label-swap randomization"
PATH_SWAP_TEST = "within-day label-swap randomization, whole days of paths"
INTERVAL_ONLY = "none (interval only)"
WILCOXON_TEST = "Pratt-Wilcoxon signed-rank"


def paired_day_differences(
    reference: pd.DataFrame,
    challenger: pd.DataFrame,
    metric: str,
) -> tuple[list[str], np.ndarray]:
    """Align one aggregate observation per market day and return B-A."""

    for name, frame in (("reference", reference), ("challenger", challenger)):
        if "Day" not in frame or metric not in frame:
            raise ValueError(f"{name} frame must contain Day and {metric}")
        if frame["Day"].astype(str).duplicated().any():
            raise ValueError(f"{name} contains multiple rows per market day; aggregate fills first")
    left = reference.assign(Day=reference["Day"].astype(str)).set_index("Day")
    right = challenger.assign(Day=challenger["Day"].astype(str)).set_index("Day")
    if set(left.index) != set(right.index):
        raise ValueError("Paired inference requires identical market-day sets")
    days = sorted(left.index)
    differences = right.loc[days, metric].to_numpy(float) - left.loc[days, metric].to_numpy(float)
    return days, differences


def _circular_block_indices(rng: np.random.Generator, n: int, block_length: int) -> np.ndarray:
    """One circular moving-block resample of n consecutive days."""

    starts = rng.integers(0, n, size=int(np.ceil(n / block_length)))
    return ((starts[:, None] + np.arange(block_length)) % n).reshape(-1)[:n]


def circular_block_bootstrap_mean_ci(
    differences: np.ndarray,
    *,
    block_length: int = 5,
    repetitions: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Circular moving-block bootstrap CI for the mean paired day effect."""

    values = np.asarray(differences, dtype=float)
    n = len(values)
    if n < 2 or not 1 <= block_length <= n or repetitions <= 0:
        raise ValueError("Need n>=2, 1<=block_length<=n, and positive repetitions")
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions)
    for repetition in range(repetitions):
        samples[repetition] = float(np.mean(values[_circular_block_indices(rng, n, block_length)]))
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(samples, tail)), float(np.quantile(samples, 1.0 - tail))


def pratt_wilcoxon_pvalue(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=float)
    if not np.any(values != 0):
        return 1.0
    return float(wilcoxon(values, zero_method="pratt", alternative="two-sided").pvalue)


def hodges_lehmann(differences: np.ndarray) -> float:
    """Median of the Walsh averages (d_i + d_j) / 2, i <= j: the location effect matching Wilcoxon."""

    values = np.asarray(differences, dtype=float)
    first, second = np.triu_indices(len(values))
    return float(np.median((values[first] + values[second]) / 2.0))


def signed_rank_critical_value(n: int, confidence: float = 0.95) -> tuple[int, float]:
    """M4: the exact Wilcoxon signed-rank critical value c of a two-sided interval, and its coverage (Bauer 1972).

    Under H0 (n continuous differences, symmetric about the centre) each of the 2^n subsets of the ranks 1..n is
    equally likely to be the set of positive ranks, so the null law of the signed-rank sum T+ follows by counting
    subsets. c is the largest integer with P(T+ <= c) <= (1 - confidence)/2 (exact rational arithmetic), and
    [W(c+1), W(M-c)] of the M = n(n+1)/2 ordered Walsh averages covers the centre with probability
    1 - 2 P(T+ <= c). For n = 23 and 95 %: c = 73, coverage 0.9516.
    """

    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    tail = (1 - Fraction(str(confidence))) / 2  # 1/40 for 0.95, exactly
    if not 0 < tail < Fraction(1, 2):
        raise ValueError(f"confidence must lie strictly between 0 and 1, got {confidence!r}")
    counts = [1] + [0] * (n * (n + 1) // 2)  # counts[s]: subsets of {1..n} with rank sum s
    for rank in range(1, n + 1):
        for total in range(len(counts) - 1, rank - 1, -1):
            counts[total] += counts[total - rank]
    cumulative, critical = 0, -1
    for total, count in enumerate(counts):
        if Fraction(cumulative + count, 2**n) > tail:
            break
        cumulative, critical = cumulative + count, total
    if critical < 0:
        raise ValueError(f"n = {n} days is too few for an exact {confidence} signed-rank interval")
    return critical, float(1 - Fraction(2 * cumulative, 2**n))


def hodges_lehmann_ci(differences: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    """M4: the Hodges-Lehmann interval [W(c+1), W(M-c)] of the ordered Walsh averages at the exact signed-rank
    critical value c (Bauer 1972), with its exact coverage. The coverage assumes continuous differences that are
    symmetric about the centre; ties and zero differences make it approximate. The interval contains
    hodges_lehmann(differences)."""

    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("differences must be a finite one-dimensional array")
    critical, coverage = signed_rank_critical_value(len(values), confidence)
    first, second = np.triu_indices(len(values))
    walsh = np.sort((values[first] + values[second]) / 2.0)
    return float(walsh[critical]), float(walsh[len(walsh) - 1 - critical]), coverage


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Holm family-wise p-value adjustment preserving input order."""

    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(1.0, values[order] * (len(values) - np.arange(len(values))))
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def ru_cvar(values: np.ndarray, rho: float = B3_TAIL_MASS) -> float:
    """CVaR_rho of all the values pooled, by empirical_ru_cvar on their sorted copy.

    Sorting first makes the computed value a function of the multiset alone, so a resample or a label swap
    that leaves a model's values unchanged reproduces its CVaR bit for bit.
    """

    return empirical_ru_cvar(np.sort(np.asarray(values, dtype=float), axis=None), rho=rho).value


def ru_cvar_of_size(size: int, rho: float = B3_TAIL_MASS):
    """``ru_cvar`` for many samples of one size, without np.quantile's per-call cost.

    empirical_ru_cvar takes nu = np.quantile(values, 1 - rho, method="higher"): the sorted sample's element at an
    index that depends on the sample size and rho alone. That index is found once, here, by the same np.quantile
    call on 0..size-1; each call then evaluates rockafellar_uryasev_value, empirical_ru_cvar's own value function,
    at that element of the sorted sample. The result equals ru_cvar bit for bit; both tests below check it on
    their observed samples (``_observed_difference``).
    """

    index = int(np.quantile(np.arange(size, dtype=float), 1.0 - rho, method="higher"))

    def cvar(values: np.ndarray) -> float:
        ordered = np.sort(values, axis=None)
        if ordered.size != size:
            raise ValueError(f"expected {size} values, got {ordered.size}")
        return rockafellar_uryasev_value(ordered, nu=float(ordered[index]), rho=rho)

    return cvar


def _observed_difference(challenger: np.ndarray, reference: np.ndarray, rho: float):
    """The observed CVaR difference by empirical_ru_cvar, and the per-draw CVaR for samples of this size,
    checked to reproduce empirical_ru_cvar on both observed samples."""

    cvar = ru_cvar_of_size(challenger.size, rho)
    observed = (ru_cvar(challenger, rho), ru_cvar(reference, rho))
    if (cvar(challenger), cvar(reference)) != observed:
        raise RuntimeError("ru_cvar_of_size does not reproduce empirical_ru_cvar on these samples")
    return observed[0] - observed[1], cvar


def paired_randomization_test_cvar(
    challenger: np.ndarray,
    reference: np.ndarray,
    *,
    rho: float = B3_TAIL_MASS,
    draws: int = RANDOMIZATION_DRAWS,
    seed: int = 42,
    exact: bool = False,
) -> dict[str, float]:
    """D-14a: the within-day label-swap randomization test of CVaR_rho(challenger) - CVaR_rho(reference).

    Inputs are (days,) or (days, m) arrays in a common day order: one seed-averaged value per day, or a day's m
    full-day paths. H0: within each day the two models' outcomes are exchangeable. Each draw gives every day an
    independent fair coin, swaps the two models' values on the days it marks (a day's m paths together), and
    recomputes the difference D* with empirical_ru_cvar (``ru_cvar``, by way of ``ru_cvar_of_size``). The
    two-sided p is (1 + #{|D*| >= |D|}) / (draws + 1) (Phipson & Smyth 2010); the draws come from numpy's
    default_rng(seed), so one seed gives one p, bit for bit. exact=True enumerates all 2^n swaps (n <= 16) instead
    and returns #{|D*| >= |D|} / 2^n, the identity included.
    """

    ch = np.asarray(challenger, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if ch.shape != ref.shape or ch.ndim not in (1, 2) or ch.shape[0] < 2 or 0 in ch.shape:
        raise ValueError("challenger and reference must be (days,) or (days, m) arrays of one shape, with >= 2 days")
    if not (np.isfinite(ch).all() and np.isfinite(ref).all()):
        raise ValueError("the randomization test needs finite values")
    days = ch.shape[0]
    if exact:
        if days > EXACT_RANDOMIZATION_MAX_DAYS:
            raise ValueError(f"exact=True enumerates 2^n swaps; n = {days} days > {EXACT_RANDOMIZATION_MAX_DAYS}")
        swaps = ((np.arange(2**days)[:, None] >> np.arange(days)) & 1).astype(bool)
    else:
        if isinstance(draws, bool) or not isinstance(draws, (int, np.integer)) or draws < 1:
            raise ValueError(f"draws must be a positive integer, got {draws!r}")
        swaps = np.random.default_rng(seed).integers(0, 2, size=(draws, days)).astype(bool)
    estimate, cvar = _observed_difference(ch, ref, rho)
    shape = (days,) + (1,) * (ch.ndim - 1)
    null = np.empty(len(swaps))
    for index, swap in enumerate(swaps):
        swapped = swap.reshape(shape)
        null[index] = cvar(np.where(swapped, ref, ch)) - cvar(np.where(swapped, ch, ref))
    extreme = int(np.count_nonzero(np.abs(null) >= abs(estimate)))
    p_value = extreme / len(swaps) if exact else (1 + extreme) / (len(swaps) + 1)
    return {"estimate": float(estimate), "p_value": float(p_value), "draws": int(len(swaps)), "exact": bool(exact)}


def paired_block_bootstrap_cvar_difference(
    reference: np.ndarray,
    challenger: np.ndarray,
    *,
    rho: float = B3_TAIL_MASS,
    block_length: int = 3,
    repetitions: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """D-14: CVaR_rho(challenger) - CVaR_rho(reference) of daily MaxDD, paired over days.

    Inputs are (days, m) arrays in a common day order: m = 1 for seed-averaged
    days, m = paths for full-day episodes. Each repetition resamples whole days
    in circular blocks, with the same indices for both models. The p-value is
    the centred bootstrap p = (1 + #{|D* - D| >= |D|}) / (B + 1) (Hall & Wilson
    1991). The CI is the interval that inverts this test, D +/- the r-th
    largest |D* - D| with r = ceil((1 - confidence)(B + 1)) - 1, so it
    excludes 0 exactly when p < 1 - confidence. Since D-14a the test is not the
    primary one: the CI is reported as an unadjusted descriptive interval (M3).
    """

    ref = np.asarray(reference, dtype=float)
    ch = np.asarray(challenger, dtype=float)
    if ref.ndim != 2 or ref.shape != ch.shape:
        raise ValueError("reference and challenger must be (days, m) arrays of the same shape")
    days = ref.shape[0]
    if days < 2 or not 1 <= block_length <= days or repetitions <= 0:
        raise ValueError("Need >= 2 days, 1 <= block_length <= days, and positive repetitions")
    rank = math.ceil(round((1.0 - confidence) * (repetitions + 1), 9)) - 1
    if rank < 1:
        raise ValueError("Too few bootstrap repetitions for the requested confidence")

    estimate, cvar = _observed_difference(ch, ref, rho)

    def cvar_difference(rows: np.ndarray) -> float:
        return cvar(ch[rows]) - cvar(ref[rows])

    rng = np.random.default_rng(seed)
    deviations = np.empty(repetitions)
    for repetition in range(repetitions):
        deviations[repetition] = abs(
            cvar_difference(_circular_block_indices(rng, days, block_length)) - estimate
        )
    half_width = float(np.sort(deviations)[::-1][rank - 1])
    return {
        "estimate": float(estimate),
        "ci_low": float(estimate - half_width),
        "ci_high": float(estimate + half_width),
        "p_value": float((1 + np.count_nonzero(deviations >= abs(estimate))) / (repetitions + 1)),
    }


def day_matrix(frame: pd.DataFrame, metric: str, days: list[str]) -> np.ndarray:
    """(days, m) values of a metric with m rows per market day (1 per day, or one per episode)."""

    values = frame.assign(Day=frame["Day"].astype(str)).groupby("Day")[metric]
    counts = values.size()
    if set(counts.index) != set(days):
        raise ValueError("Every model needs the same market days")
    if counts.nunique() != 1:
        raise ValueError("Every market day needs the same number of episodes")
    return np.vstack([values.get_group(day).to_numpy(float) for day in days])


def primary_cvar_inference_table(
    day_frames: dict[str, pd.DataFrame],
    episode_frames: dict[str, pd.DataFrame],
    *,
    block_length: int = 3,
    sensitivity_block_lengths: tuple[int, ...] = (1, 2, 4),
    repetitions: int = 10_000,
    draws: int = RANDOMIZATION_DRAWS,
    seed: int = 42,
    descriptive_contrasts: tuple[tuple[str, str], ...] = PRIMARY_DESCRIPTIVE_CONTRASTS,
) -> pd.DataFrame:
    """The primary statistic, CVaR_0.2 of daily MaxDD: the primary family and its sensitivity and descriptive rows.

    - ``primary`` (D-14, D-14a): {B3 vs B1, B3 vs B2} on the 23 seed-averaged days (F-07). p is the within-day
      label-swap randomization test's (``draws``, ``seed``); Holm runs over these two rows only. The paired
      circular block-bootstrap CI (``block_length``) is a descriptive interval, unadjusted (M3).
    - ``sensitivity`` (no Holm): the same CI at each of ``sensitivity_block_lengths``, with no test; and the
      path-level CVaR over every full-day path of each day, tested with whole-day swaps, its CI at
      ``block_length``.
    - ``descriptive`` (no Holm, not multiplicity-adjusted): ``descriptive_contrasts`` on the primary statistic,
      tested and bounded like the primary rows (D-16: B3 vs B0; D-18: B3 vs B1+ and B1+ vs B1).
    """

    def row(family, reference, challenger, metric, ref, ch, *, length, test, randomization=None):
        bootstrap = paired_block_bootstrap_cvar_difference(
            ref, ch, block_length=length, repetitions=repetitions, seed=seed
        )
        if randomization is not None and randomization["estimate"] != bootstrap["estimate"]:
            raise RuntimeError("The randomization test and the bootstrap disagree on the CVaR difference")
        return {
            "family": family,
            "reference": reference,
            "challenger": challenger,
            "metric": metric,
            "market_days": ref.shape[0],
            "rows_per_day": ref.shape[1],
            "estimate": bootstrap["estimate"],
            "test": test,
            "draws": np.nan if randomization is None else randomization["draws"],
            "p_value": np.nan if randomization is None else randomization["p_value"],
            "holm_p": np.nan,
            "ci_unadjusted_95_low": bootstrap["ci_low"],
            "ci_unadjusted_95_high": bootstrap["ci_high"],
            "ci_block_length_days": length,
            "ci_repetitions": repetitions,
        }

    def matrices(frames, reference, challenger, days):
        ref, ch = (day_matrix(frames[name], PRIMARY_METRIC, days) for name in (reference, challenger))
        if ref.shape != ch.shape:
            raise ValueError(f"{reference} and {challenger} need the same number of rows per market day")
        return ref, ch

    primary, sensitivity, descriptive = [], [], []
    for family, contrasts in (("primary", PRIMARY_CONTRASTS), ("descriptive", descriptive_contrasts)):
        for reference, challenger in contrasts:
            days, _ = paired_day_differences(day_frames[reference], day_frames[challenger], PRIMARY_METRIC)
            ref, ch = matrices(day_frames, reference, challenger, days)
            randomization = paired_randomization_test_cvar(ch, ref, draws=draws, seed=seed)
            rows = primary if family == "primary" else descriptive
            rows.append(row(family, reference, challenger, "Max DD CVaR0.2 (days)", ref, ch,
                            length=block_length, test=DAY_SWAP_TEST, randomization=randomization))
            if family != "primary":
                continue
            for length in sensitivity_block_lengths:
                sensitivity.append(row("sensitivity", reference, challenger, "Max DD CVaR0.2 (days)", ref, ch,
                                       length=length, test=INTERVAL_ONLY))
            ref_paths, ch_paths = matrices(episode_frames, reference, challenger, days)
            randomization = paired_randomization_test_cvar(ch_paths, ref_paths, draws=draws, seed=seed)
            sensitivity.append(row("sensitivity", reference, challenger, "Max DD CVaR0.2 (episodes)", ref_paths,
                                   ch_paths, length=block_length, test=PATH_SWAP_TEST, randomization=randomization))
    table = pd.DataFrame(primary + sensitivity + descriptive)
    is_primary = table["family"] == "primary"
    table.loc[is_primary, "holm_p"] = holm_adjust(table.loc[is_primary, "p_value"].to_numpy())
    return table


def paired_inference_table(
    frames: dict[str, pd.DataFrame],
    *,
    contrasts: tuple[tuple[str, str], ...] = SECONDARY_CONTRASTS,
    descriptive_contrasts: tuple[tuple[str, str], ...] = SECONDARY_DESCRIPTIVE_CONTRASTS,
    metrics: tuple[str, ...] = SECONDARY_METRICS,
    block_length: int = 5,
    bootstrap_repetitions: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Secondary family (D-14, M4) and its descriptive rows (D-18), from paired day differences.

    Each row has the Pratt-Wilcoxon p, the Hodges-Lehmann effect with its exact 95 % signed-rank interval
    (Bauer 1972) and coverage, and the mean difference with its circular block-bootstrap CI, labelled with its
    block length (``mean_diff_ci_block5_*`` by default). Holm runs over the ``secondary`` rows only; the
    ``descriptive`` rows (B3 vs B1+ and B1+ vs B1) have no Holm value.
    """

    rows = []
    for family, pairs in (("secondary", contrasts), ("descriptive", descriptive_contrasts)):
        for reference_model, model in pairs:
            for metric in metrics:
                days, differences = paired_day_differences(frames[reference_model], frames[model], metric)
                low, high = circular_block_bootstrap_mean_ci(
                    differences, block_length=block_length, repetitions=bootstrap_repetitions, seed=seed
                )
                hl_low, hl_high, hl_coverage = hodges_lehmann_ci(differences)
                rows.append({
                    "family": family,
                    "reference": reference_model,
                    "challenger": model,
                    "metric": metric,
                    "market_days": len(days),
                    "mean_diff": float(np.mean(differences)),
                    "median_diff": float(np.median(differences)),
                    "hl_estimate": hodges_lehmann(differences),
                    "hl_ci_95_low": hl_low,
                    "hl_ci_95_high": hl_high,
                    "hl_ci_95_exact_coverage": hl_coverage,
                    f"mean_diff_ci_block{block_length}_low": low,
                    f"mean_diff_ci_block{block_length}_high": high,
                    "test": WILCOXON_TEST,
                    "p_value": pratt_wilcoxon_pvalue(differences),
                    "holm_p": np.nan,
                })
    table = pd.DataFrame(rows)
    is_secondary = table["family"] == "secondary"
    table.loc[is_secondary, "holm_p"] = holm_adjust(table.loc[is_secondary, "p_value"].to_numpy())
    return table
