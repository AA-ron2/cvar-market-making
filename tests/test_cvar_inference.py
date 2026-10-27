"""K6, D-14, D-14a, D-16, D-18, M3, M4: CVaR of daily MaxDD, the inference families and rows, B3 feasibility."""

from __future__ import annotations

import importlib
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvar_mm.gym.cvar_lagrangian import empirical_ru_cvar, training_feasibility
from cvar_mm.gym.experiment_config import ReplayExperimentConfig
from cvar_mm.gym.statistical_inference import (
    DAY_SWAP_TEST,
    INTERVAL_ONLY,
    PATH_SWAP_TEST,
    SECONDARY_METRICS,
    circular_block_bootstrap_mean_ci,
    hodges_lehmann,
    hodges_lehmann_ci,
    holm_adjust,
    paired_block_bootstrap_cvar_difference,
    paired_inference_table,
    paired_randomization_test_cvar,
    primary_cvar_inference_table,
    ru_cvar,
    ru_cvar_of_size,
    signed_rank_critical_value,
)
from scripts.pipeline import common as snellius_common

# The inference tables' columns (D-14a, M3, M4), in order.
PRIMARY_COLUMNS = [
    "family", "reference", "challenger", "metric", "market_days", "rows_per_day", "estimate", "test", "draws",
    "p_value", "holm_p", "ci_unadjusted_95_low", "ci_unadjusted_95_high", "ci_block_length_days", "ci_repetitions",
]
SECONDARY_COLUMNS = [
    "family", "reference", "challenger", "metric", "market_days", "mean_diff", "median_diff", "hl_estimate",
    "hl_ci_95_low", "hl_ci_95_high", "hl_ci_95_exact_coverage", "mean_diff_ci_block5_low", "mean_diff_ci_block5_high",
    "test", "p_value", "holm_p",
]


def tail_mean(values: np.ndarray, rho: float = 0.2) -> float:
    """Upper-tail mean with a fractional weight on the boundary value."""

    ordered = np.sort(np.asarray(values, dtype=float))[::-1]
    mass = rho * len(ordered)
    whole = int(np.floor(mass))
    return float((ordered[:whole].sum() + (mass - whole) * ordered[whole]) / mass)


def test_day_level_cvar_has_the_known_value_and_matches_the_tail_mean():
    assert empirical_ru_cvar(np.arange(1.0, 24.0)).value == pytest.approx(21.173913043478262, rel=1e-12)
    rng = np.random.default_rng(3)
    for size in (23, 46, 920):
        values = rng.gamma(2.0, 1e-3, size)
        assert empirical_ru_cvar(values).value == pytest.approx(tail_mean(values), rel=1e-12)


@pytest.mark.parametrize("block_length", [1, 2, 3, 4])
def test_cvar_bootstrap_known_answers_for_shifted_and_identical_models(block_length):
    reference = np.random.default_rng(1).gamma(2.0, 1e-2, size=(23, 1))
    shifted = paired_block_bootstrap_cvar_difference(
        reference, reference + 0.013, block_length=block_length, repetitions=999, seed=5
    )
    assert shifted["estimate"] == pytest.approx(0.013, rel=1e-12)
    assert (shifted["ci_low"], shifted["ci_high"]) == pytest.approx((0.013, 0.013), rel=1e-12)
    assert shifted["p_value"] == 1 / 1000
    identical = paired_block_bootstrap_cvar_difference(
        reference, reference.copy(), block_length=block_length, repetitions=999, seed=5
    )
    assert identical["estimate"] == 0.0 and identical["p_value"] == 1.0


def test_cvar_bootstrap_interval_inverts_the_centred_test():
    rng = np.random.default_rng(9)
    for _ in range(25):
        reference = rng.lognormal(-4.0, 0.6, size=(23, 1))
        challenger = reference * rng.lognormal(rng.normal(0.0, 0.3), 0.3, size=(23, 1))
        result = paired_block_bootstrap_cvar_difference(
            reference, challenger, block_length=3, repetitions=2000, seed=int(rng.integers(1_000))
        )
        excludes_zero = result["ci_low"] > 0 or result["ci_high"] < 0
        assert excludes_zero == (result["p_value"] < 0.05)


def test_hodges_lehmann_is_the_median_of_walsh_averages():
    assert hodges_lehmann(np.array([-1.0, 0.0, 5.0])) == 1.0
    assert hodges_lehmann(np.array([1.0, 2.0, 3.0])) == 2.0
    assert hodges_lehmann(np.full(5, 0.3)) == pytest.approx(0.3)


def test_mean_bootstrap_keeps_its_random_stream_after_the_refactor():
    values = np.sin(np.arange(23) * 1.7) + 0.1 * np.arange(23)
    pinned = (0.6904656063171695, 1.4842104907612528)  # a48c9af implementation
    assert circular_block_bootstrap_mean_ci(values, block_length=3, repetitions=500, seed=11) == pytest.approx(
        pinned, rel=1e-12
    )


def test_the_per_draw_cvar_reproduces_empirical_ru_cvar_bit_for_bit():
    """D-14a: the tests recompute the CVaR per draw with ru_cvar_of_size, empirical_ru_cvar's own nu and value
    function without np.quantile's per-call cost; it must equal empirical_ru_cvar exactly, ties included."""

    rng = np.random.default_rng(23)
    for size in [*range(1, 61), 184, 460, 920]:
        cvar = ru_cvar_of_size(size)
        for trial in range(6):
            values = rng.lognormal(-4.0, 1.0, size)
            if trial % 2:
                values = np.round(values, 3)  # ties, as with equal daily MaxDDs
            shuffled = rng.permutation(values)
            assert cvar(shuffled) == ru_cvar(shuffled) == empirical_ru_cvar(np.sort(values)).value
    with pytest.raises(ValueError, match="expected 23 values"):
        ru_cvar_of_size(23)(np.ones(22))


def test_identical_models_have_randomization_p_1():
    """D-14a: with x == y every swap reproduces the observed difference 0, so p = 1, for days, paths and the
    exact path."""

    rng = np.random.default_rng(2)
    days = rng.gamma(2.0, 1e-2, size=23)
    result = paired_randomization_test_cvar(days, days.copy(), draws=999, seed=5)
    assert result == {"estimate": 0.0, "p_value": 1.0, "draws": 999, "exact": False}
    paths = rng.gamma(2.0, 1e-2, size=(23, 4))
    assert paired_randomization_test_cvar(paths, paths.copy(), draws=999, seed=5)["p_value"] == 1.0
    assert paired_randomization_test_cvar(days[:8], days[:8].copy(), exact=True)["p_value"] == 1.0


def test_exact_and_monte_carlo_randomization_p_agree_for_8_days():
    """D-14a: for n = 8 the 2^8 = 256 swaps give the exact p; 200,000 Monte Carlo draws land within 0.005."""

    rng = np.random.default_rng(7)
    reference = rng.lognormal(-4.0, 0.5, size=8)
    challenger = reference * rng.lognormal(0.2, 0.4, size=8)
    exact = paired_randomization_test_cvar(challenger, reference, exact=True)
    assert (exact["draws"], exact["exact"]) == (256, True)
    assert 0.05 < exact["p_value"] < 0.95  # a p in the middle, where the Monte Carlo error is largest
    monte_carlo = paired_randomization_test_cvar(challenger, reference, draws=200_000, seed=42)
    assert monte_carlo["estimate"] == exact["estimate"]
    assert abs(monte_carlo["p_value"] - exact["p_value"]) < 0.005


def test_the_exact_path_level_test_swaps_whole_days_of_paths():
    """D-14a: the path-level statistic swaps a day's paths together; checked against a brute-force enumeration
    with empirical_ru_cvar over the pooled paths."""

    rng = np.random.default_rng(17)
    reference = rng.gamma(2.0, 1.0, size=(5, 3))
    challenger = reference + rng.normal(0.5, 1.0, size=(5, 3))

    def cvar(rows) -> float:
        return empirical_ru_cvar(np.concatenate(rows)).value

    observed = cvar(challenger) - cvar(reference)
    extreme = 0
    for swapped in itertools.product((False, True), repeat=5):
        ch = [reference[day] if swap else challenger[day] for day, swap in enumerate(swapped)]
        ref = [challenger[day] if swap else reference[day] for day, swap in enumerate(swapped)]
        extreme += abs(cvar(ch) - cvar(ref)) >= abs(observed) - 1e-12
    result = paired_randomization_test_cvar(challenger, reference, exact=True)
    assert result["estimate"] == pytest.approx(observed, rel=1e-12)
    assert (result["p_value"], result["draws"]) == (extreme / 32, 32)
    assert 2 / 32 < result["p_value"] < 1  # more than the identity and the full swap, fewer than every swap


def test_randomization_p_is_bit_identical_for_one_seed():
    """D-14a: 10,000 draws with seed 42 give the same p, bit for bit, on two calls; p = (1 + count)/10,001."""

    rng = np.random.default_rng(17)
    reference = rng.lognormal(-4.0, 0.6, size=23)
    challenger = reference * rng.lognormal(0.1, 0.3, size=23)
    first = paired_randomization_test_cvar(challenger, reference, draws=10_000, seed=42)
    second = paired_randomization_test_cvar(challenger, reference, draws=10_000, seed=42)
    assert first == second and first["draws"] == 10_000
    count = first["p_value"] * 10_001 - 1
    assert count == pytest.approx(round(count), abs=1e-6) and 0 <= round(count) <= 10_000


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"challenger": np.ones(17), "reference": np.zeros(17), "exact": True}, "n = 17 days > 16"),
        ({"challenger": np.ones(5), "reference": np.zeros(5), "draws": 0}, "draws must be a positive integer"),
        ({"challenger": np.ones(5), "reference": np.zeros(6)}, "of one shape"),
        ({"challenger": np.ones(1), "reference": np.zeros(1)}, ">= 2 days"),
        ({"challenger": np.array([1.0, np.nan]), "reference": np.zeros(2)}, "finite"),
    ],
)
def test_the_randomization_test_refuses_what_it_cannot_test(kwargs, match):
    with pytest.raises(ValueError, match=match):
        paired_randomization_test_cvar(**kwargs)


def test_the_signed_rank_critical_value_at_23_days_is_73():
    """M4 acceptance: at n = 23 the 95 % HL interval is [W(74), W(203)] of the 276 Walsh averages (c = 73),
    with exact coverage 0.9516."""

    critical, coverage = signed_rank_critical_value(23)
    assert critical == 73 and round(coverage, 4) == 0.9516
    # the published two-sided 5 % critical values for small n; n = 5 is too few for any 95 % interval
    assert [signed_rank_critical_value(n)[0] for n in (6, 8, 10)] == [0, 3, 8]
    with pytest.raises(ValueError, match="too few"):
        signed_rank_critical_value(5)


@pytest.mark.parametrize("n", [6, 8, 10, 23])
def test_the_hodges_lehmann_interval_matches_brute_force_walsh_averages(n):
    """M4: [W(c+1), W(M-c)] of the sorted Walsh averages, with c from a brute-force enumeration of the
    signed-rank null (n <= 10) or the pinned c = 73 (n = 23); the interval contains the HL estimate."""

    differences = np.random.default_rng(n).normal(0.1, 1.0, size=n)
    walsh = sorted((differences[i] + differences[j]) / 2.0 for i in range(n) for j in range(i, n))
    if n <= 10:
        sums = np.array([
            sum(rank for rank, positive in zip(range(1, n + 1), signs) if positive)
            for signs in itertools.product((0, 1), repeat=n)
        ])
        critical = max(c for c in range(len(walsh) + 1) if np.mean(sums <= c) <= 0.025)
        coverage = 1.0 - 2.0 * np.mean(sums <= critical)
    else:
        critical, coverage = 73, 0.9515590667724609
    low, high, exact_coverage = hodges_lehmann_ci(differences)
    assert (low, high) == (walsh[critical], walsh[len(walsh) - 1 - critical])
    assert exact_coverage == pytest.approx(coverage, abs=1e-15)
    assert low <= hodges_lehmann(differences) <= high


def test_the_primary_difference_matches_a_hand_computed_frame():
    """D-14: Delta = CVaR_0.2(challenger) - CVaR_0.2(reference) over 23 seed-averaged days. With 23 values the RU
    estimator puts weight 0.6 on the 19th smallest: CVaR_0.2(1..23) = 19 + (1 + 2 + 3 + 4)/4.6; B3 caps B1's top
    four days at 19, so its CVaR is 19. Rows pair by Day, whatever their order."""

    dates = [f"2025-01-{day:02d}" for day in range(7, 30)]
    b1 = np.arange(1.0, 24.0)
    values = {"B0": b1 + 1.0, "B1": b1, "B2": 2.0 * b1, "B3": np.minimum(b1, 19.0), "B1PLUS": b1 - 1.0}
    day_frames = {name: pd.DataFrame({"Day": dates, "Max DD": series}).iloc[::-1] for name, series in values.items()}
    episode_frames = {  # two equal paths per day: the path-level CVaR equals the day-level one here
        name: pd.DataFrame({"Day": np.repeat(dates, 2), "Max DD": np.repeat(values[name], 2)})
        for name in ("B1", "B2", "B3")
    }
    table = primary_cvar_inference_table(day_frames, episode_frames, repetitions=99, draws=99, seed=1)
    tested = table[table["test"] != INTERVAL_ONLY]  # one row per family, contrast and statistic
    estimate = tested.set_index(["family", "reference", "challenger", "metric"])["estimate"].to_dict()
    assert len(estimate) == len(tested) == 7
    days, paths = "Max DD CVaR0.2 (days)", "Max DD CVaR0.2 (episodes)"
    cvar_b1 = 19.0 + 10.0 / 4.6
    assert cvar_b1 == pytest.approx(21.173913043478262, rel=1e-15)
    expected = {
        ("primary", "B1", "B3", days): 19.0 - cvar_b1,
        ("primary", "B2", "B3", days): 19.0 - 2.0 * cvar_b1,
        ("descriptive", "B0", "B3", days): 19.0 - (cvar_b1 + 1.0),
        ("descriptive", "B1PLUS", "B3", days): 19.0 - (cvar_b1 - 1.0),
        ("descriptive", "B1", "B1PLUS", days): -1.0,
        ("sensitivity", "B1", "B3", paths): 19.0 - cvar_b1,
        ("sensitivity", "B2", "B3", paths): 19.0 - 2.0 * cvar_b1,
    }
    for key, value in expected.items():
        assert estimate[key] == pytest.approx(value, rel=1e-12), key
    # the primary row and its three block-length rows report one estimate, bit for bit
    for reference in ("B1", "B2"):
        rows = table[(table["reference"] == reference) & (table["challenger"] == "B3") & (table["metric"] == days)]
        assert len(rows) == 4 and rows["estimate"].nunique() == 1


def synthetic_frames(days: int = 10, paths: int = 4):
    rng = np.random.default_rng(4)
    dates = [f"2025-01-{day:02d}" for day in range(7, 7 + days)]
    day_frames, episode_frames = {}, {}
    for offset, model in enumerate(("B0", "B1", "B2", "B3", "B1PLUS")):  # B1+ last: B0-B3 keep their draws
        maxdd = np.arange(1.0, days + 1.0) * (1.0 - 0.1 * offset)
        frame = pd.DataFrame({"Day": dates, "Max DD": maxdd})
        for metric in SECONDARY_METRICS:
            if metric != "Max DD":
                frame[metric] = rng.normal(size=days)
        day_frames[model] = frame
        episode_frames[model] = pd.DataFrame({
            "Day": np.repeat(dates, paths),
            "Fill Seed": np.tile(np.repeat([42, 43], paths // 2), days),
            "Rollout": np.tile(np.arange(paths), days),
            "Max DD": np.repeat(maxdd, paths) + np.tile(np.linspace(-0.3, 0.3, paths), days),
        })
    return dates, day_frames, episode_frames


def contrasts(table: pd.DataFrame, family: str) -> list[tuple[str, str]]:
    rows = table[table["family"] == family]
    return list(dict.fromkeys(zip(rows["reference"], rows["challenger"])))


def test_primary_sensitivity_and_descriptive_rows_and_their_holm():
    """D-14a: the randomization p with Holm over the 2 primary rows only; the bootstrap CI as an unadjusted
    interval (M3); sensitivity rows (blocks 1, 2, 4 with no test; the path level with whole-day swaps) and
    descriptive rows (B3 vs B0, D-16; B3 vs B1+ and B1+ vs B1, D-18) without Holm. B1+ is in descriptive rows only."""

    _, day_frames, episode_frames = synthetic_frames()
    primary = primary_cvar_inference_table(day_frames, episode_frames, repetitions=199, draws=199, seed=3)
    assert primary.columns.tolist() == PRIMARY_COLUMNS
    assert primary["family"].tolist() == ["primary"] * 2 + ["sensitivity"] * 8 + ["descriptive"] * 3
    rows = primary[primary["family"] == "primary"]
    assert contrasts(primary, "primary") == [("B1", "B3"), ("B2", "B3")]
    assert (rows["metric"] == "Max DD CVaR0.2 (days)").all() and (rows["ci_block_length_days"] == 3).all()
    assert (rows["test"] == DAY_SWAP_TEST).all() and (rows["draws"] == 199).all() and (rows["rows_per_day"] == 1).all()
    for _, row in rows.iterrows():  # p is the randomization test's, with the same draws and seed
        ref, ch = (day_frames[name].set_index("Day")["Max DD"].to_numpy() for name in (row["reference"], row["challenger"]))
        assert row["p_value"] == paired_randomization_test_cvar(ch, ref, draws=199, seed=3)["p_value"]
    # Holm over exactly the two primary p-values; no other row has a Holm value
    np.testing.assert_allclose(rows["holm_p"], holm_adjust(rows["p_value"].to_numpy()))
    assert primary.loc[primary["family"] != "primary", "holm_p"].isna().all()
    sensitivity = primary[primary["family"] == "sensitivity"]
    interval_only = sensitivity[sensitivity["test"] == INTERVAL_ONLY]
    assert sorted(interval_only["ci_block_length_days"]) == [1, 1, 2, 2, 4, 4]
    assert interval_only["p_value"].isna().all() and interval_only["draws"].isna().all()
    path_level = sensitivity[sensitivity["metric"] == "Max DD CVaR0.2 (episodes)"]
    assert contrasts(primary.loc[path_level.index], "sensitivity") == [("B1", "B3"), ("B2", "B3")]
    assert (path_level["test"] == PATH_SWAP_TEST).all() and (path_level["rows_per_day"] == 4).all()
    assert path_level["p_value"].notna().all() and (path_level["ci_block_length_days"] == 3).all()
    descriptive = primary[primary["family"] == "descriptive"]
    assert contrasts(primary, "descriptive") == [("B0", "B3"), ("B1PLUS", "B3"), ("B1", "B1PLUS")]
    assert (descriptive["test"] == DAY_SWAP_TEST).all() and descriptive["p_value"].notna().all()
    assert (primary["ci_unadjusted_95_low"] <= primary["estimate"]).all()
    assert (primary["estimate"] <= primary["ci_unadjusted_95_high"]).all()
    b1plus = (primary["reference"] == "B1PLUS") | (primary["challenger"] == "B1PLUS")
    assert set(primary.loc[b1plus, "family"]) == {"descriptive"}

    secondary = paired_inference_table(day_frames, bootstrap_repetitions=199, seed=3)
    assert secondary.columns.tolist() == SECONDARY_COLUMNS
    assert secondary["family"].tolist() == ["secondary"] * 28 + ["descriptive"] * 14
    assert contrasts(secondary, "secondary") == [("B1", "B0"), ("B1", "B2"), ("B1", "B3"), ("B2", "B3")]
    assert contrasts(secondary, "descriptive") == [("B1PLUS", "B3"), ("B1", "B1PLUS")]
    assert (secondary["test"] == "Pratt-Wilcoxon signed-rank").all()
    # Holm over exactly the 28 secondary rows, not pooled with the primary family and not omitted
    rows = secondary[secondary["family"] == "secondary"]
    raw_p = rows["p_value"].to_numpy()
    np.testing.assert_allclose(rows["holm_p"], holm_adjust(raw_p))
    assert rows["holm_p"].min() == pytest.approx(min(1.0, 28 * raw_p.min()))
    assert 28 * raw_p.min() < 1.0  # so the family size is visible in the smallest adjusted p
    assert secondary.loc[secondary["family"] == "descriptive", "holm_p"].isna().all()
    assert secondary["p_value"].notna().all()
    # M4: the HL interval around the HL effect, at the exact coverage for 10 days; the block-5 mean CI
    assert (secondary["hl_ci_95_low"] <= secondary["hl_estimate"]).all()
    assert (secondary["hl_estimate"] <= secondary["hl_ci_95_high"]).all()
    assert (secondary["hl_ci_95_exact_coverage"] == signed_rank_critical_value(10)[1]).all()
    first = secondary.iloc[0]
    differences = day_frames["B0"]["Sharpe"].to_numpy() - day_frames["B1"]["Sharpe"].to_numpy()
    assert (first["metric"], first["hl_ci_95_low"], first["hl_ci_95_high"]) == ("Sharpe", *hodges_lehmann_ci(differences)[:2])
    assert (first["mean_diff_ci_block5_low"], first["mean_diff_ci_block5_high"]) == circular_block_bootstrap_mean_ci(
        differences, block_length=5, repetitions=199, seed=3
    )


def test_b3_training_feasibility_pools_the_last_rollouts_after_the_warm_start():
    costs = [np.arange(10.0) + update for update in range(5)]
    feasibility = training_feasibility(costs, threshold=9.0)
    assert feasibility["rollouts"] == [2, 3, 4, 5]
    assert feasibility["paths_pooled"] == 40
    assert feasibility["pooled_cvar"] == pytest.approx(tail_mean(np.concatenate(costs[1:])), rel=1e-12)
    assert feasibility["mean_batch_cvar"] == pytest.approx(11.0)
    assert feasibility["feasible"] is False
    # D-17: d and the training rollouts use the same (stochastic) sampling.
    assert feasibility["scope"] == "training diagnostic: stochastic training rollouts; d from stochastic B1 rollouts (D-17)"
    assert training_feasibility(costs[:2], threshold=100.0)["rollouts"] == [2]
    with pytest.raises(ValueError, match="at least 2 updates"):
        training_feasibility(costs[:1], threshold=9.0)


def test_evaluation_plumbing_yields_episode_files_cvar_columns_and_both_families(tmp_path, monkeypatch):
    dates, day_frames, episode_frames = synthetic_frames()
    validation_dates = ["2025-01-05", "2025-01-06"]
    cfg = ReplayExperimentConfig(repo_root=tmp_path, datasets_subdir=tmp_path)
    cfg.ensure_artifact_dirs()
    manifest = pd.DataFrame({
        "date": validation_dates + dates,
        "split": ["train"] * 2 + ["test"] * len(dates),
        "usage": ["validation"] * 2 + ["retrospective_holdout"] * len(dates),
        "split_index": list(range(2)) + list(range(len(dates))),
        "input_path": [f"synthetic_{date}.csv" for date in validation_dates + dates],
    })
    ctx = snellius_common.FormulaPipelineContext(
        project_dir=tmp_path, data_dir=tmp_path, cfg=cfg, manifest=manifest, train_params={}, formula_kwargs={}
    )
    for model in ("b1", "b2", "b3"):
        for date in dates:
            day_part = snellius_common.eval_part_path(cfg, model=model, date=date)
            day_part.parent.mkdir(parents=True, exist_ok=True)
            day_frames[model.upper()].query("Day == @date").to_csv(day_part, index=False)
            episodes = episode_frames[model.upper()]
            episodes[episodes["Day"] == date].to_csv(
                snellius_common.eval_episode_path(cfg, model=model, date=date), index=False
            )
    for date in validation_dates:  # B2 candidates: day parts only, no episodes needed
        part = snellius_common.eval_part_path(cfg, model="b2", alpha=0.1, date=date, split="validation")
        part.parent.mkdir(parents=True, exist_ok=True)
        day_frames["B2"].head(1).assign(Day=date).to_csv(part, index=False)
    day_frames["B0"].to_csv(cfg.result_path("b0_test_results.csv"), index=False)
    episode_frames["B0"].to_csv(cfg.result_path("b0_test_episodes.csv"), index=False)
    # B1+ (D-18), better than every model on every metric: it would win every day if it were counted.
    day_frames["B1PLUS"] = day_frames["B1"].assign(**{
        "Max DD": day_frames["B1"]["Max DD"] * 0.01,
        **{metric: day_frames["B1"][metric] + 100.0 for metric in ("Sharpe", "Sortino", "P&L-to-MAP")},
    })
    episode_frames["B1PLUS"] = episode_frames["B1"].assign(**{"Max DD": episode_frames["B1"]["Max DD"] * 0.01})
    for date in dates:
        day_part = snellius_common.eval_part_path(cfg, model="b1plus", date=date)
        day_part.parent.mkdir(parents=True, exist_ok=True)
        day_frames["B1PLUS"].query("Day == @date").to_csv(day_part, index=False)
        episodes = episode_frames["B1PLUS"]
        episodes[episodes["Day"] == date].to_csv(
            snellius_common.eval_episode_path(cfg, model="b1plus", date=date), index=False
        )

    aggregate = importlib.import_module("scripts.pipeline.aggregate_final_results")
    inference = importlib.import_module("scripts.pipeline.run_paired_inference")
    metadata: dict[str, dict] = {}
    for module in (aggregate, inference):
        monkeypatch.setattr(module, "load_context", lambda **_: ctx)
        monkeypatch.setattr(module, "write_metadata", lambda job, payload: metadata.update({job: payload}))
    for model in ("b1", "b2", "b3"):
        aggregate.aggregate_model_parts(model=model)
    # B1+ through the CLI that its b1plus_agg job runs (D-18)
    monkeypatch.setattr(sys, "argv", ["aggregate_final_results.py", "--model", "b1plus"])
    aggregate.main()
    aggregate.aggregate_model_parts(model="b2", alpha=0.1, split="validation")
    assert len(pd.read_csv(cfg.result_path("b1_test_episodes.csv"))) == len(dates) * 4
    assert len(pd.read_csv(cfg.result_path("b1plus_test_results.csv"))) == len(dates)
    assert len(pd.read_csv(cfg.result_path("b1plus_test_episodes.csv"))) == len(dates) * 4
    assert not cfg.result_path("b2_alpha0.1_validation_episodes.csv").exists()
    with pytest.raises(ValueError, match="D-18"):  # B1+ trained on Jan 1-6
        aggregate.aggregate_model_parts(model="b1plus", split="validation")

    monkeypatch.setattr(sys, "argv", ["aggregate_final_results.py", "--comparison"])  # the final_compare job
    aggregate.main()
    # The comparison's metadata names (so write_metadata hashes) the day-level files it read, B1+'s included.
    assert metadata["final_comparison"]["results_paths"]["B1PLUS"] == cfg.result_path("b1plus_test_results.csv")
    summary = pd.read_csv(cfg.result_path("overall_summary_table.csv")).set_index("Model")
    assert summary.index.tolist() == ["B0", "B1", "B2", "B3", "B1PLUS"]
    for model in ("B0", "B1", "B2", "B3", "B1PLUS"):
        assert summary.loc[model, "Max DD CVaR0.2 (days)"] == pytest.approx(
            tail_mean(day_frames[model]["Max DD"]), rel=1e-12
        )
        assert summary.loc[model, "Max DD CVaR0.2 (episodes)"] == pytest.approx(
            tail_mean(episode_frames[model]["Max DD"]), rel=1e-12
        )
    # The days-best table counts B0-B3 only (B1+ is a summary row; the author's choice of 2026-10-07).
    table = pd.read_csv(cfg.result_path("comparison_table.csv")).set_index("Metric")
    assert table.columns.tolist() == ["B0", "B1", "B2", "B3"]
    for metric, higher_better in aggregate.DAYS_BEST_METRICS.items():
        values = pd.DataFrame({name: day_frames[name][metric].to_numpy() for name in ("B0", "B1", "B2", "B3")})
        winners = values.idxmax(axis=1) if higher_better else values.idxmin(axis=1)
        expected = winners.value_counts().reindex(["B0", "B1", "B2", "B3"], fill_value=0)
        assert table.loc[metric].tolist() == expected.tolist(), metric

    # The paired_infer job (D-14a, D-16, D-18): the primary, sensitivity and descriptive families with their
    # columns; B1+ in descriptive rows only; its inputs, B1+'s included, named (so hashed) in the metadata.
    monkeypatch.setattr(
        sys, "argv", ["run_paired_inference.py", "--bootstrap-repetitions", "199", "--randomization-draws", "999"]
    )
    inference.main()
    primary = pd.read_csv(cfg.result_path("paired_day_inference_primary.csv"))
    secondary = pd.read_csv(cfg.result_path("paired_day_inference.csv"))
    assert primary.columns.tolist() == PRIMARY_COLUMNS and secondary.columns.tolist() == SECONDARY_COLUMNS
    assert primary["family"].value_counts().to_dict() == {"primary": 2, "sensitivity": 8, "descriptive": 3}
    assert secondary["family"].value_counts().to_dict() == {"secondary": 28, "descriptive": 14}
    for table in (primary, secondary):
        b1plus = (table["reference"] == "B1PLUS") | (table["challenger"] == "B1PLUS")
        assert b1plus.any() and set(table.loc[b1plus, "family"]) == {"descriptive"}
        assert table.loc[table["family"].isin(["sensitivity", "descriptive"]), "holm_p"].isna().all()
    assert primary.loc[primary["family"] == "primary", "holm_p"].notna().all()
    assert (primary.loc[primary["family"] != "sensitivity", "draws"] == 999).all()
    assert contrasts(primary, "descriptive") == [("B0", "B3"), ("B1PLUS", "B3"), ("B1", "B1PLUS")]
    assert metadata["paired_day_inference"]["input_paths"]["B1PLUS"] == cfg.result_path("b1plus_test_results.csv")
    assert metadata["paired_day_inference"]["primary_family"]["randomization_draws"] == 999
    # B1+'s rows read its aggregated days (Max DD = B1's x 0.01)
    row = primary[(primary["reference"] == "B1") & (primary["challenger"] == "B1PLUS")].iloc[0]
    assert row["estimate"] == pytest.approx(
        tail_mean(day_frames["B1PLUS"]["Max DD"]) - tail_mean(day_frames["B1"]["Max DD"]), rel=1e-9
    )

    uneven = snellius_common.eval_episode_path(cfg, model="b3", date=dates[0])
    pd.read_csv(uneven).head(3).to_csv(uneven, index=False)
    with pytest.raises(ValueError, match="unequal numbers of paths per day"):
        aggregate.aggregate_model_parts(model="b3")

    cfg.result_path("b1plus_test_episodes.csv").unlink()  # final_compare waits for B1+ (D-18)
    with pytest.raises(FileNotFoundError, match="B1PLUS"):
        aggregate.write_comparison()
