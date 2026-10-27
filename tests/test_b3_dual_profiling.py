"""D-12a: the training-only rule that chooses the B3 dual step size eta_lambda."""

from __future__ import annotations

import hashlib
import io
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("stable_baselines3")

from cvar_mm.gym.cvar_lagrangian import B3_LAMBDA_MAX, B3_TAIL_MASS, B3_THRESHOLD_ESTIMATOR, B3_THRESHOLD_MULTIPLIER
from cvar_mm.gym.drawdown_objectives import UPDATE_DIAGNOSTIC_COLUMNS
from cvar_mm.gym.experiment_config import ReplayExperimentConfig
from scripts.pipeline import common, profile_b3_dual
from scripts.pipeline.profile_b3_dual import d12a_choice, dual_profile_summary, select_lambda_lr

D = 0.5  # binary-exact, so the hand-computed criteria |H - d|/d below are exact
D_AWKWARD = 0.1 + 0.2  # 0.30000000000000004: pandas' default CSV parser reads it back as 0.3


def history(violations, *, eta: float, d: float = D, lambdas=None, fill_paths: int = 16) -> pd.DataFrame:
    """A 10-update profiling history laid out as profile_eta writes it.

    lambda follows B3's dual step from lambda_0 = 0 unless given; lambda_used is lambda one update earlier.
    """

    violations = np.asarray(violations, dtype=float)
    if lambdas is None:
        lambdas, lam = [], 0.0
        for violation in violations:
            lam = float(np.clip(lam + eta * violation / d, 0.0, B3_LAMBDA_MAX))
            lambdas.append(lam)
    lambdas = np.asarray(lambdas, dtype=float)
    updates = len(violations)
    diagnostics = {
        "reward_scale_s": 1.0, "advantage_mean": 0.0, "advantage_std": 1.0, "reward_clip_fraction": 0.0,
        "adam_steps": 320, "minibatches_per_epoch": 32, "epochs_completed": 10, "kl_stopped": False,
        "approx_kl": 0.01, "clip_fraction": 0.1, "value_loss": 0.5, "explained_variance": 0.25,  # the last two: D-23
        "mean_log_std": 0.0, "adam_damping_ratio": 0.8,
        "action_shift_vs_b1_gamma": 0.0125, "action_shift_vs_b1_skew": 0.025,  # D-21, one column per action
        "elapsed_seconds": 60.0 * np.arange(1, updates + 1),
    }
    assert set(diagnostics) == set(UPDATE_DIAGNOSTIC_COLUMNS)
    return pd.DataFrame({
        "update": range(1, updates + 1),
        "eta": eta,
        "lambda_used": np.r_[0.0, lambdas[:-1]],
        "lambda": lambdas,
        "nu": d,
        "ru_cvar": d + violations,
        "violation": violations,
        "mean_full_day_maxdd": d,
        **diagnostics,
        "threshold_d": d,
        "training_fill_paths": fill_paths,
        "peak_rss_so_far_gib": None,  # as written on Windows
        "peak_rss_source": "unavailable on this platform",
    })


def grid(v05, v2, v8, **kwargs) -> dict[float, pd.DataFrame]:
    return {0.5: history(v05, eta=0.5, **kwargs), 2.0: history(v2, eta=2.0, **kwargs), 8.0: history(v8, eta=8.0, **kwargs)}


def summaries_of(histories: dict[float, pd.DataFrame]) -> dict[float, dict]:
    return {eta: dual_profile_summary(frame) for eta, frame in histories.items()}


def test_three_sign_changes_exclude_an_eta_and_exactly_two_do_not():
    histories = grid(
        [0.25, -0.25, 0.25] + [-0.0625] * 7,  # 3 sign changes, though its criterion 0.125 is the smallest
        [0.25, 0.25, -0.25, -0.25] + [0.125] * 6,  # exactly 2 sign changes: kept, criterion 0.25
        [0.25] * 10,  # criterion 0.5
    )
    summaries = summaries_of(histories)
    assert summaries[0.5] == {
        "max_lambda": 0.25,
        "violation_sign_changes": 3,
        "criterion": 0.125,
        "kept": False,
        "exclusion_reason": "H - d changed sign 3 times (more than 2)",
    }
    assert summaries[2.0]["violation_sign_changes"] == 2
    assert summaries[2.0]["kept"] and summaries[2.0]["exclusion_reason"] is None
    assert select_lambda_lr(histories) == 2.0


def test_lambda_at_lambda_max_excludes_an_eta_even_when_reached_only_at_update_10():
    histories = {
        # the smallest criterion, but lambda hits the bound after update 10 (lambda_used stays at most 9)
        0.5: history([0.03125] * 10, eta=0.5, lambdas=[*range(1, 10), B3_LAMBDA_MAX]),
        2.0: history([0.125] * 10, eta=2.0),
        8.0: history([0.25] * 10, eta=8.0),
    }
    assert histories[0.5]["lambda_used"].max() == 9.0
    summaries = summaries_of(histories)
    assert summaries[0.5] == {
        "max_lambda": 500.0,
        "violation_sign_changes": 0,
        "criterion": 0.0625,
        "kept": False,
        "exclusion_reason": "lambda reached lambda_max = 500",
    }
    assert select_lambda_lr(histories) == 2.0


def test_the_smallest_mean_relative_violation_over_updates_8_to_10_wins():
    histories = grid(
        [-0.25] * 10,  # signed mean -0.5: the pick of a signed criterion
        [0.5] * 6 + [2.0] + [0.0625] * 3,  # far off d on updates 1-7, close on 8-10
        [0.25] * 10,  # the largest eta, also kept
    )
    summaries = summaries_of(histories)
    # Exact: a criterion without abs, without /d, or over another window gives other numbers.
    assert {eta: summary["criterion"] for eta, summary in summaries.items()} == {0.5: 0.5, 2.0: 0.125, 8.0: 0.5}
    assert all(summary["kept"] for summary in summaries.values())
    # Over updates 7-9 or 1-10 eta = 2 scores 1.4167 or 1.0375, worse than 0.5; signed, eta = 0.5 wins.
    assert select_lambda_lr(histories) == 2.0


def test_an_exact_criterion_tie_goes_to_the_smaller_eta():
    assert select_lambda_lr(grid([0.25] * 10, [0.125] * 10, [0.125] * 10)) == 2.0
    assert select_lambda_lr(grid([0.125] * 10, [0.125] * 10, [0.125] * 10)) == 0.5


def test_lambda_zero_at_every_used_update_for_every_eta_is_a_tie_by_construction():
    below = [-0.25] * 10
    nudged = [-0.25] * 7 + [math.nextafter(-0.25, 0.0)] * 3  # floating-point noise between runs
    histories = grid(below, below, nudged)
    summaries = summaries_of(histories)
    assert summaries[8.0]["criterion"] < summaries[0.5]["criterion"]  # the plain minimum would pick 8
    assert d12a_choice(histories, summaries) == (0.5, profile_b3_dual.TIE_BY_CONSTRUCTION_NOTE)
    assert select_lambda_lr(histories) == 0.5

    # lambda turns positive only after update 10, which never acts: the runs still coincide
    late = grid([-0.25] * 9 + [0.25], [-0.25] * 9 + [0.25], [-0.25] * 7 + [-0.125, -0.125, 0.125])
    late_summaries = summaries_of(late)
    assert all(frame["lambda"].iloc[-1] > 0.0 for frame in late.values())
    assert late_summaries[8.0]["criterion"] == 0.25 < late_summaries[0.5]["criterion"] == 0.5
    assert d12a_choice(late, late_summaries) == (0.5, profile_b3_dual.TIE_BY_CONSTRUCTION_NOTE)

    # lambda leaves 0 for one eta only: the ordinary rule
    mixed = grid(below, below, [0.0625] * 10)
    assert d12a_choice(mixed, summaries_of(mixed)) == (8.0, None)

    # The tie takes the smallest KEPT eta. (No real run gives this: lambda_10 = 500 does not follow
    # B3's dual step here, and in a real run a smaller eta that hits the bound means every larger one does.)
    excluded_smallest = {
        0.5: history(below, eta=0.5, lambdas=[0.0] * 9 + [B3_LAMBDA_MAX]),
        2.0: history(below, eta=2.0),
        8.0: history(below, eta=8.0),
    }
    assert d12a_choice(excluded_smallest, summaries_of(excluded_smallest)) == (
        2.0, profile_b3_dual.TIE_BY_CONSTRUCTION_NOTE
    )
    assert profile_b3_dual.TIE_BY_CONSTRUCTION_NOTE == (
        "The lambda used at every update was 0 for every eta, so the runs coincide and their criteria tie by "
        "construction: the smallest kept eta is chosen (D-12a clarification, 2026-10-07)."
    )


def test_every_eta_excluded_raises_and_names_each_reason():
    histories = {
        0.5: history([0.25, -0.25, 0.25, -0.25] + [0.0625] * 6, eta=0.5),
        2.0: history([0.125] * 10, eta=2.0, lambdas=[1.0, 2.0, 3.0, 4.0] + [B3_LAMBDA_MAX] * 6),
        8.0: history([0.25, -0.25, 0.25, -0.25] + [0.25] * 6, eta=8.0, lambdas=[4.0] * 4 + [B3_LAMBDA_MAX] * 6),
    }
    with pytest.raises(RuntimeError, match="D-12a") as excinfo:
        select_lambda_lr(histories)
    message = str(excinfo.value)
    assert "eta=0.5: H - d changed sign 4 times" in message
    assert "eta=2.0: lambda reached lambda_max = 500" in message
    assert "eta=8.0: lambda reached lambda_max = 500; H - d changed sign 4 times" in message
    assert "no fallback" in message


def test_the_largest_eta_is_not_selected_when_it_overshoots_the_budget():
    histories = grid(
        [0.125] * 10,  # still above d: criterion 0.25
        [0.125] * 3 + [-0.0625] * 3 + [0.03125] * 4,  # crosses d twice, ends near it: criterion 0.0625
        [0.125] * 2 + [-0.15625] * 8,  # overshoots to H/d = 0.6875: criterion 0.3125
    )
    summaries = summaries_of(histories)
    assert summaries[8.0]["kept"] and summaries[8.0]["violation_sign_changes"] == 1
    assert summaries[2.0]["kept"] and summaries[2.0]["violation_sign_changes"] == 2
    assert (histories[8.0]["ru_cvar"] / D).iloc[-3:].tolist() == [0.6875] * 3
    assert select_lambda_lr(histories) == 2.0  # the D-12 rule (largest kept eta) would have taken 8


def test_a_zero_violation_is_not_a_sign_of_its_own():
    summary = dual_profile_summary(history([0.25, 0.0, 0.25, 0.0, -0.25] + [-0.25] * 5, eta=2.0))
    assert summary["violation_sign_changes"] == 1  # zeros counted as signs would make it 4
    assert summary["kept"]
    mirrored = dual_profile_summary(history([-0.25, 0.0, -0.25, 0.0, -0.25] + [0.25] * 5, eta=2.0))
    assert mirrored["violation_sign_changes"] == 1  # a zero counted as positive would make it 5
    assert mirrored["kept"]


def malformed(case: str) -> pd.DataFrame:
    frame = history([0.125] * 10, eta=2.0)
    if case == "nine updates":
        return frame.iloc[:9]
    if case == "labels 0-9":
        return frame.assign(update=range(10))
    if case == "a NaN violation":
        frame.loc[4, "violation"] = np.nan
        return frame
    if case == "d not constant":
        frame.loc[4, "threshold_d"] = 2 * D
        return frame
    if case == "lambda above lambda_max":
        return history([0.125] * 10, eta=2.0, lambdas=[1.0] * 9 + [B3_LAMBDA_MAX + 1])
    if case == "lambda below 0":
        return history([0.125] * 10, eta=2.0, lambdas=[-1.0] + [1.0] * 9)
    if case == "lambda_used stuck at 0":  # starts at 0 but is not lambda one update earlier
        return frame.assign(lambda_used=0.0)
    if case == "lambda_used starts above 0":  # lambda one update earlier, but lambda_0 != 0
        return frame.assign(lambda_used=np.r_[0.25, frame["lambda"].to_numpy()[:-1]])
    if case == "d not positive":
        return frame.assign(threshold_d=0.0)
    if case == "no violation column":
        return frame.drop(columns="violation")
    raise AssertionError(case)


@pytest.mark.parametrize("case", [
    "nine updates", "labels 0-9", "a NaN violation", "d not constant", "lambda above lambda_max",
    "lambda below 0", "lambda_used stuck at 0", "lambda_used starts above 0", "d not positive",
    "no violation column",
])
def test_malformed_histories_are_refused(case):
    with pytest.raises(ValueError, match="D-12a"):
        dual_profile_summary(malformed(case))


def test_empty_diagnostics_and_an_unmeasured_memory_do_not_block_the_rule():
    frame = history([0.125] * 10, eta=2.0)
    frame["adam_damping_ratio"] = np.nan  # None when an update took no Adam step
    frame["explained_variance"] = np.nan  # SB3's value when a batch's returns have zero variance (D-23)
    frame["peak_rss_so_far_gib"] = np.nan  # how a null memory column reads back from the CSV
    assert dual_profile_summary(frame)["kept"]


def test_peak_rss_is_null_on_windows_and_read_from_ru_maxrss_in_kib_on_linux(monkeypatch):
    # WORK_ORDER item 3: Windows has no resource module, so the column is null with this exact source
    assert common.peak_rss_so_far_gib("win32") == (None, "unavailable on this platform")

    asked: list[int] = []

    def getrusage(who):
        asked.append(who)
        return SimpleNamespace(ru_maxrss=3 * 1024**2)  # KiB on Linux

    monkeypatch.setitem(sys.modules, "resource", SimpleNamespace(RUSAGE_SELF=-7, getrusage=getrusage))
    assert common.peak_rss_so_far_gib("linux") == (3.0, common.PEAK_RSS_LINUX)
    assert asked == [-7]  # RUSAGE_SELF: this process, which runs all N fill paths and PPO


def write_d17_pair(txt: Path, json_path: Path, d: float, drop: tuple[str, ...] = ()) -> None:
    """A complete D-17 d file pair, as calibrate_cvar_b3.py writes its fields (minus ``drop``)."""

    cvar = d / B3_THRESHOLD_MULTIPLIER
    txt.write_text(f"{d!r}\n{cvar!r}\n0.0\n", encoding="utf-8")
    record = {
        "estimator": B3_THRESHOLD_ESTIMATOR,
        "d": d,
        "rho": B3_TAIL_MASS,
        "threshold_multiplier": B3_THRESHOLD_MULTIPLIER,
        "cvar_stochastic": cvar,
        "nu_stochastic": 0.0,
        "cvar_deterministic": 0.875 * cvar,
        "nu_deterministic": 0.0,
        "b1_log_std": [-0.25, -0.5],
        "b1_std": [math.exp(-0.25), math.exp(-0.5)],
    }
    json_path.write_text(json.dumps({key: value for key, value in record.items() if key not in drop}), encoding="utf-8")


def write_profile(root: Path, histories: dict[float, pd.DataFrame], d: float, d17_drop: tuple[str, ...] = ()):
    """What the profiling tasks leave behind: six hashed inputs, each eta's history and hash file, and each
    eta's B3 run (D-20: stand-in bytes, one distinct content per eta and file)."""

    cfg = ReplayExperimentConfig(repo_root=root, datasets_subdir=root)
    cfg.ensure_artifact_dirs()
    ctx = SimpleNamespace(cfg=cfg)
    inputs = profile_b3_dual.profiled_inputs(ctx, 42)
    model, vecnorm, d_txt, d_json, train_params, manifest_csv = inputs
    for path in (model, vecnorm, train_params, manifest_csv):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())
    write_d17_pair(d_txt, d_json, d, drop=d17_drop)
    directory = cfg.results_dir / "profiling" / "b3_dual"
    directory.mkdir(parents=True)
    current = profile_b3_dual.sha256_lines(root, inputs)
    for eta, frame in histories.items():
        frame.to_csv(profile_b3_dual.history_path(directory, eta), index=False)
        profile_b3_dual.inputs_path(directory, eta).write_text(current, encoding="utf-8")
        for key, path in common.eta_run_paths(cfg, eta).items():
            path.write_bytes(f"eta {eta} B3 run {key}".encode())
    return ctx, directory, inputs, current


def final_b3_files(ctx) -> dict:
    return {key: path.read_bytes() for key, path in common.b3_run_paths(ctx.cfg, 42).items() if path.exists()}


def eta_run_bytes(ctx, eta: float) -> dict:
    return {key: path.read_bytes() for key, path in common.eta_run_paths(ctx.cfg, eta).items()}


def overshoot_grid(d: float) -> dict[float, pd.DataFrame]:
    return grid([0.125] * 10, [0.125] * 3 + [-0.0625] * 3 + [0.03125] * 4, [0.125] * 2 + [-0.15625] * 8, d=d)


def all_excluded_grid() -> dict[float, pd.DataFrame]:
    return {
        0.5: history([0.25, -0.25, 0.25, -0.25] + [0.0625] * 6, eta=0.5),
        2.0: history([0.125] * 10, eta=2.0, lambdas=[1.0, 2.0, 3.0, 4.0] + [B3_LAMBDA_MAX] * 6),
        8.0: history([0.25, -0.25, 0.25, -0.25] + [0.25] * 6, eta=8.0),
    }


SELECTION_KEYS = {
    "rule", "rule_text", "eta_grid", "updates_per_eta", "criterion_updates", "max_sign_changes", "tie_break",
    "lambda_max", "selected_lambda_lr", "selection_note", "per_eta", "threshold_d", "training_fill_paths",
    "training_seed", "inputs_sha256", "histories_sha256", "promoted_b3", "dates_used", "excluded", "git_commit",
    "python_version", "python_executable", "timestamp_utc",
}


def test_select_records_the_d12a_choice_from_hash_checked_histories(tmp_path):
    assert pd.read_csv(io.StringIO(f"x\n{D_AWKWARD!r}\n"))["x"][0] != D_AWKWARD  # the default parser is inexact
    histories = overshoot_grid(D_AWKWARD)
    ctx, directory, inputs, current = write_profile(tmp_path, histories, D_AWKWARD)

    assert profile_b3_dual.select(ctx, directory, inputs, seed=42) == 2.0

    selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    assert set(selection) == SELECTION_KEYS
    assert selection["rule"] == "D-12a"
    assert selection["rule_text"] == (  # the recorded statement of D-12a, pinned word for word
        "eta_lambda in {0.5, 2, 8}, 10 B3 updates each on Jan 1-6 with training seed 42. Exclude an eta if "
        "lambda after any update reaches lambda_max, or if H_u - d changes sign more than twice over the 10 "
        "updates (an exact zero is not a sign). Among the rest choose the smallest mean |H_u - d|/d over "
        "updates 8-10; ties go to the smaller eta. If the lambda used at every update is 0 for every eta, the "
        "runs coincide and their criteria tie by construction. If every eta is excluded, stop: there is no "
        "fallback (DECISIONS D-12a)."
    )
    assert (selection["eta_grid"], selection["updates_per_eta"]) == ([0.5, 2.0, 8.0], 10)
    assert (selection["criterion_updates"], selection["max_sign_changes"]) == ([8, 9, 10], 2)
    assert selection["tie_break"] == "the smaller eta, on an exact tie or a tie by construction"
    assert selection["dates_used"] == [f"2025-01-0{day}" for day in range(1, 7)]
    assert selection["excluded"] == "January 7-29 is never loaded"
    assert selection["lambda_max"] == B3_LAMBDA_MAX
    assert (selection["selected_lambda_lr"], selection["selection_note"]) == (2.0, None)
    assert selection["threshold_d"] == D_AWKWARD
    assert (selection["training_fill_paths"], selection["training_seed"]) == (16, 42)
    assert selection["inputs_sha256"] == current.splitlines()
    assert selection["histories_sha256"] == [  # for item 7: the histories this choice was made from
        f"{hashlib.sha256(profile_b3_dual.history_path(directory, eta).read_bytes()).hexdigest()}  "
        f"results/profiling/b3_dual/history_eta{eta}.csv"
        for eta in (0.5, 2.0, 8.0)
    ]
    for eta, frame in histories.items():
        assert selection["per_eta"][str(eta)] == {**dual_profile_summary(frame), "seconds_per_update": 60.0}
    assert (directory / "selected_lambda_lr.txt").read_text(encoding="utf-8") == "2.0\n"
    assert profile_b3_dual.inputs_path(directory).read_text(encoding="utf-8") == current
    page = (directory / "summary.md").read_text(encoding="utf-8")
    assert page.startswith("# B3 dual step size: D-12a summary\n\n**Selected: η_λ = 2.0.**")
    assert selection["timestamp_utc"] in page and str(selection["git_commit"]) in page  # one select, one record
    # D-20: the selected eta run is now the final B3, byte for byte, and selection.json records its SHA-256.
    assert final_b3_files(ctx) == eta_run_bytes(ctx, 2.0)
    targets, sources = common.b3_run_paths(ctx.cfg, 42), common.eta_run_paths(ctx.cfg, 2.0)
    assert selection["promoted_b3"] == {
        "eta": 2.0,
        "files": {
            key: {
                "from": sources[key].relative_to(tmp_path).as_posix(),
                "to": targets[key].relative_to(tmp_path).as_posix(),
                "sha256": hashlib.sha256(targets[key].read_bytes()).hexdigest(),
            }
            for key in common.B3_RUN_FILES
        },
    }
    assert [targets[key].relative_to(tmp_path).as_posix() for key in common.B3_RUN_FILES] == [
        "models/ppo_b3_doge.zip", "models/vecnorm_b3.pkl", "results/b3_cvar_lagrangian_history.csv",
        "results/b3_training_costs.csv", "results/b3_training_feasibility.json",
    ]


@pytest.mark.parametrize(("case", "error", "match"), [
    ("history d differs from the D-17 d", ValueError, "a d other than the current D-17 d"),
    ("eta column mislabelled", ValueError, "does not hold eta=2.0"),
    ("fill paths differ across eta", ValueError, "different fill path counts"),
    ("D-17 record lacks the deterministic CVaR", ValueError, "lacks \\['cvar_deterministic'\\]"),
])
def test_select_refuses_histories_that_do_not_match_the_profiled_inputs(tmp_path, case, error, match):
    histories = overshoot_grid(D_AWKWARD)
    drop: tuple[str, ...] = ()
    if case == "history d differs from the D-17 d":
        histories[8.0] = history([0.25] * 10, eta=8.0, d=2 * D_AWKWARD)
    elif case == "eta column mislabelled":
        histories[2.0] = histories[2.0].assign(eta=8.0)
    elif case == "fill paths differ across eta":
        histories[8.0] = history([0.25] * 10, eta=8.0, d=D_AWKWARD, fill_paths=8)
    else:
        drop = ("cvar_deterministic",)
    ctx, directory, inputs, _ = write_profile(tmp_path, histories, D_AWKWARD, d17_drop=drop)
    with pytest.raises(error, match=match):
        profile_b3_dual.select(ctx, directory, inputs, seed=42)
    for name in profile_b3_dual.SELECTION_OUTPUTS:  # not even a summary after a failed check
        assert not (directory / name).exists()
    assert final_b3_files(ctx) == {}  # D-20: a refused selection promotes nothing


def test_select_with_every_eta_excluded_writes_no_choice(tmp_path):
    ctx, directory, inputs, _ = write_profile(tmp_path, all_excluded_grid(), D)
    with pytest.raises(RuntimeError, match="D-12a"):
        profile_b3_dual.select(ctx, directory, inputs, seed=42)
    for name in ("selected_lambda_lr.txt", "selection.json", "inputs.sha256"):
        assert not (directory / name).exists()
    page = (directory / "summary.md").read_text(encoding="utf-8")  # the report D-12a and D-15b ask for
    assert "**No η selected: every η is excluded.**" in page
    for reason in ("H - d changed sign 4 times (more than 2)", "lambda reached lambda_max = 500"):
        assert reason in page
    assert final_b3_files(ctx) == {}  # D-20: no B3 without a selected eta
    assert "| γ | -0.25 | 0.7788 |\n| skew | -0.5 | 0.6065 |" in page  # B1's sigma (D-15b)
    assert "deterministic − stochastic = -0.078125 (-12.5 % of the stochastic value)." in page


def test_select_writes_the_page_first_and_the_choice_last(tmp_path, monkeypatch):
    ctx, directory, inputs, _ = write_profile(tmp_path, overshoot_grid(D), D)
    written: list[str] = []
    original = Path.write_text

    def recording(self, *args, **kwargs):
        written.append(self.name)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", recording)
    assert profile_b3_dual.select(ctx, directory, inputs, seed=42) == 2.0
    # of the eta outputs the final stage needs only inputs.sha256 and the txt (beside the D-19 profile-stage
    # record), so the txt must never exist without the rest
    assert written == ["summary.md", "inputs.sha256", "selection.json", "selected_lambda_lr.txt"]


@pytest.mark.parametrize("failing", ["summary.md", "inputs.sha256", "selection.json"])
def test_a_failed_write_leaves_no_choice(tmp_path, monkeypatch, failing):
    ctx, directory, inputs, _ = write_profile(tmp_path, overshoot_grid(D), D)
    original = Path.write_text

    def failing_write(self, *args, **kwargs):
        if self.name == failing:
            raise OSError(f"cannot write {failing}")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", failing_write)
    with pytest.raises(OSError, match=failing):
        profile_b3_dual.select(ctx, directory, inputs, seed=42)
    assert not (directory / "selected_lambda_lr.txt").exists()


def test_every_select_first_moves_the_previous_selection_aside(tmp_path):
    assert profile_b3_dual.SELECTION_OUTPUTS == ("selected_lambda_lr.txt", "selection.json", "inputs.sha256", "summary.md")
    ctx, directory, inputs, _ = write_profile(tmp_path, all_excluded_grid(), D)
    planted = {name: f"an earlier {name}\n".encode() for name in profile_b3_dual.SELECTION_OUTPUTS}
    for name, content in planted.items():
        (directory / name).write_bytes(content)
    final_b3 = common.b3_run_paths(ctx.cfg, 42)
    for path in final_b3.values():  # D-20: the B3 an earlier selection promoted
        path.write_bytes(f"an earlier final B3 {path.name}".encode())
    planted_b3 = {path.name: path.read_bytes() for path in final_b3.values()}
    profiled = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file() and path.name not in planted}
    assert len(profiled) == 15  # per eta: its history and hash file, and its run's history, costs and feasibility

    with pytest.raises(RuntimeError, match="D-12a"):
        profile_b3_dual.select(ctx, directory, inputs, seed=42)

    # D-12a has no fallback: after a refused selection no earlier choice and no earlier B3 is left at the top
    # level (D-20), and summary.md is this stop's own report
    for name in ("selected_lambda_lr.txt", "selection.json", "inputs.sha256"):
        assert not (directory / name).exists()
    assert final_b3_files(ctx) == {}
    assert "**No η selected" in (directory / "summary.md").read_text(encoding="utf-8")
    (archive,) = (directory / "superseded").iterdir()
    assert {path.name: path.read_bytes() for path in archive.iterdir()} == {**planted, **planted_b3}
    assert {name: (directory / name).read_bytes() for name in profiled} == profiled  # histories and runs never move

    for eta, frame in overshoot_grid(D).items():  # re-profiled on the same inputs
        frame.to_csv(profile_b3_dual.history_path(directory, eta), index=False)
    assert profile_b3_dual.select(ctx, directory, inputs, seed=42) == 2.0
    first, second = sorted((directory / "superseded").iterdir())  # UTC stamps sort in time order
    assert first == archive
    assert [path.name for path in second.iterdir()] == ["summary.md"]  # the stop's report, moved aside
    chosen = {name: (directory / name).read_bytes() for name in planted}
    promoted = {path.name: path.read_bytes() for path in final_b3.values()}
    assert final_b3_files(ctx) == eta_run_bytes(ctx, 2.0)
    assert profile_b3_dual.select(ctx, directory, inputs, seed=42) == 2.0
    *_, latest = sorted((directory / "superseded").iterdir())
    assert {path.name: path.read_bytes() for path in latest.iterdir()} == {**chosen, **promoted}


def test_a_selection_whose_eta_run_is_incomplete_promotes_nothing(tmp_path):
    """D-20: the selected eta's run must be complete before anything is written or promoted."""

    ctx, directory, inputs, _ = write_profile(tmp_path, overshoot_grid(D), D)
    common.eta_run_paths(ctx.cfg, 2.0)["vecnorm_path"].unlink()
    with pytest.raises(FileNotFoundError, match="eta=2.0 run is incomplete"):
        profile_b3_dual.select(ctx, directory, inputs, seed=42)
    assert final_b3_files(ctx) == {}
    for name in profile_b3_dual.SELECTION_OUTPUTS:
        assert not (directory / name).exists()


D17_RECORD = {
    "d": D, "cvar_stochastic": 0.625, "cvar_deterministic": 0.546875,
    "b1_log_std": [-0.25, -0.5], "b1_std": [math.exp(-0.25), math.exp(-0.5)],
}


def test_summary_reports_the_d17_inputs_every_eta_series_and_the_sacct_cross_check():
    histories = overshoot_grid(D)
    histories[8.0].loc[3, "kl_stopped"] = True  # update 4 stopped on the KL limit
    histories[8.0].loc[4, "adam_damping_ratio"] = np.nan  # update 5 took no Adam step
    histories[8.0].loc[3, "action_shift_vs_b1_gamma"] = 0.0625  # D-21: update 4 moved gamma's mean further
    histories[8.0].loc[3, "value_loss"] = 0.125  # D-23: the critic's loss fell at update 4
    histories[8.0].loc[4, "explained_variance"] = np.nan  # D-23: SB3's NaN, a batch whose returns did not vary
    # uneven updates: the mean (63.0) differs from the first, the median and any running mean
    histories[8.0]["elapsed_seconds"] = np.cumsum([120.0, 50, 60, 70, 40, 60, 60, 80, 30, 60])
    histories[2.0]["peak_rss_so_far_gib"] = 2.5 + 0.125 * np.arange(10)  # measured, as on Linux
    histories[2.0]["peak_rss_source"] = "resource ... (Linux)"
    page = profile_b3_dual.render_summary(
        histories, summaries_of(histories), 2.0, None, D17_RECORD, fill_paths=16, seed=42, written="at a test time",
    )

    assert page.startswith("# B3 dual step size: D-12a summary\n\n**Selected: η_λ = 2.0.**")
    assert "`export B3_LAMBDA_LR=2.0`" in page and "Tie by construction" not in page
    assert "N = 16 fill paths; training seed 42; λ_max = 500. Written by `profile_b3_dual.py --select`, at a test time." in page
    # B1's sigma per action dimension, d and both CVaRs from the D-17 record
    assert "| γ | -0.25 | 0.7788 |\n| skew | -0.5 | 0.6065 |" in page
    assert "- d = 0.5, i.e. 0.8 × the stochastic CVaR₀.₂ of B1's full-day MaxDD." in page
    assert (
        "- CVaR₀.₂ stochastic = 0.625 (used for d); deterministic = 0.546875 (recorded, not used); "
        "deterministic − stochastic = -0.078125 (-12.5 % of the stochastic value)."
    ) in page
    assert profile_b3_dual.SELECTION_RULE_TEXT in page
    # per eta: max lambda, sign changes, criterion, exclusion, KL stops, seconds per update, peak memory
    assert "| 0.5 | 1.25 | 0 | 0.25 | yes | — | 0 of 10 | 60.0 | n/a (unavailable on this platform) |" in page
    assert "| 2 | 1.5 | 2 | 0.0625 | yes | — | 0 of 10 | 60.0 | 3.625 |" in page
    assert "| 8 | 4 | 1 | 0.3125 | yes | — | 1 of 10 | 63.0 | n/a (unavailable on this platform) |" in page
    # the lambda and H/d series, epochs, KL stops, damping ratio, action shift, the critic's value loss and
    # explained variance, seconds and memory, update by update
    assert "### η = 8\n" in page
    assert "| 4 | 1.5 | 0 | 0.6875 | 10 | yes | 0.800 | 0.0625 / 0.0250 | 0.125 | 0.250 | 70.0 | n/a |" in page
    assert "| 5 | 0 | 0 | 0.6875 | 10 | no | n/a | 0.0125 / 0.0250 | 0.5 | n/a | 40.0 | n/a |" in page
    assert "| 10 | 1.125 | 1.25 | 1.0625 | 10 | no | 0.800 | 0.0125 / 0.0250 | 0.5 | 0.250 | 60.0 | 3.625 |" in page
    assert "| adam_damping_ratio | action shift vs B1, γ / skew | value_loss | explained_variance | seconds |" in page
    # the memory cross-check the author asked for, and the next step
    assert "\n    sacct -j <jobid> --format=JobID,MaxRSS\n" in page
    assert "`logs/b3_dual_<jobid>.out`" in page and "`.batch` step" in page and "`--mem`" in page
    assert (
        "- Set `B3_LAMBDA_LR=2.0` in `configs/final_pipeline.env` (k = 10 is frozen, D-19), commit, and submit "
        "`PIPELINE_STAGE=final` after the go/no-go review of the whole profile stage (D-19)."
    ) in page
    assert "if k differs from 10" not in page and "B2_UPDATES" not in page  # k is no longer the author's to set

    tie = profile_b3_dual.render_summary(
        histories, summaries_of(histories), 0.5, profile_b3_dual.TIE_BY_CONSTRUCTION_NOTE, D17_RECORD,
        fill_paths=16, seed=42, written="at a test time",
    )
    assert f"**Tie by construction.** {profile_b3_dual.TIE_BY_CONSTRUCTION_NOTE}" in tie

    excluded = all_excluded_grid()
    stop = profile_b3_dual.render_summary(
        excluded, summaries_of(excluded), None, None, D17_RECORD, fill_paths=16, seed=42, written="at a test time",
    )
    assert "**No η selected: every η is excluded.**" in stop and "export B3_LAMBDA_LR" not in stop
    # D-15b: when every eta fails, B1's sigma and the deterministic-vs-stochastic CVaR gap are reported
    assert "| γ | -0.25 | 0.7788 |\n| skew | -0.5 | 0.6065 |" in stop
    assert "deterministic − stochastic = -0.078125 (-12.5 % of the stochastic value)." in stop
    assert "| no | H - d changed sign 4 times (more than 2) |" in stop
    assert "| no | lambda reached lambda_max = 500 |" in stop
    assert "- Stop and report to the author. B3 cannot be trained until D-12a is amended." in stop


@pytest.mark.parametrize(("change", "match"), [
    (lambda record: record.pop("cvar_deterministic"), "lacks"),
    (lambda record: record.update(b1_std=[1.0, 1.0, 1.0]), "one value per action"),
    (lambda record: record.update(b1_log_std=[0.0, float("nan")]), "non-finite"),
    (lambda record: record.update(cvar_stochastic="0.625"), "non-finite"),
    (lambda record: record.update(b1_std=[True, False]), "non-numeric"),  # JSON booleans are not numbers here
])
def test_the_d17_record_behind_the_summary_is_checked(tmp_path, change, match):
    ctx, *_ = write_profile(tmp_path, overshoot_grid(D), D)
    assert profile_b3_dual.read_d17_record(ctx, 42) == {
        "d": D, "cvar_stochastic": 0.625, "cvar_deterministic": 0.546875,
        "b1_log_std": [-0.25, -0.5], "b1_std": [math.exp(-0.25), math.exp(-0.5)],
    }
    path = ctx.cfg.result_path("cvar_threshold_b3.json")
    record = json.loads(path.read_text(encoding="utf-8"))
    change(record)
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match=f"{match}.*D-17"):
        profile_b3_dual.read_d17_record(ctx, 42)


@pytest.mark.parametrize(("argv", "environment_seed"), [
    (["--seed", "43"], None),
    (["--select", "--seed", "43"], None),
    ([], "43"),
])
def test_profiling_refuses_a_training_seed_other_than_42(monkeypatch, argv, environment_seed):
    if environment_seed is None:
        monkeypatch.delenv("TRAINING_SEED", raising=False)
    else:
        monkeypatch.setenv("TRAINING_SEED", environment_seed)
    monkeypatch.setattr(profile_b3_dual, "load_context", lambda **kwargs: pytest.fail("loaded data before the seed check"))
    monkeypatch.setattr(sys, "argv", ["profile_b3_dual.py", *argv])
    with pytest.raises(ValueError, match="D-12a profiles eta with training seed 42, not 43"):
        profile_b3_dual.main()
