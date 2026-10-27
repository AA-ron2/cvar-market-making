"""Training-only choice of the B3 dual step size eta_lambda (DECISIONS D-12a).

For each eta in {0.5, 2, 8}, B3 runs 10 updates on Jan 1-6 from the final B1,
with its frozen normalizer, the stochastic d (D-17), the production
TRAINING_FILL_PATHS and training seed 42. An eta is excluded if lambda reaches
lambda_max or the violation H_u - d changes sign more than twice; among the
rest, the smallest mean |H_u - d|/d over updates 8-10 is chosen, ties going to
the smaller eta; if the lambda used at every update is 0 for every eta, the runs
coincide and their criteria tie by construction (author, 2026-10-07). If every
eta is excluded, selection stops: there is no fallback. Only the six B3
training days are loaded (never Jan 7-29), and every output goes to
results/profiling/b3_dual. Each history also records the task's peak memory
after every update (peak_rss_so_far_gib; null on Windows).

Slurm: one eta per array task (``--eta-index`` or SLURM_ARRAY_TASK_ID), then
``--select``. Without either flag all three run in sequence and are selected.
The selection records the SHA-256 of the B1, its normalizer, d and the
calibration so that the final pipeline stage can verify it reuses them.

D-20: each run is exactly the final B3's configuration, so each task also saves
the run as train_formula_b3.py would (model, frozen normalizer, history, costs,
feasibility, and job metadata with its load-time hashes and commit), and
``--select`` promotes the selected run, by copy, to the final B3's names. The
final stage evaluates that B3 and does not retrain it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

if __package__:
    from .common import (
        B3_RUN_FILES,
        b1_artifact_stems,
        b3_run_paths,
        eta_run_paths,
        eta_task_job_name,
        file_sha256,
        get_git_commit,
        load_context,
        load_manifest_arrays,
        manifest_path,
        peak_rss_so_far_gib,
        print_context_banner,
        seeded_result_name,
        sha256_lines,
        train_params_json_path,
        training_fill_paths,
        write_metadata,
    )
    from .train_formula_b3 import (
        b3_provenance_at_start,
        load_b3_threshold,
        run_b3,
        save_b3_run,
        update_diagnostics_columns,
    )
else:
    from common import (  # type: ignore[no-redef]
        B3_RUN_FILES,
        b1_artifact_stems,
        b3_run_paths,
        eta_run_paths,
        eta_task_job_name,
        file_sha256,
        get_git_commit,
        load_context,
        load_manifest_arrays,
        manifest_path,
        peak_rss_so_far_gib,
        print_context_banner,
        seeded_result_name,
        sha256_lines,
        train_params_json_path,
        training_fill_paths,
        write_metadata,
    )
    from train_formula_b3 import (  # type: ignore[no-redef]
        b3_provenance_at_start,
        load_b3_threshold,
        run_b3,
        save_b3_run,
        update_diagnostics_columns,
    )
from cvar_mm.gym.cvar_lagrangian import B3_LAMBDA_MAX

PROFILE_DATES = tuple(f"2025-01-0{day}" for day in range(1, 7))
ETA_GRID = (0.5, 2.0, 8.0)
PROFILE_UPDATES = 10
CRITERION_UPDATES = (8, 9, 10)  # D-12a: mean |H_u - d|/d over these update labels
MAX_SIGN_CHANGES = 2  # D-12a: more sign changes of H_u - d than this exclude an eta
PROFILE_TRAINING_SEED = 42  # D-12a; the profiling outputs are not seed-specific
SELECTION_RULE = "D-12a"
SELECTION_RULE_TEXT = (
    "eta_lambda in {0.5, 2, 8}, 10 B3 updates each on Jan 1-6 with training seed 42. Exclude an eta if "
    "lambda after any update reaches lambda_max, or if H_u - d changes sign more than twice over the 10 "
    "updates (an exact zero is not a sign). Among the rest choose the smallest mean |H_u - d|/d over "
    "updates 8-10; ties go to the smaller eta. If the lambda used at every update is 0 for every eta, the "
    "runs coincide and their criteria tie by construction. If every eta is excluded, stop: there is no "
    "fallback (DECISIONS D-12a)."
)
TIE_BREAK = "the smaller eta, on an exact tie or a tie by construction"
TIE_BY_CONSTRUCTION_NOTE = (
    "The lambda used at every update was 0 for every eta, so the runs coincide and their criteria tie by "
    "construction: the smallest kept eta is chosen (D-12a clarification, 2026-10-07)."
)
# The columns the D-12a rule and selection.json read; the D-15 diagnostics may be empty.
RULE_COLUMNS = (
    "update", "eta", "lambda_used", "lambda", "violation", "ru_cvar",
    "threshold_d", "training_fill_paths", "elapsed_seconds",
)
# What --select writes; each run first moves the previous ones to superseded/<UTC>/.
SELECTION_OUTPUTS = ("selected_lambda_lr.txt", "selection.json", "inputs.sha256", "summary.md")
# summary.md reports these D-17 fields; the policy's action order is common.write_metadata's "policy_action".
D17_SUMMARY_FIELDS = ("d", "cvar_stochastic", "cvar_deterministic", "b1_log_std", "b1_std")
POLICY_ACTIONS = ("γ", "skew")


def select_profiling_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return exactly the Jan 1-6 training rows, or fail closed."""

    frame = manifest.copy()
    frame["date"] = frame["date"].astype(str)
    selected = frame[frame["split"] == "train"].sort_values("date").reset_index(drop=True)
    if tuple(selected["date"]) != PROFILE_DATES:
        raise ValueError("B3 dual profiling requires exactly the Jan 1-6, 2025 training rows")
    return selected


def profiled_inputs(ctx, seed: int) -> list[Path]:
    """Artifacts eta is profiled on; the final stage must reuse exactly these."""

    model_stem, vecnorm_stem = b1_artifact_stems("final", seed)
    return [
        ctx.cfg.model_path(f"{model_stem}.zip"),
        ctx.cfg.vecnorm_path(vecnorm_stem),
        ctx.cfg.result_path(seeded_result_name("cvar_threshold_d.txt", seed)),
        ctx.cfg.result_path(seeded_result_name("cvar_threshold_b3.json", seed)),
        train_params_json_path(ctx.cfg),
        manifest_path(ctx.cfg),
    ]


def validate_history(history: pd.DataFrame, lambda_max: float = B3_LAMBDA_MAX) -> None:
    """Refuse a profiling history the D-12a rule cannot be applied to."""

    missing = [column for column in RULE_COLUMNS if column not in history.columns]
    if missing:
        raise ValueError(f"An eta profiling history lacks {missing}; D-12a cannot be applied")
    if history["update"].tolist() != list(range(1, PROFILE_UPDATES + 1)):
        raise ValueError(f"D-12a needs updates 1..{PROFILE_UPDATES} in order, got {history['update'].tolist()}")
    if not np.isfinite(history[list(RULE_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError(f"An eta profiling history has non-finite values in {list(RULE_COLUMNS)} (D-12a)")
    lambdas = history["lambda"].to_numpy(dtype=float)
    if lambdas.min() < 0.0 or lambdas.max() > lambda_max:
        raise ValueError(f"lambda outside [0, {lambda_max:g}] in an eta profiling history (D-12a)")
    lambda_used = history["lambda_used"].to_numpy(dtype=float)
    if lambda_used[0] != 0.0 or not np.array_equal(lambda_used[1:], lambdas[:-1]):
        raise ValueError("lambda_used must be lambda one update earlier, starting from lambda_0 = 0 (D-12a)")
    d = history["threshold_d"].to_numpy(dtype=float)
    if not (d[0] > 0.0 and (d == d[0]).all()):
        raise ValueError("threshold_d must be one positive value per eta profiling history (D-12a)")


def dual_profile_summary(history: pd.DataFrame, lambda_max: float = B3_LAMBDA_MAX) -> dict:
    """D-12a screens and criterion for one eta's history (fails closed on a malformed one)."""

    validate_history(history, lambda_max)
    violations = history["violation"].to_numpy(dtype=float)  # H_u - d, in USDT
    d = float(history["threshold_d"].iloc[0])
    signs = np.sign(violations)
    signs = signs[signs != 0]  # an exact zero is not a sign of its own
    sign_changes = int(np.count_nonzero(signs[1:] != signs[:-1]))
    max_lambda = float(history["lambda"].max())  # lambda after each update, lambda_10 included
    window = history["update"].isin(CRITERION_UPDATES).to_numpy()
    criterion = float(np.mean(np.abs(violations[window]) / d))
    reasons = []
    if max_lambda >= lambda_max:
        reasons.append(f"lambda reached lambda_max = {lambda_max:g}")
    if sign_changes > MAX_SIGN_CHANGES:
        reasons.append(f"H - d changed sign {sign_changes} times (more than {MAX_SIGN_CHANGES})")
    return {
        "max_lambda": max_lambda,
        "violation_sign_changes": sign_changes,
        "criterion": criterion,
        "kept": not reasons,
        "exclusion_reason": "; ".join(reasons) if reasons else None,
    }


def d12a_choice(histories: dict[float, pd.DataFrame], summaries: dict[float, dict]) -> tuple[float | None, str | None]:
    """The D-12a choice and its note; None when every eta is excluded."""

    kept = sorted(eta for eta, summary in summaries.items() if summary["kept"])
    if not kept:
        return None, None
    if all((history["lambda_used"] == 0.0).all() for history in histories.values()):
        # lambda enters B3 only through the lambda used in each batch, so these runs are one
        # algorithm and their criteria are equal in exact arithmetic (author's choice, 2026-10-07).
        return kept[0], TIE_BY_CONSTRUCTION_NOTE
    return min(kept, key=lambda eta: (summaries[eta]["criterion"], eta)), None


def no_eta_message(summaries: dict[float, dict]) -> str:
    reasons = "; ".join(f"eta={eta}: {summary['exclusion_reason']}" for eta, summary in summaries.items())
    return f"Every eta_lambda is excluded by D-12a ({reasons}). Stop and report: D-12a has no fallback value."


def select_lambda_lr(histories: dict[float, pd.DataFrame], lambda_max: float = B3_LAMBDA_MAX) -> float:
    """The D-12a choice; raises when every eta is excluded."""

    summaries = {eta: dual_profile_summary(history, lambda_max) for eta, history in histories.items()}
    chosen, _ = d12a_choice(histories, summaries)
    if chosen is None:
        raise RuntimeError(no_eta_message(summaries))
    return chosen


def history_path(directory: Path, eta: float) -> Path:
    return directory / f"history_eta{eta}.csv"


def inputs_path(directory: Path, eta: float | None = None) -> Path:
    return directory / ("inputs.sha256" if eta is None else f"inputs_eta{eta}.sha256")


def archive_previous_selection(directory: Path, promoted: list[Path] = ()) -> Path | None:
    """Move the last --select's outputs aside, the B3 it promoted (``promoted``, D-20) included, so a selection
    that fails after this leaves no earlier choice and no final B3.

    D-12a has no fallback value. A --select that fails before or during this move leaves the earlier outputs
    in place but fails loudly; item 7's histories_sha256 check covers a choice left beside newer histories.
    """

    present = [directory / name for name in SELECTION_OUTPUTS if (directory / name).exists()]
    present += [path for path in promoted if path.exists()]
    if not present:
        return None
    archive = directory / "superseded" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    archive.mkdir(parents=True)  # a new folder; an existing one is an error, never merged into
    for path in present:
        path.replace(archive / path.name)
    print(f"Moved the previous selection outputs {[path.name for path in present]} to {archive}")
    return archive


def read_profiled_histories(directory: Path, current: str) -> dict[float, pd.DataFrame]:
    """Every eta's history, profiled on the current inputs and parsed exactly."""

    histories: dict[float, pd.DataFrame] = {}
    for eta in ETA_GRID:
        path = history_path(directory, eta)
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; profile every eta first")
        if inputs_path(directory, eta).read_text(encoding="utf-8") != current:
            raise RuntimeError(f"eta={eta} was profiled on other B1/d/calibration files than the current ones")
        # round_trip: pandas' default float parser is not exact, and d is compared exactly.
        history = pd.read_csv(path, float_precision="round_trip")
        if "eta" not in history.columns or not (history["eta"] == eta).all():
            raise ValueError(f"{path.name} does not hold eta={eta} throughout (D-12a)")
        histories[eta] = history
    return histories


class PeakRssPerUpdate(BaseCallback):
    """``peak_rss_so_far_gib`` after every PPO update, read where the D-15 row stamps ``elapsed_seconds``.

    SB3 has no hook after PPO.train(): the next rollout's start, or the end of
    training after the last update, is the first point after it. The value is
    the process's running maximum, so with one eta per Slurm task its last value
    is that task's peak. Each line is flushed, so a task killed for time or
    memory still leaves its memory trace in the log.
    """

    def __init__(self) -> None:
        super().__init__()
        self.values: list[float | None] = []
        self.source: str | None = None
        self._rollouts_started = 0

    def _sample(self) -> None:
        value, source = peak_rss_so_far_gib()
        if self.source not in (None, source):
            raise RuntimeError(f"The peak-memory source changed from {self.source!r} to {source!r}")
        self.source = source
        self.values.append(value)
        shown = "n/a" if value is None else f"{value:.3f} GiB"
        print(f"after update {len(self.values)}: peak RSS so far {shown} ({source})", flush=True)

    def _on_rollout_start(self) -> None:
        if self._rollouts_started:  # the previous update's PPO.train() has finished
            self._sample()
        self._rollouts_started += 1

    def _on_training_end(self) -> None:
        self._sample()

    def _on_step(self) -> bool:
        return True


def profile_eta(ctx, data, eta: float, *, threshold_d: float, fill_paths: int, seed: int):
    """One eta's 10 B3 updates: its D-12a history and the run itself (D-20: it may become the final B3)."""

    memory = PeakRssPerUpdate()
    run = run_b3(
        ctx,
        *data,
        threshold_d=threshold_d,
        lambda_lr=eta,
        updates=PROFILE_UPDATES,
        fill_paths=fill_paths,
        seed=seed,
        tensorboard_log=None,
        extra_callbacks=[memory],
    )
    callback = run.callback
    if len(memory.values) != len(callback.lambda_history):
        raise RuntimeError(f"{len(memory.values)} peak-memory samples for {len(callback.lambda_history)} updates")
    history = pd.DataFrame({
        "update": range(1, len(callback.lambda_history) + 1),
        "eta": eta,
        "lambda_used": callback.lambda_used_history,
        "lambda": callback.lambda_history,
        "nu": callback.nu_history,
        "ru_cvar": callback.cvar_history,
        "violation": callback.violation_history,
        "mean_full_day_maxdd": callback.mean_maxdd_history,
        **update_diagnostics_columns(callback),  # D-15, incl. elapsed_seconds
        "threshold_d": threshold_d,
        "training_fill_paths": fill_paths,
        "peak_rss_so_far_gib": memory.values,  # null where not measured (Windows)
        "peak_rss_source": memory.source,
    })
    return history, run


def promote_b3(cfg, eta: float, seed: int) -> dict:
    """D-20: copy the selected eta run's B3 files to the final B3's names, byte for byte, and return their
    SHA-256 for selection.json (the profile record checks the files against it)."""

    sources, targets = eta_run_paths(cfg, eta), b3_run_paths(cfg, seed)
    files = {}
    for key in B3_RUN_FILES:
        shutil.copyfile(sources[key], targets[key])
        digest = file_sha256(targets[key])
        if digest != file_sha256(sources[key]):
            raise RuntimeError(f"The copy of {sources[key]} differs from it (D-20)")
        files[key] = {
            "from": sources[key].relative_to(cfg.repo_root).as_posix(),
            "to": targets[key].relative_to(cfg.repo_root).as_posix(),
            "sha256": digest,
        }
    return {"eta": eta, "files": files}


def read_d17_record(ctx, seed: int) -> dict:
    """The D-17 fields summary.md reports, from the d JSON that load_b3_threshold has vetted."""

    path = ctx.cfg.result_path(seeded_result_name("cvar_threshold_b3.json", seed))
    record = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in D17_SUMMARY_FIELDS if key not in record]
    if missing:
        raise ValueError(f"{path.name} lacks {missing}; rerun calibrate_cvar_b3.py (DECISIONS D-17)")
    values = [record["d"], record["cvar_stochastic"], record["cvar_deterministic"]]
    for key in ("b1_log_std", "b1_std"):
        if not isinstance(record[key], list) or len(record[key]) != len(POLICY_ACTIONS):
            raise ValueError(f"{path.name}: {key} must give one value per action {POLICY_ACTIONS} (DECISIONS D-17)")
        values += record[key]
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in values
    ):
        raise ValueError(f"{path.name} has a non-numeric or non-finite D-17 field (DECISIONS D-17)")
    return {key: record[key] for key in D17_SUMMARY_FIELDS}


def _number(value, spec: str = ".6g") -> str:
    return "n/a" if value is None or not math.isfinite(float(value)) else format(float(value), spec)


def render_summary(
    histories: dict[float, pd.DataFrame],
    summaries: dict[float, dict],
    chosen: float | None,
    note: str | None,
    d17: dict,
    *,
    fill_paths: int,
    seed: int,
    written: str,
) -> str:
    """summary.md, the η page of the author's go/no-go review of the profile stage (WORK_ORDER item 3; D-15b, D-19)."""

    d = d17["d"]
    gap = d17["cvar_deterministic"] - d17["cvar_stochastic"]
    if chosen is None:
        lines = [
            "# B3 dual step size: D-12a summary",
            "",
            "**No η selected: every η is excluded.** Stop and report to the author: D-12a has no fallback "
            "value, and no `selected_lambda_lr.txt` was written.",
        ]
    else:
        lines = [
            "# B3 dual step size: D-12a summary",
            "",
            f"**Selected: η_λ = {chosen}.** The line for `configs/final_pipeline.env` is "
            f"`export B3_LAMBDA_LR={chosen}`.",
        ]
    if note is not None:
        lines += ["", f"**Tie by construction.** {note} The constraint never bound within the profiled horizon."]
    lines += [
        "",
        f"Jan 1–6 only (Jan 7–29 never loaded); {PROFILE_UPDATES} B3 updates per η ∈ "
        f"{{{', '.join(f'{eta:g}' for eta in ETA_GRID)}}}; N = {fill_paths} fill paths; training seed {seed}; "
        f"λ_max = {B3_LAMBDA_MAX:g}. Written by `profile_b3_dual.py --select`, {written}.",
        "",
        "## B1 and the threshold (D-17)",
        "",
        "B1's policy noise, fixed at σ₀ = POLICY_STD in B1 (D-21) and frozen during B3 (D-15b), in PPO action "
        "units on [−1, 1]:",
        "",
        "| Action | log σ | σ |",
        "|---|---|---|",
        *(
            f"| {action} | {_number(log_std, '.4g')} | {_number(std, '.4g')} |"
            for action, log_std, std in zip(POLICY_ACTIONS, d17["b1_log_std"], d17["b1_std"])
        ),
        "",
        f"- d = {_number(d)}, i.e. 0.8 × the stochastic CVaR₀.₂ of B1's full-day MaxDD.",
        f"- CVaR₀.₂ stochastic = {_number(d17['cvar_stochastic'])} (used for d); deterministic = "
        f"{_number(d17['cvar_deterministic'])} (recorded, not used); deterministic − stochastic = {_number(gap)} "
        f"({_number(100 * gap / d17['cvar_stochastic'], '.3g')} % of the stochastic value).",
        "",
        "## Rule (D-12a)",
        "",
        SELECTION_RULE_TEXT,
        "",
        "## Per η",
        "",
        "| η | max λ | sign changes of Ĥ − d | criterion: mean \\|Ĥ − d\\|/d, updates 8–10 | kept | exclusion "
        "reason | KL stops | mean s / update | peak RSS (GiB) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for eta, history in histories.items():
        summary = summaries[eta]
        memory = history["peak_rss_so_far_gib"].to_numpy(dtype=float)
        last_memory = _number(memory[-1], ".3f")
        if last_memory == "n/a":
            last_memory = f"n/a ({history['peak_rss_source'].iloc[0]})"
        lines.append(
            f"| {eta:g} | {_number(summary['max_lambda'], '.4g')} | {summary['violation_sign_changes']} | "
            f"{_number(summary['criterion'])} | {'yes' if summary['kept'] else 'no'} | "
            f"{summary['exclusion_reason'] or '—'} | {int(history['kl_stopped'].eq(True).sum())} of {len(history)} | "
            f"{_number(history['elapsed_seconds'].iloc[-1] / len(history), '.1f')} | {last_memory} |"
        )
    for eta, history in histories.items():
        seconds = np.diff(history["elapsed_seconds"].to_numpy(dtype=float), prepend=0.0)
        lines += [
            "",
            f"### η = {eta:g}",
            "",
            "| update | λ used | λ after | Ĥ/d | epochs completed | KL stop | adam_damping_ratio | "
            "action shift vs B1, γ / skew | value_loss | explained_variance | seconds | peak RSS so far (GiB) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for row, second in zip(history.to_dict("records"), seconds):  # "lambda" is not an attribute name
            lines.append(
                f"| {row['update']} | {_number(row['lambda_used'], '.4g')} | {_number(row['lambda'], '.4g')} | "
                f"{_number(row['ru_cvar'] / row['threshold_d'], '.4f')} | {_number(row['epochs_completed'], '.0f')} | "
                f"{'yes' if row['kl_stopped'] == True else 'no'} | "  # noqa: E712 (a NumPy bool, or NaN if empty)
                f"{_number(row['adam_damping_ratio'], '.3f')} | "
                f"{_number(row['action_shift_vs_b1_gamma'], '.4f')} / {_number(row['action_shift_vs_b1_skew'], '.4f')} | "
                f"{_number(row['value_loss'], '.4g')} | {_number(row['explained_variance'], '.3f')} | "  # D-23: the critic
                f"{_number(second, '.1f')} | "
                f"{_number(row['peak_rss_so_far_gib'], '.3f')} |"
            )
    lines += [
        "",
        "## Memory cross-check",
        "",
        "`peak_rss_so_far_gib` is the process's running maximum of `ru_maxrss` after each update. With one η "
        "per Slurm array task (the layout `submit_final_pipeline.sh` submits), an η's last value is that "
        "task's peak; if the three η ran in one process (no `--eta-index` and no `SLURM_ARRAY_TASK_ID`), each "
        "value is the maximum since that process started, so a later η's values include the earlier η's "
        "peaks. Cross-check it against Slurm's own record:",
        "",
        "    sacct -j <jobid> --format=JobID,MaxRSS",
        "",
        "Each task's job ID is in its log name, `logs/b3_dual_<jobid>.out`; the array job ID that "
        "`submit_final_pipeline.sh` prints as \"eta profile\" covers all three (task i is ETA_GRID[i]). Compare "
        "with the `.batch` step, and with the `--mem` the jobs request (`scripts/pipeline/run_python_job.sh`).",
        "",
        "## Next step",
        "",
    ]
    if chosen is None:
        lines.append("- Stop and report to the author. B3 cannot be trained until D-12a is amended.")
    else:
        lines += [
            f"- Set `B3_LAMBDA_LR={chosen}` in `configs/final_pipeline.env` (k = 10 is frozen, D-19), commit, and "
            "submit `PIPELINE_STAGE=final` after the go/no-go review of the whole profile stage (D-19).",
        ]
    return "\n".join(lines) + "\n"


def select(ctx, directory: Path, inputs: list[Path], *, seed: int) -> float:
    final_b3 = b3_run_paths(ctx.cfg, seed)
    # first, so a refused selection leaves no choice and no final B3 behind (D-12a, D-20)
    archive_previous_selection(directory, list(final_b3.values()))
    current = sha256_lines(ctx.cfg.repo_root, inputs)
    histories = read_profiled_histories(directory, current)
    summaries = {eta: dual_profile_summary(history) for eta, history in histories.items()}
    d = load_b3_threshold(ctx, seed)
    for eta, history in histories.items():
        if not (history["threshold_d"] == d).all():
            raise ValueError(f"eta={eta} was profiled with a d other than the current D-17 d = {d!r}")
    fill_paths = sorted({int(value) for history in histories.values() for value in history["training_fill_paths"]})
    if len(fill_paths) != 1:
        raise ValueError(f"The eta histories use different fill path counts {fill_paths}; D-12a needs one N")
    d17 = read_d17_record(ctx, seed)
    chosen, note = d12a_choice(histories, summaries)
    git_commit = get_git_commit()
    timestamp = datetime.now(timezone.utc).isoformat()
    summary_page = render_summary(
        histories, summaries, chosen, note, d17,
        fill_paths=fill_paths[0], seed=seed, written=f"{timestamp} (git {git_commit})",
    )
    if chosen is None:
        # D-12a: stop and report. The page is the report D-15b asks for; no choice is written.
        (directory / "summary.md").write_text(summary_page, encoding="utf-8")
        raise RuntimeError(f"{no_eta_message(summaries)} See {directory / 'summary.md'}.")
    missing = [str(path) for path in eta_run_paths(ctx.cfg, chosen).values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"The selected eta={chosen} run is incomplete, so no B3 can be promoted (D-20): {missing}")
    (directory / "summary.md").write_text(summary_page, encoding="utf-8")
    promoted = promote_b3(ctx.cfg, chosen, seed)  # D-20: the final B3, before the choice is written
    payload = {
        "rule": SELECTION_RULE,
        "rule_text": SELECTION_RULE_TEXT,
        "eta_grid": list(ETA_GRID),
        "updates_per_eta": PROFILE_UPDATES,
        "criterion_updates": list(CRITERION_UPDATES),
        "max_sign_changes": MAX_SIGN_CHANGES,
        "tie_break": TIE_BREAK,
        "lambda_max": B3_LAMBDA_MAX,
        "selected_lambda_lr": chosen,
        "selection_note": note,
        "per_eta": {
            str(eta): {
                **summaries[eta],
                "seconds_per_update": float(history["elapsed_seconds"].iloc[-1] / len(history)),
            }
            for eta, history in histories.items()
        },
        "threshold_d": d,
        "training_fill_paths": fill_paths[0],
        "training_seed": seed,
        "inputs_sha256": current.splitlines(),
        "histories_sha256": sha256_lines(
            ctx.cfg.repo_root, [history_path(directory, eta) for eta in ETA_GRID]
        ).splitlines(),
        "promoted_b3": promoted,
        "dates_used": list(PROFILE_DATES),
        "excluded": "January 7-29 is never loaded",
        "git_commit": git_commit,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "timestamp_utc": timestamp,
    }
    selection_text = json.dumps(payload, indent=2, allow_nan=False)
    inputs_path(directory).write_text(current, encoding="utf-8", newline="\n")  # LF: sha256sum --check reads it
    (directory / "selection.json").write_text(selection_text, encoding="utf-8")
    # Last: the final stage requires this file, so it exists only once everything else is written.
    (directory / "selected_lambda_lr.txt").write_text(f"{chosen}\n", encoding="utf-8")
    print(f"Promoted the eta={chosen} run to the final B3: {[file['to'] for file in promoted['files'].values()]}")
    print(f"Selected eta_lambda={chosen}. Review {directory / 'summary.md'}, then put this line in configs/final_pipeline.env:")
    print(f"export B3_LAMBDA_LR={chosen}")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the B3 dual step size on Jan 1-6 (DECISIONS D-12a).")
    parser.add_argument("--eta-index", type=int, default=None, help="index into ETA_GRID; default SLURM_ARRAY_TASK_ID")
    parser.add_argument("--select", action="store_true", help="apply the D-12a rule to the saved histories")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("TRAINING_SEED", 42)))
    args = parser.parse_args()
    if args.seed != PROFILE_TRAINING_SEED:
        raise ValueError(
            f"D-12a profiles eta with training seed {PROFILE_TRAINING_SEED}, not {args.seed}; "
            "the profiling outputs are not seed-specific"
        )

    ctx = load_context(require_prepared=True)  # M2: never prepares the split or calibrates here
    print_context_banner(ctx, "profile_b3_dual")
    directory = ctx.cfg.results_dir / "profiling" / "b3_dual"
    directory.mkdir(parents=True, exist_ok=True)
    inputs = profiled_inputs(ctx, args.seed)
    if args.select:
        select(ctx, directory, inputs, seed=args.seed)
        return

    task = args.eta_index if args.eta_index is not None else os.environ.get("SLURM_ARRAY_TASK_ID")
    etas = ETA_GRID if task is None else (ETA_GRID[int(task)],)
    data = load_manifest_arrays(ctx, select_profiling_manifest(ctx.manifest), label="B3 dual profiling")
    threshold_d = load_b3_threshold(ctx, args.seed)
    fill_paths = training_fill_paths()
    hashes = sha256_lines(ctx.cfg.repo_root, inputs)
    provenance_at_start = b3_provenance_at_start(ctx, args.seed)  # D-20: each run may become the final B3
    for eta in etas:
        history, run = profile_eta(ctx, data, eta, threshold_d=threshold_d, fill_paths=fill_paths, seed=args.seed)
        history.to_csv(history_path(directory, eta), index=False)
        inputs_path(directory, eta).write_text(hashes, encoding="utf-8", newline="\n")
        payload = save_b3_run(
            ctx, run, eta_run_paths(ctx.cfg, eta), threshold_d=threshold_d, lambda_lr=eta, updates=PROFILE_UPDATES,
            fill_paths=fill_paths, seed=args.seed, train_dates=data[2], provenance_at_start=provenance_at_start,
        )
        write_metadata(eta_task_job_name(eta), {
            **payload,
            "eta": eta,
            "d12a_history_path": history_path(directory, eta),
            "profiled_inputs_path": inputs_path(directory, eta),
        })
        print(f"Saved eta={eta} history -> {history_path(directory, eta)}, and its B3 run (D-20)")
    if task is None:
        select(ctx, directory, inputs, seed=args.seed)


if __name__ == "__main__":
    main()
