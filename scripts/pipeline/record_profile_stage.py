"""Record what the final stage takes from the profile stage (DECISIONS D-19, D-20; work-order item 4).

The final stage evaluates B1, the final B2, B1+ and B3, which the profile stage trained: B3 is the eta-profiling
run D-12a selected, promoted by copy (D-20), and the final B2 is the parallel final B2 of the selected alpha,
promoted by copy (D-22). After the B2 promotion, B1+ and the eta selection, this job writes two files under
results/profiling/:
- profile_stage.sha256 holds the SHA-256 of fifteen files: B1 and its normalizer, final B2 and its normalizer,
  B1+ and its normalizer, the B2 selection (b2_best_alpha.txt, b2_alpha_sweep_summary.csv), the calibration,
  the manifest, and the promoted B3 (its model and normalizer, history, training costs and feasibility), in
  `sha256sum` format relative to the project, with LF line ends.
- profile_stage.commit holds the commit the profile stage was submitted at. The final stage refuses to run
  if the training code or the frozen configuration differs from it, so B3 runs the code and the settings
  B2 and B1+ ran with (D-11, D-18; the author's choice of 2026-10-07).

submit_final_pipeline.sh writes profile_stage.submitted (that commit and the UTC time) when it submits the
profile stage from a clean checkout. Before writing anything, this job checks the stage against it:
- HEAD has the submitted commit's training code, with nothing uncommitted or untracked;
- every job of the stage ran that code: each job metadata written since the submission (results/
  job_metadata/), with no uncommitted training-code change, and the eta selection's commit;
- the final B2 files are the ones b2_promotion.json recorded, of the alpha b2_best_alpha.txt selects, from the
  current selection, and the ones that alpha's parallel final B2 job of this stage saved (D-22); B1+ is the
  file its training job saved. Both ran k = 10, seed 42 and one N, and each hashed its inputs when it loaded
  them (`provenance_at_start`): the current B1 and its normalizer, the current calibration and manifest.
- the final B3 files are the ones selection.json recorded when it promoted them and the ones the selected eta
  task saved; that task started from the submitted training code and the current B1, normalizer, d,
  calibration and manifest, with k = 10, seed 42, final B2's N, the selected eta and the current d (D-20).
A rerun of B1, prep or the selection can therefore not be recorded next to a final B2 or B1+ trained or
promoted before it: rerun every job that depends on the one you reran, then this job. Each run first moves an earlier record
aside, then checks, then writes; a refusal leaves no record.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# common puts the project root on sys.path, so it must precede the cvar_mm imports.
if __package__:
    from .common import (
        B2_RUN_FILES,
        B3_RUN_FILES,
        TRAINING_CODE_PATHS,
        b2_final_paths,
        b2_promotion_path,
        b2_selection_paths,
        b3_inputs,
        b3_run_paths,
        eta_task_job_name,
        file_sha256,
        final_b2_inputs,
        final_b2_training_record,
        final_model_stems,
        load_context,
        manifest_path,
        print_context_banner,
        seeded_result_name,
        sha256_lines,
        train_params_json_path,
        write_metadata,
    )
else:
    from common import (  # type: ignore[no-redef]
        B2_RUN_FILES,
        B3_RUN_FILES,
        TRAINING_CODE_PATHS,
        b2_final_paths,
        b2_promotion_path,
        b2_selection_paths,
        b3_inputs,
        b3_run_paths,
        eta_task_job_name,
        file_sha256,
        final_b2_inputs,
        final_b2_training_record,
        final_model_stems,
        load_context,
        manifest_path,
        print_context_banner,
        seeded_result_name,
        sha256_lines,
        train_params_json_path,
        write_metadata,
    )

# The final stage checks the seed-42 paths; TRAINING_SEED = 42 is frozen (F-07).
RECORD_SEED = 42
RECORD_UPDATES = 10  # k, fixed a priori (D-19)
RECORD_NAME = "profile_stage.sha256"
COMMIT_NAME = "profile_stage.commit"
SUBMITTED_NAME = "profile_stage.submitted"  # written by submit_final_pipeline.sh when it submits the profile stage
TRAINING_SCRIPTS = frozenset(path for path in TRAINING_CODE_PATHS if path.startswith("scripts/"))
EXCLUDED = frozenset(path.removeprefix(":(exclude)") for path in TRAINING_CODE_PATHS if path.startswith(":(exclude)"))


def profile_stage_outputs(cfg) -> list[Path]:
    """The fifteen recorded files, for training seed 42: ten of D-19 and the promoted B3's five (D-20)."""

    paths = []
    for model in ("b1", "b2", "b1plus"):
        model_stem, vecnorm_stem = final_model_stems(model, RECORD_SEED)
        paths += [cfg.model_path(f"{model_stem}.zip"), cfg.vecnorm_path(vecnorm_stem)]
    return paths + [
        cfg.result_path(seeded_result_name("b2_best_alpha.txt", RECORD_SEED)),
        cfg.result_path(seeded_result_name("b2_alpha_sweep_summary.csv", RECORD_SEED)),
        train_params_json_path(cfg),
        manifest_path(cfg),
        *b3_run_paths(cfg, RECORD_SEED).values(),
    ]


def archive_previous_record(directory: Path) -> Path | None:
    """Move an earlier record and commit file aside, so that neither outlives a failed or refused run."""

    present = [directory / name for name in (RECORD_NAME, COMMIT_NAME) if (directory / name).exists()]
    if not present:
        return None
    archive = directory / "superseded" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    archive.mkdir(parents=True)  # a new folder; an existing one is an error, never merged into
    for path in present:
        path.replace(archive / path.name)
    print(f"Moved the previous profile-stage record {[path.name for path in present]} to {archive}")
    return archive


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {root} (D-19): {result.stderr.strip()}")
    return result


def touches_training_code(path: str) -> bool:
    path = path.strip().strip('"')
    if path in EXCLUDED:
        return False
    return path == "cvar_mm" or path.startswith("cvar_mm/") or path in TRAINING_SCRIPTS


def uncommitted_training_code(status: str) -> list[str]:
    """The `git status --porcelain` lines that touch the training code (both sides of a rename)."""

    return [line for line in status.splitlines() if any(touches_training_code(part) for part in line[3:].split(" -> "))]


def differs_in_training_code(root: Path, commit: str, other: str) -> bool:
    git(root, "cat-file", "-e", f"{commit}^{{commit}}")
    return git(root, "diff", "--quiet", commit, other, "--", *TRAINING_CODE_PATHS, check=False).returncode != 0


def read_submission(directory: Path) -> tuple[str, str]:
    path = directory / SUBMITTED_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}: submit PIPELINE_STAGE=profile, which writes it (D-19)")
    commit, submitted_at = path.read_text(encoding="utf-8").split()
    return commit, submitted_at


def ran_submitted_code(root: Path, submitted: str, commit: str | None, status: str | None) -> str | None:
    """None if a job with this commit and training-code status ran the submitted code; else the reason."""

    if not commit:
        return "no commit recorded"
    if status is None:
        return "no git status recorded"
    dirty = uncommitted_training_code(status)
    if dirty:
        return f"uncommitted training code {dirty}"
    if differs_in_training_code(root, commit, submitted):
        return f"training code of {commit}"
    return None


def check_one_training_code(cfg, submitted: str, submitted_at: str) -> list[str]:
    """HEAD and every job of the stage ran the submitted training code, with nothing uncommitted (D-19)."""

    root = cfg.repo_root
    if differs_in_training_code(root, submitted, "HEAD"):
        raise RuntimeError(f"HEAD's training code differs from the submitted profile stage's ({submitted}) (D-19): rerun the profile stage")
    status = git(root, "status", "--porcelain", "--untracked-files=all", "--", *TRAINING_CODE_PATHS).stdout
    if status.strip():
        raise RuntimeError(f"The training code has uncommitted or untracked changes (D-19):\n{status}")
    checked, offenders = [], []
    for path in sorted((cfg.results_dir / "job_metadata").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        written = record.get("timestamp_utc")
        if not isinstance(written, str) or not written:
            raise ValueError(f"{path} has no timestamp_utc, so nothing shows which stage wrote it (D-19)")
        if written < submitted_at:
            continue  # an earlier stage's job
        reason = ran_submitted_code(root, submitted, record.get("git_commit"), record.get("git_status_porcelain"))
        if reason:
            offenders.append(f"{path.name}: {reason}")
        checked.append(path.name)
    eta = cfg.results_dir / "profiling" / "b3_dual" / "selection.json"
    if not eta.is_file():
        raise FileNotFoundError(f"Missing {eta}: the record follows the eta selection (D-19)")
    reason = ran_submitted_code(root, submitted, json.loads(eta.read_text(encoding="utf-8")).get("git_commit"), "")
    if reason:
        offenders.append(f"{eta.name}: {reason}")
    if offenders:
        raise RuntimeError(
            f"These profile-stage jobs did not run the training code of {submitted} (D-11, D-18, D-19); "
            f"rerun the profile stage: {offenders}"
        )
    return checked


def training_record(cfg, model: str) -> tuple[str, dict]:
    """The newest training metadata whose saved model and normalizer are the current files, byte for byte (B1+;
    the final B2 is found through its promotion, D-22)."""

    model_stem, vecnorm_stem = final_model_stems(model, RECORD_SEED)
    current = (file_sha256(cfg.model_path(f"{model_stem}.zip")), file_sha256(cfg.vecnorm_path(vecnorm_stem)))
    pattern = "train_formula_b1plus_*.json" if model == "b1plus" else "train_formula_b2_alpha*.json"
    matches = []
    for path in sorted((cfg.results_dir / "job_metadata").glob(pattern)):
        record = json.loads(path.read_text(encoding="utf-8"))
        hashes = record.get("artifact_sha256", {})
        if (
            record.get("model") == model
            and record.get("phase") == "final"
            and (hashes.get("payload.model_path"), hashes.get("payload.vecnorm_path")) == current
        ):
            matches.append((str(record.get("timestamp_utc", "")), path.name, record))
    if not matches:
        raise RuntimeError(
            f"No training metadata in results/job_metadata matches the current {model} files, so nothing shows "
            "what they were trained from (D-19): train them through the pipeline"
        )
    _, name, record = max(matches, key=lambda match: match[:2])
    return name, record


INPUT_LABELS = {
    "warm_start_model": "B1",
    "warm_start_vecnorm": "B1's normalizer",
    "train_params": "calibration",
    "manifest": "manifest",
}


def check_training_lineage(cfg, submitted: str, label: str, name: str, record: dict) -> dict:
    """A final B2's or B1+'s training job loaded the current B1, normalizer, calibration and manifest, at the
    submitted code, with k = 10 and seed 42 (D-11, D-18, D-19)."""

    start = record.get("provenance_at_start") or {}
    loaded = start.get("input_sha256") or {}
    stale = [
        INPUT_LABELS[key] for key, path in final_b2_inputs(cfg, RECORD_SEED).items() if loaded.get(key) != file_sha256(path)
    ]
    if stale:
        raise RuntimeError(
            f"{label} ({name}) was not trained from the current {', '.join(stale)} (D-11, D-18, D-19). "
            "Rerun every job that depends on the one you reran, then this record"
        )
    if (record.get("risk_aware_updates"), record.get("training_seed")) != (RECORD_UPDATES, RECORD_SEED):
        raise RuntimeError(
            f"{label} ({name}) ran k = {record.get('risk_aware_updates')} with seed {record.get('training_seed')}; "
            f"D-19 and F-07 need k = {RECORD_UPDATES} and seed {RECORD_SEED}"
        )
    reason = ran_submitted_code(cfg.repo_root, submitted, start.get("git_commit"), start.get("uncommitted_training_code"))
    if reason:
        raise RuntimeError(f"{label} ({name}) did not start from the submitted training code (D-19): {reason}")
    return {
        "metadata": name,
        "slurm_job_id": record.get("slurm_job_id"),
        "start_commit": start.get("git_commit"),
        "alpha": record.get("alpha"),
        "training_fill_paths": record.get("training_fill_paths"),
    }


def check_promoted_b2(cfg, submitted: str, submitted_at: str) -> dict:
    """D-22: the final B2 is the parallel final B2 of the selected alpha, promoted by copy. b2_promotion.json names
    the alpha b2_best_alpha.txt selects and the selection as it is now; the final B2's files are the ones it
    recorded, and the ones that alpha's final B2 job of this stage saved under its own names. That job loaded the
    current B1, normalizer, calibration and manifest, at the submitted code, with k = 10 and seed 42."""

    root = cfg.repo_root
    path = b2_promotion_path(cfg, RECORD_SEED)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}: the record follows b2_promote (D-22)")
    promotion = json.loads(path.read_text(encoding="utf-8"))
    selection = b2_selection_paths(cfg, RECORD_SEED)
    selected = float(selection["b2_best_alpha"].read_text(encoding="utf-8"))
    alpha = promotion.get("alpha")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or float(alpha) != selected:
        raise RuntimeError(
            f"The promoted final B2 has alpha = {alpha!r}, but b2_best_alpha.txt selects {selected} (F-05, D-22): "
            "rerun b2_promote, then this record"
        )
    current_selection = {path.relative_to(root).as_posix(): file_sha256(path) for path in selection.values()}
    if promotion.get("selection_sha256") != current_selection:
        raise RuntimeError("The B2 selection changed after b2_promote (D-22): rerun b2_promote, then this record")
    final = b2_final_paths(cfg, RECORD_SEED)
    current = {key: file_sha256(path) for key, path in final.items()}
    recorded = {key: (promotion.get("files") or {}).get(key, {}).get("sha256") for key in B2_RUN_FILES}
    if recorded != current:
        raise RuntimeError("The final B2 files differ from the run b2_promote promoted (D-22): rerun b2_promote")
    name, record = final_b2_training_record(cfg, float(alpha), RECORD_SEED, current, since=submitted_at)
    lineage = check_training_lineage(cfg, submitted, "Final B2", name, record)
    return {**lineage, "promoted_sha256": {final[key].relative_to(root).as_posix(): digest for key, digest in current.items()}}


def check_lineage(cfg, submitted: str, submitted_at: str) -> dict[str, dict]:
    """Final B2 (the promoted parallel final B2 of the selected alpha, D-22) and B1+ were trained from the current
    inputs, at the submitted code, with k = 10, seed 42 and one N (D-11, D-18, D-19)."""

    lineage = {"b2": check_promoted_b2(cfg, submitted, submitted_at)}
    lineage["b1plus"] = check_training_lineage(cfg, submitted, "B1+", *training_record(cfg, "b1plus"))
    if lineage["b2"]["training_fill_paths"] != lineage["b1plus"]["training_fill_paths"]:
        raise RuntimeError("Final B2 and B1+ ran with different N (D-12, D-18)")
    return lineage


def check_promoted_b3(cfg, submitted: str, submitted_at: str, training_fill_paths) -> dict:
    """D-20: the final B3 is the eta-profiling run D-12a selected, promoted by copy. Its files are the ones
    selection.json recorded and the selected eta task saved. That task started from the submitted training code
    (its end is checked with every other job), loaded the current B1, normalizer, d, calibration and manifest,
    and ran k = 10, seed 42, the selected eta, the current d and final B2's N."""

    root = cfg.repo_root
    directory = cfg.results_dir / "profiling" / "b3_dual"
    selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    eta = selection.get("selected_lambda_lr")
    chosen = directory / "selected_lambda_lr.txt"
    if eta is None or not chosen.is_file() or float(chosen.read_text(encoding="utf-8")) != float(eta):
        raise RuntimeError("The eta selection is incomplete: rerun profile_b3_dual.py --select (D-12a, D-20)")
    histories = selection.get("histories_sha256")
    if not histories:
        raise ValueError("selection.json has no histories_sha256: rerun profile_b3_dual.py --select (D-12a)")
    for line in histories:
        digest, relative = line.split("  ", 1)
        if not (root / relative).is_file() or file_sha256(root / relative) != digest:
            raise RuntimeError(f"{relative} changed after the eta selection: rerun --select, then this record (D-12a, D-20)")
    final = b3_run_paths(cfg, RECORD_SEED)
    current = {key: file_sha256(path) for key, path in final.items()}
    promoted = selection.get("promoted_b3") or {}
    recorded = {key: (promoted.get("files") or {}).get(key, {}).get("sha256") for key in B3_RUN_FILES}
    if promoted.get("eta") != eta or recorded != current:
        raise RuntimeError("The final B3 files differ from the eta run the selection promoted (D-20): rerun --select")
    matches = []
    for path in sorted((cfg.results_dir / "job_metadata").glob(f"{eta_task_job_name(eta)}_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        hashes = record.get("artifact_sha256", {})
        if str(record.get("timestamp_utc", "")) >= submitted_at and all(
            hashes.get(f"payload.{key}") == digest for key, digest in current.items()
        ):
            matches.append((str(record["timestamp_utc"]), path.name, record))
    if not matches:
        raise RuntimeError(
            f"No eta task of this profile stage saved the promoted B3 (eta = {eta}), so nothing shows what it was "
            "trained from (D-20): rerun the eta profiling and the selection"
        )
    _, name, record = max(matches, key=lambda match: match[:2])
    start = record.get("provenance_at_start") or {}
    loaded = start.get("input_sha256") or {}
    stale = [key for key, path in b3_inputs(cfg, RECORD_SEED).items() if loaded.get(key) != file_sha256(path)]
    if stale:
        raise RuntimeError(
            f"B3 ({name}) was not trained from the current {', '.join(stale)} (D-19, D-20). "
            "Rerun every job that depends on the one you reran, then this record"
        )
    if (record.get("risk_aware_updates"), record.get("training_seed")) != (RECORD_UPDATES, RECORD_SEED):
        raise RuntimeError(
            f"B3 ({name}) ran k = {record.get('risk_aware_updates')} with seed {record.get('training_seed')}; "
            f"D-19 and F-07 need k = {RECORD_UPDATES} and seed {RECORD_SEED}"
        )
    d = float(cfg.result_path(seeded_result_name("cvar_threshold_d.txt", RECORD_SEED)).read_text(encoding="utf-8").splitlines()[0])
    if (record.get("eta"), record.get("lambda_learning_rate"), record.get("threshold_d"), selection.get("threshold_d")) != (eta, eta, d, d):
        raise RuntimeError(
            f"B3 ({name}) ran eta = {record.get('eta')} at d = {record.get('threshold_d')}; the selection is "
            f"eta = {eta} at the current d = {d} (D-12a, D-17, D-20)"
        )
    if record.get("training_fill_paths") != training_fill_paths:
        raise RuntimeError("B3 ran with a different N than final B2 and B1+ (D-12, D-18, D-20)")
    reason = ran_submitted_code(root, submitted, start.get("git_commit"), start.get("uncommitted_training_code"))
    if reason:
        raise RuntimeError(f"B3 ({name}) did not start from the submitted training code (D-19, D-20): {reason}")
    return {
        "metadata": name,
        "slurm_job_id": record.get("slurm_job_id"),
        "start_commit": start.get("git_commit"),
        "eta": eta,
        "threshold_d": d,
        "risk_aware_updates": record.get("risk_aware_updates"),
        "training_seed": record.get("training_seed"),
        "training_fill_paths": record.get("training_fill_paths"),
        "promoted_sha256": {final[key].relative_to(root).as_posix(): digest for key, digest in current.items()},
    }


def require_recorded(cfg, paths: list[Path]) -> None:
    """D-20: files a final-stage job uses are exactly the ones the profile-stage record hashed (sha256sum lines)."""

    record = cfg.results_dir / "profiling" / RECORD_NAME
    if not record.is_file():
        raise FileNotFoundError(f"Missing {record}: the final stage evaluates only what the profile stage recorded (D-19, D-20)")
    listed = set(record.read_text(encoding="utf-8").splitlines())
    differing = [line.split("  ", 1)[1] for line in sha256_lines(cfg.repo_root, paths).splitlines() if line not in listed]
    if differing:
        raise RuntimeError(f"{differing} differ from the profile-stage record {record} (D-19, D-20)")


def write_profile_stage_record(cfg) -> tuple[Path, Path, dict]:
    """Archive, check, write. Raises, leaving no record at the top level, if a file is missing, the stage did
    not run the submitted training code throughout, the final B2 is not the promoted run of the selected alpha
    (D-22), or final B2 and B1+ were not trained from the current inputs."""

    directory = cfg.results_dir / "profiling"
    directory.mkdir(parents=True, exist_ok=True)
    archive_previous_record(directory)
    outputs = profile_stage_outputs(cfg)
    missing = [str(path) for path in outputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"The profile stage is incomplete; no record written (D-19). Missing: {missing}")
    submitted, submitted_at = read_submission(directory)
    checked = check_one_training_code(cfg, submitted, submitted_at)
    lineage = check_lineage(cfg, submitted, submitted_at)
    lineage["b3"] = check_promoted_b3(cfg, submitted, submitted_at, lineage["b2"]["training_fill_paths"])
    record = directory / RECORD_NAME
    record.write_text(sha256_lines(cfg.repo_root, outputs), encoding="utf-8", newline="\n")
    commit_path = directory / COMMIT_NAME
    commit_path.write_text(f"{submitted}\n", encoding="utf-8", newline="\n")
    return record, commit_path, {"submitted": submitted, "submitted_at": submitted_at, "jobs_checked": checked, **lineage}


def main() -> None:
    parser = argparse.ArgumentParser(description="Record the profile stage's outputs for the final stage (D-19).")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("TRAINING_SEED", RECORD_SEED)))
    args = parser.parse_args()
    if args.seed != RECORD_SEED:
        raise ValueError(f"The profile-stage record covers training seed {RECORD_SEED} only, the paths the final stage checks (D-19)")

    ctx = load_context(require_prepared=True)
    print_context_banner(ctx, "record_profile_stage")
    record, commit_path, provenance = write_profile_stage_record(ctx.cfg)
    write_metadata(
        "record_profile_stage",
        {
            "record": record,
            "commit_file": commit_path,
            "recorded": profile_stage_outputs(ctx.cfg),
            "provenance": provenance,
            "training_code_paths": TRAINING_CODE_PATHS,
            "training_seed": args.seed,
        },
    )
    print(f"Wrote {record} and {commit_path}:")
    print(record.read_text(encoding="utf-8"), end="")
    print(f"Provenance: {json.dumps(provenance, indent=1)}")


if __name__ == "__main__":
    main()
