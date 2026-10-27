"""Run submit_final_pipeline.sh under bash with a stub sbatch and inspect the job graph (K3, K7)."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cvar_mm.gym.experiment_config import ReplayExperimentConfig
from scripts.pipeline import common as snellius_common
from scripts.pipeline import record_profile_stage

REPO = Path(__file__).resolve().parents[1]
SUBMIT = REPO / "scripts" / "pipeline" / "submit_final_pipeline.sh"
FROZEN_ENV = REPO / "configs" / "final_pipeline.env"
SBATCH_STUB = """#!/usr/bin/env bash
count=$(( $(cat "$SBATCH_STATE" 2>/dev/null || echo 1000) + 1 ))
echo "$count" > "$SBATCH_STATE"
record=absent
if [[ -e "$PROJECT_DIR/results/profiling/profile_stage.sha256" || -e "$PROJECT_DIR/results/profiling/profile_stage.commit" ]]; then
  record=present
fi
{ echo "JOB $count"; echo "STRIDE=$CALIBRATION_START_STRIDE_SECONDS"; echo "RECORD=$record"; printf '%s\\n' "$@"; echo "END"; } >> "$SBATCH_LOG"
if [[ "$count" == "${SBATCH_FAIL_AT:-}" ]]; then  # M2: a failing sbatch, or one that answers without a job ID
  case "$SBATCH_FAIL_MODE" in
    error) echo "sbatch: error: Batch job submission failed (stub)" >&2; exit 1 ;;
    no-id) echo "sbatch: submitted, but no job ID" ;;
  esac
  exit 0
fi
echo "$count"
"""
POST_PROFILE = {"B3_LAMBDA_LR": "0.5"}  # k = 10 is frozen in the file itself (D-19)
# The runtime block of configs/final_pipeline.env (D-19; the author's values of 2026-10-08), and a distinct
# sentinel per limit, so that a test sees which variable each job's --time came from.
TIME_LIMITS = {
    "B1_TIME": "24:00:00", "B1_CAND_TIME": "24:00:00", "CVAR_CAL_TIME": "48:00:00",
    "DUAL_PROFILE_TIME": "72:00:00", "RISK_TRAIN_TIME": "72:00:00",
    "ABLATION_TIME": "24:00:00", "B0_TIME": "24:00:00", "SHAP_TIME": "08:00:00",
    "EVAL_TIME": "08:00:00",  # every evaluate_formula_model.py job (the author's value of 2026-10-13)
}
SENTINEL_TIMES = {
    "B1_TIME": "11:11:11", "B1_CAND_TIME": "22:22:22", "CVAR_CAL_TIME": "33:33:33",
    "DUAL_PROFILE_TIME": "44:44:44", "RISK_TRAIN_TIME": "77:00:00",
    "ABLATION_TIME": "55:55:55", "B0_TIME": "66:06:06", "SHAP_TIME": "08:08:08",
    "EVAL_TIME": "09:09:09",
}
STAGE_ONE_FILES = (
    "models/ppo_b1_doge.zip",
    "models/vecnorm_b1.pkl",
    "results/cvar_threshold_d.txt",
    "results/cvar_threshold_b3.json",
    "results/formula_train_params.json",
    "results/formula_manifest.csv",
)
# The rest of the D-19 record (record_profile_stage.py), and stand-ins for the training code it pins. The final
# B2's two files are the promoted copy of the selected alpha's parallel final B2 (D-22).
FINAL_B2_FILES = ("models/ppo_b2_doge.zip", "models/vecnorm_b2.pkl")
PROFILE_STAGE_FILES = (
    *FINAL_B2_FILES,
    "models/ppo_b1plus_doge.zip",
    "models/vecnorm_b1plus.pkl",
    "results/b2_best_alpha.txt",
    "results/b2_alpha_sweep_summary.csv",
)
# D-20: the final B3, promoted by --select from the selected eta run, and recorded with the rest.
B3_FILES = (
    "models/ppo_b3_doge.zip",
    "models/vecnorm_b3.pkl",
    "results/b3_cvar_lagrangian_history.csv",
    "results/b3_training_costs.csv",
    "results/b3_training_feasibility.json",
)
ETA_GRID = (0.5, 2.0, 8.0)  # profile_b3_dual.ETA_GRID (D-12a)
B2_GRID = (0.1, 1.0, 10.0)  # B2_ALPHAS (F-05): one parallel final B2 each (D-22)
SELECTED_ALPHA = 1.0  # the stand-in B2 selection
D_VALUE = 0.05  # the stand-in d of cvar_threshold_d.txt
# Stand-ins for the training code: one file per pathspec of common.TRAINING_CODE_PATHS, and the excluded one.
TRAINING_CODE_FILES = (
    "cvar_mm/gym/shared.py",
    "cvar_mm/gym/statistical_inference.py",
    "scripts/pipeline/common.py",
    "scripts/pipeline/train_formula_b2.py",
    "scripts/pipeline/train_formula_b3.py",
)
RECORD = "results/profiling/profile_stage.sha256"
RECORD_COMMIT = "results/profiling/profile_stage.commit"
RECORD_SUBMITTED = "results/profiling/profile_stage.submitted"
SUBMITTED_AT, TRAINED_AT = "20261008T000000Z", "20261008T120000Z"


def git(project: Path, *args: str) -> str:
    identity = ("-c", "user.name=test", "-c", "user.email=test@example.com")
    return subprocess.run(
        ["git", "-C", str(project), *identity, *args], capture_output=True, text=True, check=True
    ).stdout


def commit_all(project: Path, message: str) -> None:
    git(project, "add", "-A")
    git(project, "commit", "-q", "-m", message)


def frozen_values(path: Path = FROZEN_ENV) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("export "):
            name, _, value = line[len("export "):].partition("=")
            values[name] = (shlex.split(value) or [""])[0]  # an empty value, e.g. "export B3_LAMBDA_LR="
    return values


def find_bash() -> str | None:
    """Git Bash on Windows (its wrapper puts coreutils on PATH), plain bash elsewhere."""

    if sys.platform != "win32":
        return shutil.which("bash")
    git = shutil.which("git")
    if git is None:
        return None
    for parent in Path(git).resolve().parents:
        if (parent / "bin" / "bash.exe").exists() and (parent / "usr" / "bin").exists():
            return str(parent / "bin" / "bash.exe")
    return None


def parse_jobs(log: Path) -> dict[str, dict]:
    jobs: dict[str, dict] = {}
    if not log.exists():
        return jobs
    for block in log.read_text(encoding="utf-8").split("END\n"):
        lines = block.strip().splitlines()
        if not lines:
            continue
        args = lines[3:]
        jobs[args[args.index("--job-name") + 1]] = {
            "id": lines[0].split()[1],
            "stride": lines[1].split("=", 1)[1],
            "record": lines[2].split("=", 1)[1],  # whether a profile-stage record existed at this sbatch call
            "dependency": args[args.index("--dependency") + 1] if "--dependency" in args else "",
            "array": args[args.index("--array") + 1] if "--array" in args else "",
            "time": args[args.index("--time") + 1],
            "export": next(arg for arg in args if arg.startswith("--export=")),
        }
    return jobs


def set_exports(project: Path, **values: str) -> None:
    """Replace export lines of the project's frozen file and commit it (the profile stage needs it committed)."""

    env = project / "configs" / "final_pipeline.env"
    text = env.read_text(encoding="utf-8")
    for name, value in values.items():
        text, count = re.subn(rf"^export {name}=.*$", f"export {name}={value}", text, flags=re.MULTILINE)
        assert count == 1, name
    env.write_text(text, encoding="utf-8", newline="\n")
    commit_all(project, f"set {', '.join(sorted(values))}")


def frozen_text_without_eta() -> str:
    """The frozen file without its `export B3_LAMBDA_LR=` line, the author's sign-off on eta (D-20, D-24), so that
    each test sets eta itself: make_project appends one unless post_profile=False (no eta export, as before the
    sign-off)."""

    return "".join(
        line for line in FROZEN_ENV.read_text(encoding="utf-8").splitlines(keepends=True)
        if not line.startswith("export B3_LAMBDA_LR=")
    )


def make_project(tmp_path: Path, *, post_profile: bool = True, lambda_lr: str = "0.5", git_repo: bool = True) -> Path:
    """A project with the frozen file, committed in a git repository (the profile stage records its commit)."""

    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    text = frozen_text_without_eta()
    if post_profile:
        values = {**POST_PROFILE, "B3_LAMBDA_LR": lambda_lr}
        text += "".join(f"export {name}={value}\n" for name, value in values.items())
    (project / "configs" / "final_pipeline.env").write_text(text, encoding="utf-8", newline="\n")
    if git_repo:
        git(project, "init", "-q")
        commit_all(project, "the frozen configuration")
    return project


def history_text(eta: float, updates: int = 10, lam: float = 0.0) -> str:
    """An eta history as the profiler writes it (one row per B3 update), cut down to a few of its columns."""

    return "update,eta,lambda\n" + "".join(f"{update},{eta},{lam}\n" for update in range(1, updates + 1))


def write_inputs_sha256(project: Path) -> None:
    lines = [
        f"{hashlib.sha256((project / relative).read_bytes()).hexdigest()}  {relative}\n" for relative in STAGE_ONE_FILES
    ]
    (project / "results" / "profiling" / "b3_dual" / "inputs.sha256").write_text(
        "".join(lines), encoding="utf-8", newline="\n"
    )


def write_training_metadata(project: Path, model: str, commit: str, alpha: float = SELECTED_ALPHA, **overrides) -> Path:
    """The metadata train_formula_b2.py leaves for a parallel final B2 of ``alpha`` (D-22: its alpha-specific files,
    no B2 selection read) or for B1+ (write_metadata: the payload, the commit and status when the job ends, and
    artifact_sha256), for the stand-in files, with its provenance_at_start."""

    cfg = ReplayExperimentConfig(repo_root=project)
    if model == "b2":
        files = snellius_common.b2_final_run_paths(cfg, alpha, 42)
    else:
        model_stem, vecnorm_stem = snellius_common.final_model_stems(model, 42)
        files = {"model_path": cfg.model_path(f"{model_stem}.zip"), "vecnorm_path": cfg.vecnorm_path(vecnorm_stem)}
    loaded = snellius_common.final_b2_inputs(cfg, 42)
    payload = {
        "model": model, "phase": "final", "alpha": alpha if model == "b2" else 0.0, "training_seed": 42,
        "risk_aware_updates": 10, "training_fill_paths": 16, **files,
        "warm_start_model": loaded["warm_start_model"], "warm_start_vecnorm": loaded["warm_start_vecnorm"],
        "provenance_at_start": {
            "git_commit": commit,
            "uncommitted_training_code": "",
            "input_sha256": {name: snellius_common.file_sha256(path) for name, path in loaded.items()},
        },
        **overrides,
    }
    record = {
        "job_name": f"train_formula_{'b1plus' if model == 'b1plus' else f'b2_alpha{alpha}'}",
        "slurm_job_id": "local", "timestamp_utc": TRAINED_AT, "git_commit": commit, "git_status_porcelain": "",
        **payload,
        "artifact_sha256": snellius_common._existing_file_hashes(payload),
    }
    path = cfg.results_dir / "job_metadata" / f"{record['job_name']}_local_{TRAINED_AT}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snellius_common._json_safe(record)), encoding="utf-8")
    return path


def write_eta_task_metadata(project: Path, eta: float, commit: str, **overrides) -> Path:
    """The metadata one eta task of profile_b3_dual.py leaves (D-20): its B3 run's payload, with the files it
    saved, the load-time hashes and start commit (provenance_at_start), and write_metadata's end state."""

    cfg = ReplayExperimentConfig(repo_root=project)
    payload = {
        "model": "b3", "eta": eta, "lambda_learning_rate": eta, "risk_aware_updates": 10, "training_seed": 42,
        "training_fill_paths": 16, "threshold_d": D_VALUE, **snellius_common.eta_run_paths(cfg, eta),
        "provenance_at_start": {
            "git_commit": commit,
            "uncommitted_training_code": "",
            "input_sha256": {
                name: snellius_common.file_sha256(path) for name, path in snellius_common.b3_inputs(cfg, 42).items()
            },
        },
        **overrides,
    }
    record = {
        "job_name": snellius_common.eta_task_job_name(eta), "slurm_job_id": "local", "timestamp_utc": TRAINED_AT,
        "git_commit": commit, "git_status_porcelain": "", **payload,
        "artifact_sha256": snellius_common._existing_file_hashes(payload),
    }
    path = cfg.results_dir / "job_metadata" / f"{record['job_name']}_local_{TRAINED_AT}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snellius_common._json_safe(record)), encoding="utf-8")
    return path


def promote(project: Path, eta: float) -> dict:
    """What profile_b3_dual.py --select does for the selected eta (D-20): copy its run to the final B3's names."""

    cfg = ReplayExperimentConfig(repo_root=project)
    sources, targets = snellius_common.eta_run_paths(cfg, eta), snellius_common.b3_run_paths(cfg, 42)
    files = {}
    for key in snellius_common.B3_RUN_FILES:
        shutil.copyfile(sources[key], targets[key])
        files[key] = {
            "from": sources[key].relative_to(project).as_posix(), "to": targets[key].relative_to(project).as_posix(),
            "sha256": snellius_common.file_sha256(targets[key]),
        }
    return {"eta": eta, "files": files}


def write_stage_one(project: Path, selected: str = "0.5") -> None:
    """A finished profile stage: its artifacts, the hashes profile_b3_dual.py records for the eta inputs, each
    eta task's B3 run and metadata, the eta selection with the B3 it promoted (D-20), the submission (commit
    and time), the three parallel final B2s (D-22) and B1+ with their training metadata, the promotion of the
    selected alpha's final B2 (common.promote_b2, the b2_promote job's own code), and the D-19 record that
    record_profile_stage.py writes (its own code), in the project's git repository."""

    cfg = ReplayExperimentConfig(repo_root=project)
    profile = project / "results" / "profiling" / "b3_dual"
    profile.mkdir(parents=True)
    finals = [path for alpha in B2_GRID for path in snellius_common.b2_final_run_paths(cfg, alpha, 42).values()]
    stand_ins = [project / relative for relative in STAGE_ONE_FILES + PROFILE_STAGE_FILES + TRAINING_CODE_FILES
                 if relative not in FINAL_B2_FILES]  # the final B2 is the promoted copy below (D-22)
    for path in stand_ins + finals:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"profile stage {path.relative_to(project).as_posix()}".encode())
    (project / "results" / "b2_best_alpha.txt").write_text(str(SELECTED_ALPHA), encoding="utf-8")
    (project / "results" / "cvar_threshold_d.txt").write_text(f"{D_VALUE}\n{D_VALUE / 0.8}\n0.0\n", encoding="utf-8")
    histories = [profile / f"history_eta{eta}.csv" for eta in ETA_GRID]
    for eta, history in zip(ETA_GRID, histories):
        history.write_text(history_text(eta), encoding="utf-8", newline="\n")
        for key, path in snellius_common.eta_run_paths(cfg, eta).items():
            path.write_bytes(f"eta {eta} B3 run {key}".encode())
    promoted = promote(project, float(selected))
    write_inputs_sha256(project)
    (profile / "selected_lambda_lr.txt").write_text(f"{selected}\n", encoding="utf-8", newline="\n")
    commit_all(project, "the profile stage's commit")
    commit = git(project, "rev-parse", "HEAD").strip()
    (project / RECORD_SUBMITTED).write_text(f"{commit} {SUBMITTED_AT}\n", encoding="utf-8", newline="\n")
    (profile / "selection.json").write_text(json.dumps({
        "rule": "D-12a", "eta_grid": list(ETA_GRID), "updates_per_eta": 10,
        "git_commit": commit, "selected_lambda_lr": float(selected), "threshold_d": D_VALUE,
        "training_fill_paths": 16, "promoted_b3": promoted,
        "histories_sha256": snellius_common.sha256_lines(project, histories).splitlines(),
    }), encoding="utf-8")
    for eta in ETA_GRID:
        write_eta_task_metadata(project, eta, commit)
    for alpha in B2_GRID:
        write_training_metadata(project, "b2", commit, alpha=alpha)
    write_training_metadata(project, "b1plus", commit)
    snellius_common.promote_b2(cfg, 42)  # what b2_promote does (D-22)
    record_profile_stage.write_profile_stage_record(cfg)


def submit(tmp_path: Path, project: Path, **env: str):
    bash = find_bash()
    if bash is None:
        pytest.skip("needs a POSIX bash (Git Bash on Windows)")
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir(exist_ok=True)
    stub = stub_dir / "sbatch"
    stub.write_text(SBATCH_STUB, encoding="utf-8", newline="\n")
    stub.chmod(0o755)
    log = tmp_path / "sbatch.log"
    # The venv's Python, which the final stage reads selection.json with (item 7b): this test's interpreter.
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text(f'#!/usr/bin/env bash\nexec "{Path(sys.executable).as_posix()}" "$@"\n', encoding="utf-8", newline="\n")
    python.chmod(0o755)
    frozen = frozen_values(project / "configs" / "final_pipeline.env")
    # k, B1_PASSES, POLICY_STD, FINE_TUNE_LR, B3_LAMBDA_LR (sourcing the signed-off file exports it, D-24) and the time
    # limits too, even where a test removed their export; a leftover TOTAL_TIMESTEPS exits 2 (F-16), and so would a stub
    # failure mode left in the caller's environment; NB6 and SHAP take the script's defaults unless a test sets them
    unset = {"PIPELINE_STAGE", "B2_UPDATES", "B3_UPDATES", "B1_PASSES", "POLICY_STD", "FINE_TUNE_LR", "B3_LAMBDA_LR",
             "TOTAL_TIMESTEPS", *TIME_LIMITS, "SBATCH_FAIL_AT", "SBATCH_FAIL_MODE", "RUN_NB6", "RUN_SHAP"}
    run_env = {key: value for key, value in os.environ.items() if key not in frozen and key not in unset}
    run_env.update(
        PATH=f"{stub_dir}{os.pathsep}{os.environ['PATH']}",
        PROJECT_DIR=project.as_posix(),
        DATA_DIR=(tmp_path / "data").as_posix(),
        VENV_PATH=(tmp_path / "venv").as_posix(),
        SBATCH_LOG=log.as_posix(),
        SBATCH_STATE=(tmp_path / "sbatch.state").as_posix(),
    )
    run_env.update(env)
    # The stub must be the sbatch the script finds: on a noexec --basetemp bash would skip it and reach a real
    # sbatch (on a Snellius node, a real submission). Refuse to run the script then.
    probe = subprocess.run(
        [bash, "-c", 'grep -c SBATCH_STATE "$(command -v sbatch)"'], env=run_env, capture_output=True, text=True
    )
    if probe.returncode != 0 or probe.stdout.strip() in ("", "0"):
        pytest.fail(f"The stub sbatch in {stub_dir} is not the first sbatch on PATH; not running the submit script")
    result = subprocess.run(
        [bash, SUBMIT.as_posix()], env=run_env, capture_output=True, text=True, timeout=180
    )
    return result, parse_jobs(log)


@pytest.mark.parametrize(
    ("name", "value"), [("EVALUATION_FILL_SEEDS", "42"), ("CVAR_ROLLOUTS_PER_SEED", "3")]  # the latter sets d
)
def test_a_frozen_value_mismatch_stops_before_anything_is_submitted(tmp_path, name, value):
    project = make_project(tmp_path)
    result, jobs = submit(tmp_path, project, PIPELINE_STAGE="profile", **{name: value})
    assert result.returncode == 1
    assert name in result.stderr
    assert jobs == {} and not (project / "logs").exists()


def test_the_pipeline_stage_must_be_named(tmp_path):
    result, jobs = submit(tmp_path, make_project(tmp_path))
    assert result.returncode == 2 and jobs == {}
    assert "Set PIPELINE_STAGE=profile" in result.stderr  # this check, not another exit-2 check


def test_k_is_frozen_at_10_once_each():
    """D-19: k = 10 for B2 (all phases), B3 and B1+ is a frozen export; no pending k line remains."""

    lines = FROZEN_ENV.read_text(encoding="utf-8").splitlines()
    for name in ("B2_UPDATES", "B3_UPDATES"):
        assert [line for line in lines if name in line] == [f"export {name}=10"]


@pytest.mark.parametrize(
    ("stage", "caller", "edit"),
    [
        ("profile", {"B2_UPDATES": "9"}, None),
        ("final", {"B3_UPDATES": "11"}, None),
        ("profile", {"B2_UPDATES": "10.0"}, None),  # exact text: the jobs parse k as an integer
        ("final", {}, ("export B2_UPDATES=10\n", "export B2_UPDATES=9\n")),
        # A later export overrides the frozen line: only the sourced-value check sees it.
        ("final", {}, ("export B3_LAMBDA_LR=0.5\n", "export B3_LAMBDA_LR=0.5\nexport B3_UPDATES=3\n")),
        # No k export in the file: only the file-line check sees it, since the caller's 10 survives `source`.
        ("profile", {"B2_UPDATES": "10", "B3_UPDATES": "10"}, ("export B2_UPDATES=10\nexport B3_UPDATES=10\n", "")),
        ("final", {"B2_UPDATES": "10", "B3_UPDATES": "10"}, ("export B2_UPDATES=10\nexport B3_UPDATES=10\n", "")),
    ],
)
def test_any_k_other_than_10_exits_2_before_anything_is_submitted(tmp_path, stage, caller, edit):
    """D-19: k = 10, fixed a priori. Exit 2 (not K7's exit 1), whether the environment or the file differs."""

    project = make_project(tmp_path)
    if edit is not None:
        env = project / "configs" / "final_pipeline.env"
        text = env.read_text(encoding="utf-8")
        assert edit[0] in text
        env.write_text(text.replace(edit[0], edit[1]), encoding="utf-8", newline="\n")
    result, jobs = submit(tmp_path, project, PIPELINE_STAGE=stage, **caller)
    assert result.returncode == 2, result.stderr
    assert "D-19" in result.stderr and "must be exactly 10" in result.stderr
    assert jobs == {} and not (project / "logs").exists()


def test_b1_passes_is_frozen_at_1_once():
    """F-16: B1 and B1_cand each train exactly one pass; the frozen export is the only line naming it."""

    lines = FROZEN_ENV.read_text(encoding="utf-8").splitlines()
    assert [line for line in lines if "B1_PASSES" in line] == ["export B1_PASSES=1"]
    assert not [line for line in lines if "TOTAL_TIMESTEPS" in line and line.startswith("export")]


@pytest.mark.parametrize(
    ("stage", "caller", "edit"),
    [
        ("profile", {"B1_PASSES": "2"}, None),
        ("final", {"B1_PASSES": "1.0"}, None),  # exact text, as for k
        ("profile", {}, ("export B1_PASSES=1\n", "export B1_PASSES=2\n")),
        # A later export overrides the frozen line: only the sourced-value check sees it.
        ("final", {}, ("export B3_LAMBDA_LR=0.5\n", "export B3_LAMBDA_LR=0.5\nexport B1_PASSES=3\n")),
        # No export in the file: only the file-line check sees it, since the caller's 1 survives `source`.
        ("profile", {"B1_PASSES": "1"}, ("export B1_PASSES=1\n", "")),
    ],
)
def test_any_b1_passes_other_than_1_exits_2_before_anything_is_submitted(tmp_path, stage, caller, edit):
    """F-16 (I3): one pass, checked like k (exit 2, not K7's exit 1)."""

    project = make_project(tmp_path)
    if edit is not None:
        env = project / "configs" / "final_pipeline.env"
        text = env.read_text(encoding="utf-8")
        assert edit[0] in text
        env.write_text(text.replace(edit[0], edit[1]), encoding="utf-8", newline="\n")
    result, jobs = submit(tmp_path, project, PIPELINE_STAGE=stage, **caller)
    assert result.returncode == 2, result.stderr
    assert "B1_PASSES must be exactly 1" in result.stderr and "F-16" in result.stderr
    assert jobs == {} and not (project / "logs").exists()


def test_policy_std_is_frozen_at_0_3_once():
    """D-21: sigma0 = 0.3 is a frozen export; no other line of the file names it."""

    lines = FROZEN_ENV.read_text(encoding="utf-8").splitlines()
    assert [line for line in lines if "POLICY_STD" in line] == ["export POLICY_STD=0.3"]


@pytest.mark.parametrize(
    ("stage", "caller", "edit"),
    [
        ("profile", {"POLICY_STD": "0.5"}, None),  # overridden by the caller
        ("final", {"POLICY_STD": "0.30"}, None),  # exact text, as for k
        ("profile", {}, ("export POLICY_STD=0.3\n", "export POLICY_STD=0.5\n")),
        # A later export overrides the frozen line: only the sourced-value check sees it.
        ("final", {}, ("export B3_LAMBDA_LR=0.5\n", "export B3_LAMBDA_LR=0.5\nexport POLICY_STD=0.5\n")),
        # No export in the file: only the file-line check sees it, since the caller's 0.3 survives `source`.
        ("profile", {"POLICY_STD": "0.3"}, ("export POLICY_STD=0.3\n", "")),
        ("final", {}, ("export POLICY_STD=0.3\n", "")),  # missing everywhere
        # grep -F: the dot is literal, so 0x3 is not 0.3 even where a later line restores the sourced value.
        ("profile", {}, ("export POLICY_STD=0.3\n", "export POLICY_STD=0x3\nPOLICY_STD=0.3\n")),
    ],
)
def test_any_policy_std_other_than_0_3_exits_2_before_anything_is_submitted(tmp_path, stage, caller, edit):
    """D-21 test (d): POLICY_STD missing, 0.5 or overridden exits 2 (not K7's exit 1), naming D-21, with zero
    sbatch calls."""

    project = make_project(tmp_path)
    if edit is not None:
        env = project / "configs" / "final_pipeline.env"
        text = env.read_text(encoding="utf-8")
        assert edit[0] in text
        env.write_text(text.replace(edit[0], edit[1]), encoding="utf-8", newline="\n")
    result, jobs = submit(tmp_path, project, PIPELINE_STAGE=stage, **caller)
    assert result.returncode == 2, result.stderr
    assert "POLICY_STD must be exactly 0.3" in result.stderr and "D-21" in result.stderr
    assert jobs == {} and not (project / "logs").exists()


def test_fine_tune_lr_is_frozen_at_0_003_once():
    """D-23: the fine-tunes' learning rate 3e-3 is a frozen export; no other line of the file names it."""

    lines = FROZEN_ENV.read_text(encoding="utf-8").splitlines()
    assert [line for line in lines if "FINE_TUNE_LR" in line] == ["export FINE_TUNE_LR=0.003"]


@pytest.mark.parametrize(
    ("stage", "caller", "edit"),
    [
        ("profile", {"FINE_TUNE_LR": "0.001"}, None),  # overridden by the caller
        ("final", {"FINE_TUNE_LR": "0.0003"}, None),  # B1's rate
        ("profile", {"FINE_TUNE_LR": "3e-3"}, None),  # exact text, as for k: the same float is not the frozen value
        ("final", {"FINE_TUNE_LR": "0.0030"}, None),
        ("profile", {}, ("export FINE_TUNE_LR=0.003\n", "export FINE_TUNE_LR=0.001\n")),
        # A later export overrides the frozen line: only the sourced-value check sees it.
        ("final", {}, ("export B3_LAMBDA_LR=0.5\n", "export B3_LAMBDA_LR=0.5\nexport FINE_TUNE_LR=0.001\n")),
        # No export in the file: only the file-line check sees it, since the caller's 0.003 survives `source`.
        ("profile", {"FINE_TUNE_LR": "0.003"}, ("export FINE_TUNE_LR=0.003\n", "")),
        ("final", {}, ("export FINE_TUNE_LR=0.003\n", "")),  # missing everywhere
        # grep -F: the dot is literal, so 0x003 is not 0.003 even where a later line restores the sourced value.
        ("profile", {}, ("export FINE_TUNE_LR=0.003\n", "export FINE_TUNE_LR=0x003\nFINE_TUNE_LR=0.003\n")),
    ],
)
def test_any_fine_tune_lr_other_than_0_003_exits_2_before_anything_is_submitted(tmp_path, stage, caller, edit):
    """D-23 test (c): FINE_TUNE_LR missing, 0.001 or overridden exits 2 (not K7's exit 1), naming D-23, with zero
    sbatch calls."""

    project = make_project(tmp_path)
    if edit is not None:
        env = project / "configs" / "final_pipeline.env"
        text = env.read_text(encoding="utf-8")
        assert edit[0] in text
        env.write_text(text.replace(edit[0], edit[1]), encoding="utf-8", newline="\n")
    result, jobs = submit(tmp_path, project, PIPELINE_STAGE=stage, **caller)
    assert result.returncode == 2, result.stderr
    assert "FINE_TUNE_LR must be exactly 0.003" in result.stderr and "D-23" in result.stderr
    assert jobs == {} and not (project / "logs").exists()


@pytest.mark.parametrize("stage", ["profile", "final"])
def test_a_leftover_total_timesteps_exits_2_before_anything_is_submitted(tmp_path, stage):
    """F-16 (I3): the pipeline no longer uses or exports TOTAL_TIMESTEPS, and B1 refuses it."""

    project = make_project(tmp_path)
    result, jobs = submit(tmp_path, project, PIPELINE_STAGE=stage, TOTAL_TIMESTEPS="1000000")
    assert result.returncode == 2, result.stderr
    assert "TOTAL_TIMESTEPS is set" in result.stderr and "F-16" in result.stderr
    assert jobs == {} and not (project / "logs").exists()


def test_the_jobs_get_no_total_timesteps_export():
    """F-16 (I3): the sbatch --export list names ABLATION_TIMESTEPS (NB6) but not TOTAL_TIMESTEPS."""

    (export,) = re.findall(r"--export=ALL,[^)]*", SUBMIT.read_text(encoding="utf-8"))
    assert "TOTAL_TIMESTEPS" not in export.replace("ABLATION_TIMESTEPS", "")
    assert "ABLATION_TIMESTEPS=" in export


ALPHAS = ("0.1", "1.0", "10.0")
FEATURE_SETS = ("base_state", "base_plus_vol", "full")
# name: (script, arguments, the jobs it waits for, array). D-19 puts every training- and validation-side job
# here (F-19); each script's days are pinned in test_risk_aware_training.py, and prep only hashes Jan 7-29.
PROFILE_JOBS = {
    "prep_formula": ("prepare_formula_pipeline.py", "--force", (), ""),
    "b1_train": ("train_formula_b1.py", "", ("prep_formula",), ""),
    "b3_cvarcal": ("calibrate_cvar_b3.py", "", ("b1_train",), ""),
    "b3_dual": ("profile_b3_dual.py", "", ("b3_cvarcal",), "0-2"),
    "b3_dual_select": ("profile_b3_dual.py", "--select", ("b3_dual",), ""),
    "b1_cand": ("train_formula_b1.py", "--phase candidate", ("prep_formula",), ""),
    **{f"b2_a{a}": ("train_formula_b2.py", f"--phase candidate --alpha {a}", ("b1_cand",), "") for a in ALPHAS},
    **{f"b2e_a{a}": ("evaluate_formula_model.py", f"--model b2 --split validation --alpha {a}", (f"b2_a{a}",), "0-1%2")
       for a in ALPHAS},
    **{f"b2g_a{a}": ("aggregate_final_results.py", f"--model b2 --split validation --alpha {a}", (f"b2e_a{a}",), "")
       for a in ALPHAS},
    "b2_select": ("aggregate_final_results.py", "--select-b2", tuple(f"b2g_a{a}" for a in ALPHAS), ""),
    # D-22: one final B2 per alpha, after B1 only, and the promotion of the selected one, after the selection and all three.
    **{f"b2_final_a{a}": ("train_formula_b2.py", f"--phase final --alpha {a}", ("b1_train",), "") for a in ALPHAS},
    "b2_promote": ("train_formula_b2.py", "--promote", ("b2_select", *(f"b2_final_a{a}" for a in ALPHAS)), ""),
    "b1plus_train": ("train_formula_b2.py", "--phase final --control b1plus", ("b1_train",), ""),
    "profile_record": ("record_profile_stage.py", "", ("b2_promote", "b1plus_train", "b3_dual_select"), ""),
    **{f"abl_{fs}": ("run_feature_ablation.py", f"--feature-set {fs}", ("prep_formula",), "") for fs in FEATURE_SETS},
    "abl_agg": ("aggregate_final_results.py", "--ablation", tuple(f"abl_{fs}" for fs in FEATURE_SETS), ""),
    "b1_shap": ("run_shap_diagnostics.py", "", ("b1_train",), ""),
}
TEST_ARRAY = "0-22%6"
AGGREGATES = ("b0_fair", "b1_agg", "b2_agg", "b3_agg", "b1plus_agg")
FINAL_JOBS = {
    "b0_fair": ("run_b0_fair.py", "", (), ""),
    "b1_eval": ("evaluate_formula_model.py", "--model b1", (), TEST_ARRAY),
    "b1_agg": ("aggregate_final_results.py", "--model b1", ("b1_eval",), ""),
    "b2_eval": ("evaluate_formula_model.py", "--model b2", (), TEST_ARRAY),
    "b2_agg": ("aggregate_final_results.py", "--model b2", ("b2_eval",), ""),
    "b1plus_eval": ("evaluate_formula_model.py", "--model b1plus", (), TEST_ARRAY),
    "b1plus_agg": ("aggregate_final_results.py", "--model b1plus", ("b1plus_eval",), ""),
    "b3_eval": ("evaluate_formula_model.py", "--model b3", (), TEST_ARRAY),  # B3 is the promoted eta run (D-20)
    "b3_agg": ("aggregate_final_results.py", "--model b3", ("b3_eval",), ""),
    "final_compare": ("aggregate_final_results.py", "--comparison", AGGREGATES, ""),
    "paired_infer": ("run_paired_inference.py", "", AGGREGATES, ""),
}
# Every evaluate_formula_model.py job; each array task is one day x 2 fill seeds x 20 rollouts (EVAL_TIME, D-19).
VALIDATION_EVALUATIONS = tuple(f"b2e_a{a}" for a in ALPHAS)
TEST_DAY_EVALUATIONS = ("b1_eval", "b2_eval", "b1plus_eval", "b3_eval")


def command(job: dict) -> tuple[str, str]:
    """A stubbed job's script and arguments, from its --export argument (no exported value holds a comma)."""

    fields = dict(item.partition("=")[::2] for item in job["export"].split(",")[1:])
    return fields["SCRIPT_PATH"].removeprefix("scripts/pipeline/"), " ".join(fields["SCRIPT_ARGS"].split())


def assert_job_table(jobs: dict[str, dict], expected: dict[str, tuple]) -> None:
    assert sorted(jobs) == sorted(expected)
    for name, (script, args, waits_for, array) in expected.items():
        assert command(jobs[name]) == (script, args), name
        dependency = "afterok:" + ":".join(jobs[other]["id"] for other in waits_for) if waits_for else ""
        assert (jobs[name]["dependency"], jobs[name]["array"]) == (dependency, array), name


def test_profile_stage_submits_every_training_and_validation_job_and_no_other(tmp_path):
    """D-19 (d), D-22 test (c): the training- and validation-side jobs with their dependencies; one final B2 per
    alpha after b1_train only, b2_promote after b2_select and the three, B1+ after b1_train only, the record
    after b2_promote, B1+ and the eta selection. No B3 training and no job that reads Jan 7-29. An earlier
    record is moved aside before the first job, and the submission is recorded."""

    project = make_project(tmp_path, post_profile=False)
    write_stage_one(project)  # an earlier profile stage's record
    set_exports(project, **SENTINEL_TIMES)  # D-19: every limit comes from the runtime block, committed
    earlier = {name: (project / name).read_bytes() for name in (RECORD, RECORD_COMMIT, RECORD_SUBMITTED)}
    head = git(project, "rev-parse", "HEAD").strip()
    result, jobs = submit(
        tmp_path, project, PIPELINE_STAGE="profile",
        RUN_NB6="1", RUN_SHAP="1",
        B2_UPDATES="10", B3_UPDATES="10",  # a caller's k of exactly 10 passes (D-19)
        POLICY_STD="0.3",  # and a caller's sigma0 of exactly 0.3 (D-21)
        FINE_TUNE_LR="0.003",  # and a caller's fine-tune learning rate of exactly 0.003 (D-23)
    )
    assert result.returncode == 0, result.stderr
    assert_job_table(jobs, PROFILE_JOBS)
    for job in jobs.values():
        script, args = command(job)
        assert script not in {"train_formula_b3.py", "run_b0_fair.py", "run_paired_inference.py"}
        if script in {"evaluate_formula_model.py", "aggregate_final_results.py"}:  # their default is Jan 7-29
            assert "--split validation" in args or args in {"--select-b2", "--ablation"}, (script, args)
    for name in (*(f"b2_a{a}" for a in ALPHAS), *(f"b2_final_a{a}" for a in ALPHAS), "b1plus_train"):
        assert jobs[name]["time"] == SENTINEL_TIMES["RISK_TRAIN_TIME"], name
    assert (jobs["b2_promote"]["time"], jobs["profile_record"]["time"]) == ("00:30:00", "00:30:00")
    assert (jobs["b1_train"]["time"], jobs["b1_cand"]["time"]) == (
        SENTINEL_TIMES["B1_TIME"], SENTINEL_TIMES["B1_CAND_TIME"])
    assert (jobs["b3_cvarcal"]["time"], jobs["b3_dual"]["time"]) == (
        SENTINEL_TIMES["CVAR_CAL_TIME"], SENTINEL_TIMES["DUAL_PROFILE_TIME"])
    for name in (f"abl_{feature_set}" for feature_set in FEATURE_SETS):  # each NB6 job trains a pass over Jan 1-4
        assert jobs[name]["time"] == SENTINEL_TIMES["ABLATION_TIME"], name
    assert (jobs["b1_shap"]["time"], jobs["abl_agg"]["time"]) == (SENTINEL_TIMES["SHAP_TIME"], "01:00:00")
    for name in VALIDATION_EVALUATIONS:  # B2 validation, one Jan 5 or Jan 6 day per task
        assert jobs[name]["time"] == SENTINEL_TIMES["EVAL_TIME"], name
    # The sourced frozen values reach every job's environment.
    assert all(job["stride"] == "15" for job in jobs.values())
    assert all("EVALUATION_FILL_SEEDS=42 43," in job["export"] for job in jobs.values())
    # F-16 (I3): no TOTAL_TIMESTEPS in any job's --export; ABLATION_TIMESTEPS stays.
    assert not any(re.search(r"(^|,)TOTAL_TIMESTEPS=", job["export"]) for job in jobs.values())
    assert all(",ABLATION_TIMESTEPS=" in job["export"] for job in jobs.values())
    # The earlier record was gone before the first sbatch, and is kept under superseded/.
    assert {job["record"] for job in jobs.values()} == {"absent"}
    assert not (project / RECORD).exists() and not (project / RECORD_COMMIT).exists()
    (archive,) = (project / "results" / "profiling" / "superseded").iterdir()
    assert {name: (archive / Path(name).name).read_bytes() for name in earlier} == earlier
    commit, submitted_at = (project / RECORD_SUBMITTED).read_text(encoding="utf-8").split()
    assert commit == head and re.fullmatch(r"\d{8}T\d{6}Z", submitted_at)


@pytest.mark.parametrize(("switch", "shap"), [({}, True), ({"RUN_SHAP": "0"}, False)], ids=["default", "switched-off"])
def test_the_profile_stage_submits_shap_unless_switched_off(tmp_path, switch, shap):
    """D-19 (F-19): SHAP reads B1 on Jan 1, so it is a profile-stage job and runs by default, like NB6. It used to
    default to off, and the 2026-10-12 submission left it out (item 7b)."""

    project = make_project(tmp_path, post_profile=False)
    result, jobs = submit(tmp_path, project, PIPELINE_STAGE="profile", **switch)
    assert result.returncode == 0, result.stderr
    assert_job_table(jobs, {name: job for name, job in PROFILE_JOBS.items() if shap or name != "b1_shap"})


def test_the_runtime_block_holds_each_time_limit_once():
    """D-19: the Slurm time limits are explicit values in configs/final_pipeline.env (the author's, 2026-10-08)."""

    lines = FROZEN_ENV.read_text(encoding="utf-8").splitlines()
    for name, value in TIME_LIMITS.items():
        assert [line for line in lines if line.startswith(f"export {name}=")] == [f"export {name}={value}"]
    assert frozen_values().keys() >= TIME_LIMITS.keys()


@pytest.mark.parametrize("stage", ["profile", "final"])
@pytest.mark.parametrize(
    ("name", "value"),
    [
        *[(name, None) for name in TIME_LIMITS],  # no export line, while the caller's valid value survives `source`
        ("RISK_TRAIN_TIME", "72h"),
        ("DUAL_PROFILE_TIME", "72:00"),  # Slurm would read 72 minutes
        ("CVAR_CAL_TIME", "2-00:00:00"),
        ("B1_TIME", "00:00:00"),  # no limit at all
        ("B1_CAND_TIME", "24:60:00"),
        ("RISK_TRAIN_TIME", '""'),
        ("ABLATION_TIME", "4h"),
        ("B0_TIME", "24:00"),
        ("SHAP_TIME", "0:00:00"),
        ("EVAL_TIME", "8h"),
        ("EVAL_TIME", "00:00:00"),
    ],
)
def test_a_missing_or_malformed_time_limit_exits_2_before_anything_is_submitted(tmp_path, stage, name, value):
    """D-19 (runtime): no default and no guess; the job graph is never submitted with a bad limit."""

    project = make_project(tmp_path, post_profile=(stage == "final"))
    env = project / "configs" / "final_pipeline.env"
    text = env.read_text(encoding="utf-8")
    line = f"export {name}={TIME_LIMITS[name]}\n"
    assert line in text
    env.write_text(text.replace(line, "" if value is None else f"export {name}={value}\n"), encoding="utf-8", newline="\n")
    commit_all(project, f"{name} edited")
    caller = {name: TIME_LIMITS[name]} if value is None else {}
    result, jobs = submit(tmp_path, project, PIPELINE_STAGE=stage, **caller)
    assert result.returncode == 2, result.stderr
    assert f"{name} must be exported" in result.stderr and "D-19" in result.stderr
    assert jobs == {} and not (project / "logs").exists()


def test_a_caller_time_limit_other_than_the_file_s_is_a_k7_mismatch(tmp_path):
    """The file is the only source of the limits: a different value in the shell exits 1 (K7), never overrides."""

    project = make_project(tmp_path, post_profile=False)
    result, jobs = submit(tmp_path, project, PIPELINE_STAGE="profile", RISK_TRAIN_TIME="99:00:00")
    assert result.returncode == 1 and "Frozen value mismatch: RISK_TRAIN_TIME" in result.stderr
    assert jobs == {}


@pytest.mark.parametrize("stage", ["profile", "final"])
def test_every_evaluation_array_gets_eval_time(tmp_path, stage):
    """D-19 (runtime): every evaluate_formula_model.py job takes EVAL_TIME (the author's 08:00:00 of 2026-10-13),
    not the fixed 03:00:00 that one test day x 2 fill seeds x 20 rollouts could exceed: the B2 validation arrays
    in the profile stage, the four test-day arrays in the final stage."""

    project = make_project(tmp_path, post_profile=(stage == "final"))
    if stage == "final":
        write_stage_one(project)
    result, jobs = submit(tmp_path, project, PIPELINE_STAGE=stage)
    assert result.returncode == 0, result.stderr
    evaluations = {name: job["time"] for name, job in jobs.items() if command(job)[0] == "evaluate_formula_model.py"}
    expected = VALIDATION_EVALUATIONS if stage == "profile" else TEST_DAY_EVALUATIONS
    assert evaluations == dict.fromkeys(expected, "08:00:00")
    assert "03:00:00" not in {job["time"] for job in jobs.values()}


@pytest.mark.parametrize(
    ("fail_at", "mode", "job"),
    [
        (1001, "error", "prep_formula"),  # the first job: `x=$(submit_job ...)`
        (1003, "no-id", "b3_cvarcal"),
        (1007, "empty", "b2_a0.1"),  # inside the alpha loop
        (1018, "error", "b2_final_a1.0"),  # inside the final-B2 loop (D-22)
        (1020, "no-id", "b2_promote"),
        (1023, "error", "abl_base_state"),  # the NB6 loop
        (1026, "no-id", "abl_agg"),  # `submit_job ... >/dev/null`
        (1027, "error", "b1_shap"),
    ],
)
def test_a_failed_sbatch_or_one_without_a_job_id_stops_the_submission(tmp_path, fail_at, mode, job):
    """M2: exit 2 naming the job, and no further sbatch call, so no dependant gets an empty afterok:."""

    project = make_project(tmp_path, post_profile=False)
    result, jobs = submit(
        tmp_path, project, PIPELINE_STAGE="profile", RUN_NB6="1", RUN_SHAP="1",
        SBATCH_FAIL_AT=str(fail_at), SBATCH_FAIL_MODE=mode,
    )
    assert result.returncode == 2, result.stderr
    assert f"for job {job}" in result.stderr and "(M2)" in result.stderr
    assert len(jobs) == fail_at - 1000 and max(jobs, key=lambda name: int(jobs[name]["id"])) == job
    assert "Submitted profile stage" not in result.stdout
    if mode == "error":
        assert "Batch job submission failed (stub)" in result.stderr  # sbatch's own message is no longer hidden


def test_a_modified_training_file_on_the_first_status_line_is_uncommitted(tmp_path, monkeypatch):
    """`git status --porcelain` keeps its status columns: stripped, " M cvar_mm/gym/shared.py" (modified, not
    staged) parsed as the path "rocs/gym/shared.py" and passed as clean, at a job's start and at its end."""

    project = make_project(tmp_path)
    write_stage_one(project)
    head = git(project, "rev-parse", "HEAD").strip()
    (project / "cvar_mm" / "gym" / "shared.py").write_text("# edited, not staged\n", encoding="utf-8")
    monkeypatch.setattr(snellius_common, "PROJECT_DIR", project)
    state = snellius_common.training_code_state()
    assert state["uncommitted_training_code"] == " M cvar_mm/gym/shared.py"
    assert snellius_common._git_text("status", "--porcelain").splitlines()[0] == " M cvar_mm/gym/shared.py"
    expected = "uncommitted training code [' M cvar_mm/gym/shared.py']"
    assert record_profile_stage.ran_submitted_code(project, head, head, state["uncommitted_training_code"]) == expected


def test_b3_lambda_lr_is_exported_once_as_a_positive_number():
    """D-20, D-24: the committed frozen file exports B3_LAMBDA_LR, the author's sign-off on eta, exactly once, and no
    other uncommented line names it. The value is a positive number as the final stage reads one (the script's
    NUMBER pattern, finite and > 0). The fixtures drop this line, so each test sets eta itself."""

    lines = FROZEN_ENV.read_text(encoding="utf-8").splitlines()
    named = [line for line in lines if "B3_LAMBDA_LR" in line and not line.lstrip().startswith("#")]
    assert len(named) == 1 and named[0].startswith("export B3_LAMBDA_LR="), named
    value = named[0].removeprefix("export B3_LAMBDA_LR=")
    number = re.search(r'^NUMBER = re\.compile\(r"(.+)"\)$', SUBMIT.read_text(encoding="utf-8"), re.MULTILINE)
    assert number, "no NUMBER pattern in submit_final_pipeline.sh, the one the final stage reads eta with"
    assert re.fullmatch(number.group(1), value), value
    assert 0 < float(value) < math.inf


def test_final_stage_needs_eta_in_the_frozen_file(tmp_path):
    project = make_project(tmp_path, post_profile=False)
    # As before the sign-off: a commented-out export, which must not count (the final stage greps for ^export).
    env = project / "configs" / "final_pipeline.env"
    env.write_text(env.read_text(encoding="utf-8") + "# export B3_LAMBDA_LR=0.5\n", encoding="utf-8", newline="\n")
    write_stage_one(project)
    result, jobs = submit(tmp_path, project, PIPELINE_STAGE="final", **POST_PROFILE)  # a shell-only eta
    assert result.returncode == 2
    assert "B3_LAMBDA_LR must be an uncommented export" in result.stderr
    assert jobs == {}


def test_final_stage_trains_nothing_and_runs_the_jan_7_29_jobs(tmp_path):
    """D-19 (d), D-20: no training at all (the profile stage made B1, B2, B1+ and, as the selected eta run, B3),
    no NB6 or SHAP; B1+ and B3 are evaluated and aggregated like the others, and the comparison and the
    inference wait for them (D-18). The record holds, and the changes allowed between the stages pass: eta and
    a time limit committed to the frozen file, and the final-stage-only statistical_inference.py."""

    project = make_project(tmp_path, post_profile=False)
    write_stage_one(project)
    env = project / "configs" / "final_pipeline.env"
    env.write_text(env.read_text(encoding="utf-8") + "export B3_LAMBDA_LR=0.5\n", encoding="utf-8", newline="\n")
    (project / "cvar_mm" / "gym" / "statistical_inference.py").write_text("# an inference fix\n", encoding="utf-8")
    commit_all(project, "eta from the profile stage, and an inference fix")
    set_exports(project, **SENTINEL_TIMES)  # the time limits may change between the stages (D-19)
    result, jobs = submit(
        tmp_path, project, PIPELINE_STAGE="final", RUN_NB6="1", RUN_SHAP="1", MAX_CONCURRENT="6",
    )
    assert result.returncode == 0, result.stderr
    assert_job_table(jobs, FINAL_JOBS)
    assert not [command(job)[0] for job in jobs.values() if command(job)[0].startswith("train_")]  # D-20
    assert "B3 train" not in result.stdout and "B3 eval" in result.stdout
    assert jobs["b0_fair"]["time"] == SENTINEL_TIMES["B0_TIME"]  # B0 tunes gamma on Jan 1-6, then evaluates
    for name in TEST_DAY_EVALUATIONS:  # one Jan 7-29 day per task
        assert jobs[name]["time"] == SENTINEL_TIMES["EVAL_TIME"], name


def test_the_final_stage_accepts_an_eval_time_the_profile_commit_lacks(tmp_path):
    """D-19: between the stages only B3_LAMBDA_LR and the time limits (*_TIME) may change. The profile stage was
    submitted from dab3c6f, whose frozen file has no EVAL_TIME; HEAD adds it with its one-line comment
    (2026-10-13), and the final stage submits, each test-day evaluation with HEAD's EVAL_TIME."""

    project = make_project(tmp_path, post_profile=False)
    env = project / "configs" / "final_pipeline.env"
    head = env.read_text(encoding="utf-8")
    at_profile = "".join(line for line in head.splitlines(keepends=True) if "EVAL_TIME" not in line)
    assert len(head.splitlines()) - len(at_profile.splitlines()) == 2  # the export and its comment: dab3c6f's file
    env.write_text(at_profile, encoding="utf-8", newline="\n")
    commit_all(project, "the frozen file without EVAL_TIME, as at dab3c6f")
    write_stage_one(project)  # the profile stage's commit
    profile_commit = (project / RECORD_COMMIT).read_text(encoding="utf-8").strip()
    assert "EVAL_TIME" not in git(project, "show", f"{profile_commit}:configs/final_pipeline.env")
    env.write_text(head + "export B3_LAMBDA_LR=0.5\n", encoding="utf-8", newline="\n")
    commit_all(project, "EVAL_TIME, and eta from the profile stage")
    result, jobs = submit(tmp_path, project, PIPELINE_STAGE="final")
    assert result.returncode == 0, result.stderr
    assert_job_table(jobs, FINAL_JOBS)
    assert {name: jobs[name]["time"] for name in TEST_DAY_EVALUATIONS} == dict.fromkeys(TEST_DAY_EVALUATIONS, "08:00:00")


def test_the_final_stage_refuses_a_profile_stage_without_fine_tune_lr(tmp_path):
    """D-19, D-23: a profile stage submitted before D-23 (its frozen file has no FINE_TUNE_LR, as at 8a42ad8) trained
    the fine-tunes at B1's 3e-4, so the final stage refuses its record and asks for the profile stage again."""

    project = make_project(tmp_path, post_profile=False)
    env = project / "configs" / "final_pipeline.env"
    head = env.read_text(encoding="utf-8")
    at_profile = "".join(line for line in head.splitlines(keepends=True) if "FINE_TUNE_LR" not in line)
    assert len(head.splitlines()) - len(at_profile.splitlines()) == 1  # the export; its comment does not name it
    env.write_text(at_profile, encoding="utf-8", newline="\n")
    commit_all(project, "the frozen file without FINE_TUNE_LR, as at 8a42ad8")
    write_stage_one(project)  # the profile stage's commit
    env.write_text(head + "export B3_LAMBDA_LR=0.5\n", encoding="utf-8", newline="\n")
    commit_all(project, "FINE_TUNE_LR (D-23), and eta from the profile stage")
    result, jobs = submit(tmp_path, project, PIPELINE_STAGE="final")
    assert result.returncode == 2, result.stderr
    assert "The frozen configuration changed since the profile stage" in result.stderr
    assert jobs == {}


def test_final_stage_refuses_changed_stage_one_artifacts(tmp_path):
    project = make_project(tmp_path)
    write_stage_one(project)
    (project / "results" / "cvar_threshold_d.txt").write_text("0.123\n", encoding="utf-8")
    result, jobs = submit(tmp_path, project, PIPELINE_STAGE="final")
    assert result.returncode == 2
    assert "differ from those eta was profiled on" in result.stderr
    assert jobs == {}


SELECTION = "results/profiling/b3_dual/selection.json"


def edit_selection(**fields):
    """selection.json with some fields changed. The record does not list it, so only the final stage's checks
    of the eta selection (item 7b) see the change."""

    def change(project: Path) -> None:
        path = project / SELECTION
        path.write_text(json.dumps({**json.loads(path.read_text(encoding="utf-8")), **fields}), encoding="utf-8")

    return change


def rewrite_history(eta: float, updates: int = 10, *, reselect: bool):
    """One eta history written again (other lambda values). With `reselect`, selection.json's histories_sha256
    follows it, i.e. the choice was made from that history; without, the history changed after the choice."""

    def change(project: Path) -> None:
        profile = project / "results" / "profiling" / "b3_dual"
        (profile / f"history_eta{eta}.csv").write_text(history_text(eta, updates, lam=1.0), encoding="utf-8", newline="\n")
        if reselect:
            histories = [profile / f"history_eta{other}.csv" for other in ETA_GRID]
            edit_selection(histories_sha256=snellius_common.sha256_lines(project, histories).splitlines())(project)

    return change


def drop_a_history(project: Path) -> None:
    listed = json.loads((project / SELECTION).read_text(encoding="utf-8"))["histories_sha256"]
    edit_selection(histories_sha256=listed[:-1])(project)


@pytest.mark.parametrize(
    ("lambda_lr", "selected", "change", "message"),
    [
        ("2.0", "0.5", None, "B3_LAMBDA_LR=2.0 differs from the profiled choice 0.5"),
        ("2.5", "2.0", None, "B3_LAMBDA_LR=2.5 differs from the profiled choice 2.0"),
        ("", "0.5", None, "B3_LAMBDA_LR='' is not a positive number"),
        ("-0.5", "0.5", None, "B3_LAMBDA_LR='-0.5' is not a positive number"),
        ("0.5", "0.5", edit_selection(training_fill_paths=8), "TRAINING_FILL_PATHS=16 differs from the N of the eta runs, 8"),
        ("0.5", "0.5", rewrite_history(2.0, 9, reselect=True), "history_eta2.0.csv has 9 rows, not the 10 updates D-12a scores"),
        ("0.5", "0.5", edit_selection(rule="D-12"), "rule is 'D-12', not 'D-12a'"),
        ("0.5", "0.5", edit_selection(updates_per_eta=9), "updates_per_eta is 9, not 10"),
        ("0.5", "0.5", rewrite_history(8.0, reselect=False), "history_eta8.0.csv changed after the eta selection"),
        ("0.5", "0.5", drop_a_history, "histories_sha256 must list one history per eta of [0.5, 2.0, 8.0]"),
        ("0.5", "0.5", edit_selection(selected_lambda_lr=2.0), "selected_lambda_lr.txt holds '0.5'"),
        ("0.5", "0.5", lambda project: (project / SELECTION).unlink(), f"Missing {SELECTION}: run PIPELINE_STAGE=profile first"),
    ],
    ids=["eta-mismatch", "eta-2.5-vs-2", "empty-eta", "negative-eta", "n-mismatch", "9-row-history", "rule-not-d12a",
         "9-updates-per-eta", "history-modified", "history-unlisted", "txt-and-json-disagree", "no-selection"],
)
def test_the_final_stage_takes_only_the_eta_selection_d12a_made(tmp_path, lambda_lr, selected, change, message):
    """Item 7b (M1): before any sbatch, B3_LAMBDA_LR, N and the selection itself are checked against D-12a's
    record of the choice; each failure exits 2 and names it."""

    project = make_project(tmp_path, lambda_lr=lambda_lr)
    write_stage_one(project, selected=selected)
    if change is not None:
        change(project)
    result, jobs = submit(tmp_path, project, PIPELINE_STAGE="final")
    assert result.returncode == 2, result.stderr
    assert message in result.stderr
    assert jobs == {}


def test_the_final_stage_needs_n_to_be_a_power_of_two(tmp_path):
    """Item 7b (D-12: N is a power of two). Set in the profile stage's commit too, so the drift check passes."""

    project = make_project(tmp_path)
    set_exports(project, TRAINING_FILL_PATHS="12")
    write_stage_one(project)
    result, jobs = submit(tmp_path, project, PIPELINE_STAGE="final")
    assert result.returncode == 2, result.stderr
    assert "TRAINING_FILL_PATHS='12' is not a power of two >= 2 (D-12)" in result.stderr
    assert jobs == {}


def test_an_eta_of_2_signs_off_the_profiled_2_0(tmp_path):
    """Item 7b: B3_LAMBDA_LR is compared with the choice as a number (the author's sign-off, 2026-10-08), so
    `export B3_LAMBDA_LR=2` signs off the 2.0 in selection.json and selected_lambda_lr.txt, and the stage submits."""

    project = make_project(tmp_path, lambda_lr="2")
    write_stage_one(project, selected="2.0")
    assert (project / "results/profiling/b3_dual/selected_lambda_lr.txt").read_text(encoding="utf-8") == "2.0\n"
    result, jobs = submit(tmp_path, project, PIPELINE_STAGE="final")
    assert result.returncode == 0, result.stderr
    assert_job_table(jobs, FINAL_JOBS)


def test_the_final_stage_needs_the_venv_python_to_read_the_selection(tmp_path):
    project = make_project(tmp_path)
    write_stage_one(project)
    result, jobs = submit(tmp_path, project, PIPELINE_STAGE="final", VENV_PATH=(tmp_path / "no_venv").as_posix())
    assert result.returncode == 2, result.stderr
    assert "VENV_PATH does not contain an executable Python" in result.stderr
    assert jobs == {}


def test_select_writes_an_inputs_sha256_that_the_final_stage_accepts(tmp_path):
    """Item-3 hand-off 2 (item 7b), end to end and meant for Windows: profile_b3_dual.select() writes
    inputs.sha256 with LF line ends, and the final stage's `sha256sum --check` (Git Bash's on Windows) accepts
    it, so the script stops only at its next check, the missing record. The same file with CRLF line ends
    stops at the inputs check on Windows: this test would see a text-mode write."""

    from scripts.pipeline import profile_b3_dual
    from test_b3_dual_profiling import D_AWKWARD, overshoot_grid, write_profile  # the D-12a tests' stand-ins

    project = tmp_path / "project"
    ctx, directory, inputs, _ = write_profile(project, overshoot_grid(D_AWKWARD), D_AWKWARD)
    assert profile_b3_dual.select(ctx, directory, inputs, seed=42) == 2.0
    written = profile_b3_dual.inputs_path(directory)
    assert b"\r" not in written.read_bytes() and written.read_bytes().count(b"\n") == len(inputs) == 6
    (project / "configs").mkdir()
    (project / "configs" / "final_pipeline.env").write_text(
        frozen_text_without_eta() + "export B3_LAMBDA_LR=2.0\n", encoding="utf-8", newline="\n"
    )
    result, jobs = submit(tmp_path, project, PIPELINE_STAGE="final")
    assert (result.returncode, jobs) == (2, {}), result.stderr
    assert "differ from those eta was profiled on" not in result.stderr
    assert f"Missing {RECORD}: run PIPELINE_STAGE=profile first (D-19)" in result.stderr
    written.write_bytes(written.read_bytes().replace(b"\n", b"\r\n"))
    result, jobs = submit(tmp_path, project, PIPELINE_STAGE="final")
    assert (result.returncode, jobs) == (2, {}), result.stderr
    if sys.platform == "win32":  # Git Bash's sha256sum (coreutils 8.32) reads the CR as part of each file name
        assert "differ from those eta was profiled on" in result.stderr


def rerun_b1plus(project: Path) -> None:
    (project / "models" / "ppo_b1plus_doge.zip").write_bytes(b"B1+ retrained after the record")


def rerun_b1_to_eta(project: Path) -> None:
    """A partial rerun of B1 -> d -> eta after B2 and B1+: inputs.sha256 matches the new B1, the record not."""

    (project / "models" / "ppo_b1_doge.zip").write_bytes(b"B1 retrained after B2 and B1+")
    write_inputs_sha256(project)


def add_a_line(project: Path) -> None:
    extra = project / "results" / "extra.txt"
    extra.write_text("extra\n", encoding="utf-8")
    with (project / RECORD).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{hashlib.sha256(extra.read_bytes()).hexdigest()}  results/extra.txt\n")


def drop_a_line(project: Path) -> None:
    lines = (project / RECORD).read_text(encoding="utf-8").splitlines(keepends=True)
    kept = [line for line in lines if not line.rstrip().endswith("models/vecnorm_b1plus.pkl")]
    assert len(kept) == len(lines) - 1
    (project / RECORD).write_text("".join(kept), encoding="utf-8", newline="\n")


def commit_a_change(relative: str):
    def change(project: Path) -> None:
        (project / relative).write_text("# changed after the profile stage\n", encoding="utf-8")
        commit_all(project, f"a change to {relative}")

    return change


def leave_a_training_change_uncommitted(project: Path) -> None:
    (project / "scripts" / "pipeline" / "train_formula_b2.py").write_text("# changed, not committed\n", encoding="utf-8")


def add_an_untracked_training_file(project: Path) -> None:
    (project / "cvar_mm" / "gym" / "new_module.py").write_text("# new, not committed\n", encoding="utf-8")


def hide_an_untracked_training_file(project: Path) -> None:
    git(project, "config", "status.showUntrackedFiles", "no")  # plain `git status` no longer lists it
    add_an_untracked_training_file(project)


def edit_the_frozen_file_without_committing(project: Path) -> None:
    with (project / "configs" / "final_pipeline.env").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("# an edit\n")


def change_a_frozen_value(project: Path) -> None:
    env = project / "configs" / "final_pipeline.env"
    text = env.read_text(encoding="utf-8")
    assert "export TRAINING_FILL_PATHS=16\n" in text
    env.write_text(text.replace("export TRAINING_FILL_PATHS=16\n", "export TRAINING_FILL_PATHS=8\n"), encoding="utf-8", newline="\n")
    commit_all(project, "N changed between the stages")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (rerun_b1plus, "differ from the profile stage's record (D-19, D-20)"),
        (rerun_b1_to_eta, "differ from the profile stage's record (D-19, D-20)"),
        (lambda project: (project / "models" / "ppo_b3_doge.zip").write_bytes(b"B3 changed after the record"),
         "differ from the profile stage's record (D-19, D-20)"),
        (add_a_line, "must list exactly"),
        (drop_a_line, "must list exactly"),
        (lambda project: (project / RECORD).unlink(), "run PIPELINE_STAGE=profile first (D-19)"),
        (lambda project: (project / RECORD_COMMIT).unlink(), "Missing or unknown"),
        *[(commit_a_change(relative), "The training code changed since the profile stage")
          for relative in TRAINING_CODE_FILES if not relative.endswith("statistical_inference.py")],
        (leave_a_training_change_uncommitted, "The training code changed since the profile stage"),
        (add_an_untracked_training_file, "The training code changed since the profile stage"),
        (hide_an_untracked_training_file, "The training code changed since the profile stage"),
        (change_a_frozen_value, "The frozen configuration changed since the profile stage"),
    ],
    ids=["b1plus-rerun", "b1-eta-rerun", "b3-changed", "extra-line", "missing-line", "no-record", "no-commit",
         "cvar_mm-change", "common-change", "b2-script-change", "b3-script-change",
         "uncommitted-code-change", "untracked-code-file", "hidden-untracked-code-file", "frozen-value-change"],
)
def test_final_stage_refuses_what_the_profile_stage_did_not_record(tmp_path, change, message):
    """D-19 (c), and the author's choice of 2026-10-07 to refuse drift between the stages; before any sbatch."""

    project = make_project(tmp_path)
    write_stage_one(project)
    change(project)
    result, jobs = submit(tmp_path, project, PIPELINE_STAGE="final")
    assert result.returncode == 2, result.stderr
    assert message in result.stderr
    assert jobs == {}


@pytest.mark.parametrize(
    "change",
    [leave_a_training_change_uncommitted, add_an_untracked_training_file, hide_an_untracked_training_file,
     edit_the_frozen_file_without_committing],
    ids=["uncommitted-code-change", "untracked-code-file", "hidden-untracked-code-file", "uncommitted-env-change"],
)
def test_the_profile_stage_needs_committed_training_code_and_configuration(tmp_path, change):
    """D-19: one recorded commit must describe the code and the frozen values the whole stage runs with."""

    project = make_project(tmp_path, post_profile=False)
    write_stage_one(project)
    change(project)
    result, jobs = submit(tmp_path, project, PIPELINE_STAGE="profile")
    assert result.returncode == 2, result.stderr
    assert "before the profile stage (D-19)" in result.stderr
    assert jobs == {}


def test_the_profile_stage_needs_a_git_checkout(tmp_path):
    result, jobs = submit(tmp_path, make_project(tmp_path, post_profile=False, git_repo=False), PIPELINE_STAGE="profile")
    assert result.returncode == 2 and "must be a git checkout" in result.stderr
    assert jobs == {}


def test_the_submit_script_checks_what_the_writer_records():
    assert snellius_common.TRAINING_CODE_PATHS == (  # the D-19 training code, pinned
        "cvar_mm",
        ":(exclude)cvar_mm/gym/statistical_inference.py",
        "scripts/pipeline/common.py",
        "scripts/pipeline/train_formula_b2.py",
        "scripts/pipeline/train_formula_b3.py",
    )
    text = SUBMIT.read_text(encoding="utf-8")
    code = re.search(r"^TRAINING_CODE=\((.*)\)$", text, re.MULTILINE).group(1)
    assert tuple(shlex.split(code)) == snellius_common.TRAINING_CODE_PATHS
    recorded = re.search(r'^RECORDED="(.*)"$', text, re.MULTILINE).group(1).split()
    cfg = ReplayExperimentConfig(repo_root=REPO)
    assert recorded == sorted(  # C-locale order, as the script compares them
        path.relative_to(cfg.repo_root).as_posix() for path in record_profile_stage.profile_stage_outputs(cfg)
    )
    assert sorted(STAGE_ONE_FILES[:2] + STAGE_ONE_FILES[4:] + PROFILE_STAGE_FILES + B3_FILES) == recorded


def test_the_writer_records_the_fifteen_files_and_the_submitted_commit_with_lf_line_ends(tmp_path):
    project = make_project(tmp_path)
    write_stage_one(project)
    record = (project / RECORD).read_bytes()
    assert b"\r" not in record
    expected = sorted(
        f"{hashlib.sha256((project / relative).read_bytes()).hexdigest()}  {relative}"
        for relative in STAGE_ONE_FILES[:2] + STAGE_ONE_FILES[4:] + PROFILE_STAGE_FILES + B3_FILES
    )
    assert sorted(record.decode("utf-8").splitlines()) == expected
    # D-20: the promoted B3 is the selected (0.5) eta run, byte for byte
    cfg = ReplayExperimentConfig(repo_root=project)
    sources, targets = snellius_common.eta_run_paths(cfg, 0.5), snellius_common.b3_run_paths(cfg, 42)
    assert {key: path.read_bytes() for key, path in targets.items()} == {key: path.read_bytes() for key, path in sources.items()}
    submitted = (project / RECORD_SUBMITTED).read_text(encoding="utf-8").split()[0]
    assert (project / RECORD_COMMIT).read_text(encoding="utf-8") == f"{submitted}\n"

    record_profile_stage.write_profile_stage_record(ReplayExperimentConfig(repo_root=project))  # once more
    (archive,) = (project / "results" / "profiling" / "superseded").iterdir()
    assert (archive / "profile_stage.sha256").read_bytes() == record == (project / RECORD).read_bytes()


def rewrite_metadata(model: str, **overrides):
    """Final B2's or B1+'s training metadata, written again with some fields changed."""

    def change(project: Path) -> None:
        commit = (project / RECORD_SUBMITTED).read_text(encoding="utf-8").split()[0]
        for path in (project / "results" / "job_metadata").glob("train_formula_*.json"):
            if json.loads(path.read_text(encoding="utf-8"))["model"] == model:
                path.unlink()
        write_training_metadata(project, model, commit, **overrides)

    return change


def commit_other_training_code_and_revert(project: Path) -> str:
    """A commit with other training code, reverted at once: HEAD's training code is the submitted one again."""

    shared = project / "cvar_mm" / "gym" / "shared.py"
    original = shared.read_text(encoding="utf-8")
    shared.write_text("# for a while\n", encoding="utf-8")
    commit_all(project, "a training-code change")
    other = git(project, "rev-parse", "HEAD").strip()
    shared.write_text(original, encoding="utf-8")
    commit_all(project, "reverted")
    return other


def start_from_other_training_code(project: Path) -> None:
    """B1+ started from a commit with other training code, although it ended at the submitted one."""

    other = commit_other_training_code_and_revert(project)
    submitted = (project / RECORD_SUBMITTED).read_text(encoding="utf-8").split()[0]
    for path in (project / "results" / "job_metadata").glob("train_formula_b1plus_*.json"):
        path.unlink()
    record = write_training_metadata(project, "b1plus", submitted)
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload["provenance_at_start"]["git_commit"] = other
    record.write_text(json.dumps(payload), encoding="utf-8")


def add_a_job_with_other_code(project: Path) -> None:
    other = commit_other_training_code_and_revert(project)
    job = {"job_name": "train_formula_b1_cand", "timestamp_utc": TRAINED_AT, "git_commit": other, "git_status_porcelain": ""}
    (project / "results" / "job_metadata" / f"train_formula_b1_cand_local_{TRAINED_AT}.json").write_text(json.dumps(job), encoding="utf-8")


def start_with_uncommitted_training_code(project: Path) -> None:
    """The selected alpha's final B2 started with an uncommitted training-code change, although its commit is the
    submitted one."""

    (path,) = (project / "results" / "job_metadata").glob(f"train_formula_b2_alpha{SELECTED_ALPHA}_*.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["provenance_at_start"]["uncommitted_training_code"] = " M cvar_mm/gym/shared.py"
    path.write_text(json.dumps(payload), encoding="utf-8")


def add_a_job(**fields):
    """A job of the stage, at the submitted commit with a clean tree unless `fields` say otherwise (None drops
    a field: a metadata with no commit or no status shows nothing, so it cannot pass as clean)."""

    def change(project: Path) -> None:
        commit = (project / RECORD_SUBMITTED).read_text(encoding="utf-8").split()[0]
        job = {"job_name": "calibrate_cvar_b3", "timestamp_utc": TRAINED_AT, "git_commit": commit, "git_status_porcelain": "", **fields}
        job = {key: value for key, value in job.items() if value is not None}
        (project / "results" / "job_metadata" / f"calibrate_cvar_b3_local_{TRAINED_AT}.json").write_text(json.dumps(job), encoding="utf-8")

    return change


def rewrite_eta_task(eta: float = 0.5, start: dict | None = None, **fields):
    """The selected eta task's metadata, written again with some fields (or provenance_at_start fields)
    changed (D-20)."""

    def change(project: Path) -> None:
        commit = (project / RECORD_SUBMITTED).read_text(encoding="utf-8").split()[0]
        cfg = ReplayExperimentConfig(repo_root=project)
        for path in (cfg.results_dir / "job_metadata").glob(f"{snellius_common.eta_task_job_name(eta)}_*.json"):
            path.unlink()
        path = write_eta_task_metadata(project, eta, commit, **fields)
        if start:
            record = json.loads(path.read_text(encoding="utf-8"))
            record["provenance_at_start"].update(start)
            path.write_text(json.dumps(record), encoding="utf-8")

    return change


def start_b3_from_other_training_code(project: Path) -> None:
    """The selected eta task (the final B3, D-20) started from a commit with other training code."""

    rewrite_eta_task(start={"git_commit": commit_other_training_code_and_revert(project)})(project)


def drop_the_eta_task(project: Path) -> None:
    for path in (project / "results" / "job_metadata").glob("profile_b3_dual_eta0.5_*.json"):
        path.unlink()


def select_other_eta_by_hand(project: Path) -> None:
    (project / "results/profiling/b3_dual/selected_lambda_lr.txt").write_text("2.0\n", encoding="utf-8")


def eta_from_other_code(project: Path) -> None:
    other = commit_other_training_code_and_revert(project)
    (project / "results" / "profiling" / "b3_dual" / "selection.json").write_text(json.dumps({"git_commit": other}), encoding="utf-8")


def edit_promotion(**fields):
    """b2_promotion.json edited by hand (D-22)."""

    def change(project: Path) -> None:
        path = project / "results" / "b2_promotion.json"
        path.write_text(json.dumps({**json.loads(path.read_text(encoding="utf-8")), **fields}), encoding="utf-8")

    return change


def rewrite(relative: str, text: str):
    def change(project: Path) -> None:
        (project / relative).write_text(text, encoding="utf-8")

    return change


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        (lambda project: (project / "models" / "vecnorm_b1plus.pkl").unlink(), FileNotFoundError, "incomplete"),
        (leave_a_training_change_uncommitted, RuntimeError, "uncommitted or untracked"),
        (add_an_untracked_training_file, RuntimeError, "uncommitted or untracked"),
        (hide_an_untracked_training_file, RuntimeError, "uncommitted or untracked"),
        (commit_a_change("cvar_mm/gym/shared.py"), RuntimeError, "HEAD's training code differs"),
        (lambda project: (project / RECORD_SUBMITTED).unlink(), FileNotFoundError, "submit PIPELINE_STAGE=profile"),
        (lambda project: (project / "results/profiling/b3_dual/selection.json").unlink(), FileNotFoundError, "eta selection"),
        (eta_from_other_code, RuntimeError, "did not run the training code"),
        (add_a_job_with_other_code, RuntimeError, "did not run the training code"),
        (add_a_job(git_status_porcelain=" M scripts/pipeline/common.py\n?? results/x.csv"), RuntimeError, "uncommitted training code"),
        (add_a_job(git_status_porcelain="R  docs/old.py -> cvar_mm/gym/moved.py"), RuntimeError, "uncommitted training code"),
        (add_a_job(git_status_porcelain=None), RuntimeError, "no git status recorded"),
        (add_a_job(git_commit=None), RuntimeError, "no commit recorded"),
        (add_a_job(timestamp_utc=None), ValueError, "no timestamp_utc"),
        # The lineage: a rerun recorded next to a final B2 or B1+ trained before it (D-11, D-18).
        (rewrite("models/ppo_b1_doge.zip", "B1 rerun"), RuntimeError, "not trained from the current B1"),
        (rewrite("models/vecnorm_b1.pkl", "B1 rerun"), RuntimeError, "B1's normalizer"),
        (rewrite("results/formula_train_params.json", "prep rerun"), RuntimeError, "calibration"),
        (rewrite("results/formula_manifest.csv", "prep rerun"), RuntimeError, "manifest"),
        # D-22: the final B2 is the promoted parallel final B2 of the selected alpha, and nothing else.
        (rewrite("results/b2_best_alpha.txt", "10.0"), RuntimeError, "alpha = 1.0, but b2_best_alpha.txt selects 10.0"),
        (edit_promotion(alpha=10.0), RuntimeError, "alpha = 10.0, but b2_best_alpha.txt selects 1.0"),
        (rewrite("results/b2_alpha_sweep_summary.csv", "a rerun\n"), RuntimeError, "selection changed after b2_promote"),
        (rewrite("models/ppo_b2_doge.zip", "B2 trained by hand"), RuntimeError, "differ from the run b2_promote promoted"),
        (rewrite("models/vecnorm_b2.pkl", "another normalizer"), RuntimeError, "differ from the run b2_promote promoted"),
        (lambda project: (project / "results" / "b2_promotion.json").unlink(), FileNotFoundError, "b2_promote"),
        (rewrite_metadata("b2", timestamp_utc="20261007T000000Z"), RuntimeError, "No job metadata of a final B2"),
        (rewrite_metadata("b1plus", risk_aware_updates=3), RuntimeError, "k = 10"),
        (rewrite_metadata("b2", training_seed=43), RuntimeError, "seed 42"),
        (rewrite_metadata("b1plus", training_fill_paths=8), RuntimeError, "different N"),
        (rewrite_metadata("b2", alpha=10.0), RuntimeError, "No job metadata of a final B2 with alpha = 1.0"),
        (start_from_other_training_code, RuntimeError, "did not start from the submitted training code"),
        (start_with_uncommitted_training_code, RuntimeError, "did not start from the submitted training code"),
        # D-20: the promoted B3 is the selected eta task's run, at the submitted code, k = 10, seed 42, one N, d.
        (rewrite("models/ppo_b3_doge.zip", "B3 retrained by hand"), RuntimeError, "differ from the eta run the selection promoted"),
        (rewrite("results/b3_training_feasibility.json", "{}"), RuntimeError, "differ from the eta run the selection promoted"),
        (start_b3_from_other_training_code, RuntimeError, "B3 .* did not start from the submitted training code"),
        (rewrite_eta_task(start={"uncommitted_training_code": " M cvar_mm/gym/shared.py"}), RuntimeError,
         "B3 .* did not start from the submitted training code"),
        (drop_the_eta_task, RuntimeError, "No eta task of this profile stage saved the promoted B3"),
        (rewrite_eta_task(timestamp_utc="20261007T000000Z"), RuntimeError, "No eta task of this profile stage"),
        (rewrite_eta_task(risk_aware_updates=3), RuntimeError, "k = 10"),
        (rewrite_eta_task(training_fill_paths=8), RuntimeError, "different N than final B2"),
        (rewrite_eta_task(threshold_d=0.07), RuntimeError, "at the current d"),
        (rewrite("results/cvar_threshold_d.txt", "0.07\n"), RuntimeError, "not trained from the current threshold_d"),
        (rewrite("results/profiling/b3_dual/history_eta2.0.csv", "rerun\n"), RuntimeError, "changed after the eta selection"),
        (select_other_eta_by_hand, RuntimeError, "The eta selection is incomplete"),
    ],
    ids=["missing-file", "uncommitted-code", "untracked-code", "hidden-untracked-code", "head-moved",
         "no-submission", "no-eta-selection", "eta-other-code", "job-other-code", "job-uncommitted-code",
         "job-renamed-into-code", "job-no-status", "job-no-commit", "job-no-timestamp", "b1-rerun", "b1-normalizer-rerun", "calibration-rerun", "manifest-rerun", "selection-rerun",
         "promotion-alpha-edited", "sweep-summary-rerun", "b2-model-tampered", "b2-normalizer-tampered", "no-promotion",
         "b2-trained-before-submission", "k-3", "seed-43", "other-n", "other-alpha", "b1plus-started-elsewhere", "b2-started-dirty",
         "b3-model-rewritten", "b3-output-rewritten", "b3-started-elsewhere", "b3-started-dirty", "no-eta-task",
         "eta-task-before-submission", "b3-k-3", "b3-other-n", "b3-other-d", "d-rerun", "eta-history-changed",
         "selection-edited"],
)
def test_the_writer_moves_the_old_record_aside_then_refuses_without_writing(tmp_path, change, error, message):
    """D-19: no record unless the whole stage ran the submitted training code and final B2 and B1+ were
    trained from the current B1, normalizer, calibration and manifest, with k = 10, seed 42, one N; D-22 test (b):
    nor unless the final B2 is, byte for byte, the promoted run of the alpha b2_best_alpha.txt selects."""

    project = make_project(tmp_path)
    write_stage_one(project)
    change(project)
    with pytest.raises(error, match=message):
        record_profile_stage.write_profile_stage_record(ReplayExperimentConfig(repo_root=project))
    assert not (project / RECORD).exists() and not (project / RECORD_COMMIT).exists()
    (archive,) = (project / "results" / "profiling" / "superseded").iterdir()
    assert sorted(path.name for path in archive.iterdir()) == ["profile_stage.commit", "profile_stage.sha256"]


def test_the_writer_accepts_changes_outside_the_training_code(tmp_path):
    """statistical_inference.py, the docs and the results are not training code (D-19): neither a job's status
    nor the checkout now may block the record for them. The record names the submitted commit, which the jobs
    ran, not a later HEAD."""

    project = make_project(tmp_path)
    write_stage_one(project)
    (project / "docs").mkdir(exist_ok=True)
    (project / "docs" / "notes.md").write_text("a note during the profile stage\n", encoding="utf-8")
    commit_all(project, "a docs commit during the profile stage")
    add_a_job(git_status_porcelain=" M cvar_mm/gym/statistical_inference.py\n?? results/x.csv\nR  docs/a.md -> docs/b.md")(project)
    (project / "cvar_mm" / "gym" / "statistical_inference.py").write_text("# changed, not committed\n", encoding="utf-8")
    record_profile_stage.write_profile_stage_record(ReplayExperimentConfig(repo_root=project))
    submitted = (project / RECORD_SUBMITTED).read_text(encoding="utf-8").split()[0]
    assert submitted != git(project, "rev-parse", "HEAD").strip()
    assert (project / RECORD_COMMIT).read_text(encoding="utf-8") == f"{submitted}\n"


def test_a_job_start_sees_untracked_training_code_that_git_status_hides(tmp_path, monkeypatch):
    """provenance_at_start lists an untracked training file even under status.showUntrackedFiles=no (D-19)."""

    project = make_project(tmp_path)
    (project / "cvar_mm" / "gym").mkdir(parents=True, exist_ok=True)
    hide_an_untracked_training_file(project)
    monkeypatch.setattr(snellius_common, "PROJECT_DIR", project)
    state = snellius_common.training_code_state()
    assert state["git_commit"] == git(project, "rev-parse", "HEAD").strip()
    assert "cvar_mm/gym/new_module.py" in state["uncommitted_training_code"]


def test_the_writer_covers_training_seed_42_only(monkeypatch):
    monkeypatch.setattr(record_profile_stage, "load_context", lambda **_: pytest.fail("loaded before the seed check"))
    monkeypatch.setattr(sys, "argv", ["record_profile_stage.py", "--seed", "43"])
    with pytest.raises(ValueError, match="seed 42 only"):
        record_profile_stage.main()


def test_scripts_import_the_same_names_as_files_and_as_modules():
    """The jobs run each script as a file (run_python_job.sh: the `else` branch); tests import it as a module
    (the `if __package__` branch). Both branches must import the same names."""

    for path in sorted((REPO / "scripts" / "pipeline").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "__package__":
                def names(body):
                    return sorted(alias.asname or alias.name for statement in body
                                  if isinstance(statement, ast.ImportFrom) for alias in statement.names)

                assert names(node.body) == names(node.orelse), path.name


def test_python_defaults_equal_the_frozen_values(monkeypatch):
    frozen = frozen_values()
    cfg = ReplayExperimentConfig()
    assert float(frozen["FORMULA_GAMMA_MIN"]) == snellius_common.DEFAULT_FORMULA_GAMMA_MIN == cfg.as_gamma_range[0]
    assert float(frozen["FORMULA_GAMMA_MAX"]) == snellius_common.DEFAULT_FORMULA_GAMMA_MAX == cfg.as_gamma_range[1]
    assert float(frozen["FORMULA_SKEW_TICKS_MAX"]) == snellius_common.DEFAULT_FORMULA_SKEW_TICKS_MAX
    assert int(frozen["FEATURE_WINDOW"]) == cfg.feature_window
    assert int(frozen["EVALUATION_ROLLOUTS"]) == cfg.evaluation_rollouts
    assert int(frozen["CVAR_ROLLOUTS_PER_SEED"]) == cfg.evaluation_rollouts  # calibrate_cvar_b3 default
    assert float(frozen["FIXED_CLOCK_SECONDS"]) == cfg.fixed_clock_seconds
    assert int(frozen["TRAINING_SEED"]) == snellius_common.DEFAULT_TRAINING_SEED
    assert tuple(float(alpha) for alpha in frozen["B2_ALPHAS"].split()) == snellius_common.DEFAULT_ALPHAS
    assert int(frozen["TRAINING_FILL_PATHS"]) == snellius_common.DEFAULT_TRAINING_FILL_PATHS
    monkeypatch.delenv("EVALUATION_FILL_SEEDS", raising=False)
    assert snellius_common.parse_seed_list() == [int(seed) for seed in frozen["EVALUATION_FILL_SEEDS"].split()]
    assert snellius_common.parse_seed_list() == [42, 43]

    for name in (
        "CALIBRATION_CENSOR_SECONDS",
        "CALIBRATION_START_STRIDE_SECONDS",
        "CALIBRATION_MIN_HITS",
        "CALIBRATION_VOL_WINDOW_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    for name in ("CALIBRATION_PROXY", "CALIBRATION_START_RULE", "CALIBRATION_DEPTH_TICKS"):
        monkeypatch.setenv(name, frozen[name])
    calibration = snellius_common.calibration_config_from_env(1e-5)
    assert calibration.censor_horizon_seconds == float(frozen["CALIBRATION_CENSOR_SECONDS"])
    assert calibration.exposure_stride_seconds == float(frozen["CALIBRATION_START_STRIDE_SECONDS"])
    assert calibration.min_hits_per_depth == int(frozen["CALIBRATION_MIN_HITS"])
    assert calibration.volatility_window_seconds == float(frozen["CALIBRATION_VOL_WINDOW_SECONDS"])
    assert calibration.depth_ticks == tuple(float(depth) for depth in frozen["CALIBRATION_DEPTH_TICKS"].split())


def test_first_passage_defaults_equal_the_frozen_calibration(monkeypatch):
    """Item 8: FirstPassageCalibrationConfig's defaults are the F-03 calibration in configs/final_pipeline.env,
    which is also exactly what the pipeline builds from that file."""

    from dataclasses import asdict

    from cvar_mm.gym.calibration import FirstPassageCalibrationConfig

    frozen = frozen_values()
    default = FirstPassageCalibrationConfig()
    assert (default.execution_proxy, default.exposure_start_rule) == (
        frozen["CALIBRATION_PROXY"], frozen["CALIBRATION_START_RULE"]) == ("best_price_reach", "fixed_stride")
    assert default.depth_ticks == tuple(float(depth) for depth in frozen["CALIBRATION_DEPTH_TICKS"].split())
    assert default.censor_horizon_seconds == float(frozen["CALIBRATION_CENSOR_SECONDS"]) == 15.0
    assert default.exposure_stride_seconds == float(frozen["CALIBRATION_START_STRIDE_SECONDS"]) == 15.0
    assert default.min_hits_per_depth == int(frozen["CALIBRATION_MIN_HITS"]) == 10
    assert default.volatility_window_seconds == float(frozen["CALIBRATION_VOL_WINDOW_SECONDS"]) == 600.0
    assert default.tick_size == ReplayExperimentConfig().tick_size
    for name, value in frozen.items():
        if name.startswith("CALIBRATION_"):
            monkeypatch.setenv(name, value)
    assert asdict(snellius_common.calibration_config_from_env(ReplayExperimentConfig().tick_size)) == asdict(default)


def test_snellius_readme_follows_the_frozen_configuration():
    readme = (REPO / "scripts" / "pipeline" / "README.md").read_text(encoding="utf-8")
    assert "source configs/final_pipeline.env" in readme
    assert "PIPELINE_STAGE=profile" in readme and "PIPELINE_STAGE=final" in readme
    for stale in (
        "midpoint_first_passage",
        "non_overlapping",
        "B3_NU_LR",
        "B3_TRAJECTORY_BATCH_CYCLES",
        "TOTAL_TIMESTEPS=",
        "/gpfs/",
    ):
        assert stale not in readme
    for name, value in frozen_values().items():
        for line in readme.splitlines():
            if line.strip().startswith((f"{name}=", f"export {name}=")):
                assert value in line
