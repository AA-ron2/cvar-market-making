#!/bin/bash
# Submit the corrected formula-AS pipeline as Snellius jobs, in two stages (DECISIONS D-19;
# scripts/pipeline/README.md):
#   PIPELINE_STAGE=profile  training and validation days only: prep, B1, d, eta profiling and selection (the
#                           selected eta run is the final B3, D-20); B1_cand, the B2 candidates with their
#                           Jan 5-6 validation and selection; one final B2 per alpha, in parallel, and the
#                           promotion of the selected one (D-22); B1+; the NB6 ablation and SHAP. No job reads
#                           Jan 7-29 (prep only hashes those files).
#   PIPELINE_STAGE=final    B0 and every Jan 7-29 evaluation, aggregation, comparison and inference of the
#                           profile stage's checked models; nothing is trained (D-19, D-20)
#
# Required environment:
#   PROJECT_DIR=/home/<user>/thesis
#   DATA_DIR=/home/<user>/datasets
#   VENV_PATH=/home/<user>/venvs/thesis_corrected
# Frozen values come only from configs/final_pipeline.env, which is sourced here.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
FROZEN_ENV="$PROJECT_DIR/configs/final_pipeline.env"
if [[ ! -f "$FROZEN_ENV" ]]; then
  echo "Missing frozen configuration $FROZEN_ENV" >&2
  exit 2
fi

# K7: a frozen value already set by the caller must equal the file; never override silently.
mapfile -t FROZEN_NAMES < <(sed -nE 's/^export ([A-Za-z_][A-Za-z0-9_]*)=.*/\1/p' "$FROZEN_ENV")
declare -A CALLER_VALUES=()
for name in "${FROZEN_NAMES[@]}"; do
  if [[ -n "${!name+x}" ]]; then
    CALLER_VALUES[$name]="${!name}"
  fi
done
# shellcheck source=/dev/null
source "$FROZEN_ENV"
# D-19: k = 10 for B2 (all phases), B3 and B1+, fixed a priori. Checked before the generic comparison
# below (exit 1), so any k other than 10 exits 2: the caller's value, the file's export line and the
# sourced value must all be exactly 10.
for name in B2_UPDATES B3_UPDATES; do
  if [[ -n "${CALLER_VALUES[$name]+x}" && "${CALLER_VALUES[$name]}" != "10" ]] \
     || ! grep -qx "export ${name}=10" "$FROZEN_ENV" || [[ "${!name:-}" != "10" ]]; then
    echo "$name must be exactly 10, exported in $FROZEN_ENV and not overridden (DECISIONS D-19: k = 10, fixed a priori)" >&2
    exit 2
  fi
done
# F-16: B1 and B1_cand each train exactly one pass over their days, checked like k. TOTAL_TIMESTEPS is no
# longer used or exported; a leftover one in the caller's shell stops the submission.
if [[ -n "${CALLER_VALUES[B1_PASSES]+x}" && "${CALLER_VALUES[B1_PASSES]}" != "1" ]] \
   || ! grep -qx "export B1_PASSES=1" "$FROZEN_ENV" || [[ "${B1_PASSES:-}" != "1" ]]; then
  echo "B1_PASSES must be exactly 1, exported in $FROZEN_ENV and not overridden (DECISIONS F-16)" >&2
  exit 2
fi
if [[ -n "${TOTAL_TIMESTEPS+x}" ]]; then
  echo "TOTAL_TIMESTEPS is set: B1 and B1_cand train exactly B1_PASSES = 1 pass and refuse it (DECISIONS F-16); run 'unset TOTAL_TIMESTEPS'" >&2
  exit 2
fi
# D-21: every PPO policy explores with a fixed sigma0 = 0.3 in action units, checked like k (grep -F: the dot
# is literal). The jobs read it through --export=ALL; B1, B1_cand and NB6 train with it and every warm start and
# the d job refuse a B1 with another sigma.
if [[ -n "${CALLER_VALUES[POLICY_STD]+x}" && "${CALLER_VALUES[POLICY_STD]}" != "0.3" ]] \
   || ! grep -qxF "export POLICY_STD=0.3" "$FROZEN_ENV" || [[ "${POLICY_STD:-}" != "0.3" ]]; then
  echo "POLICY_STD must be exactly 0.3, exported in $FROZEN_ENV and not overridden (DECISIONS D-21: fixed exploration sigma0 = 0.3)" >&2
  exit 2
fi
# D-23: every fine-tune (B2 in all phases, B3 through the eta profiler, B1+) trains with PPO learning rate 3e-3,
# checked like k (grep -F: the dot is literal). The jobs read it through --export=ALL; B1, B1_cand and NB6 keep
# 3e-4 and never read it.
if [[ -n "${CALLER_VALUES[FINE_TUNE_LR]+x}" && "${CALLER_VALUES[FINE_TUNE_LR]}" != "0.003" ]] \
   || ! grep -qxF "export FINE_TUNE_LR=0.003" "$FROZEN_ENV" || [[ "${FINE_TUNE_LR:-}" != "0.003" ]]; then
  echo "FINE_TUNE_LR must be exactly 0.003, exported in $FROZEN_ENV and not overridden (DECISIONS D-23: fine-tune learning rate 3e-3)" >&2
  exit 2
fi
# D-19 (runtime, not methodology): every Slurm time limit is an explicit, non-zero HH:MM:SS export in the
# runtime block of the frozen file; the file's line is checked too, since a caller's value survives `source`.
for name in B1_TIME B1_CAND_TIME CVAR_CAL_TIME DUAL_PROFILE_TIME RISK_TRAIN_TIME ABLATION_TIME B0_TIME SHAP_TIME EVAL_TIME; do
  if ! grep -qE "^export ${name}=" "$FROZEN_ENV" || [[ ! "${!name:-}" =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ ]] \
     || [[ "${!name}" =~ ^0+:00:00$ ]]; then
    echo "$name must be exported in $FROZEN_ENV as a non-zero HH:MM:SS time limit, got '${!name:-}' (D-19)" >&2
    exit 2
  fi
done
mismatch=0
for name in "${!CALLER_VALUES[@]}"; do
  if [[ "${CALLER_VALUES[$name]}" != "${!name}" ]]; then
    echo "Frozen value mismatch: $name='${CALLER_VALUES[$name]}' in the environment, '${!name}' in $FROZEN_ENV" >&2
    mismatch=1
  fi
done
if (( mismatch )); then
  exit 1
fi

PIPELINE_STAGE="${PIPELINE_STAGE:-}"
if [[ "$PIPELINE_STAGE" != "profile" && "$PIPELINE_STAGE" != "final" ]]; then
  echo "Set PIPELINE_STAGE=profile (training and validation days only) or PIPELINE_STAGE=final; see scripts/pipeline/README.md" >&2
  exit 2
fi
if [[ "$PIPELINE_STAGE" == "final" ]]; then
  for name in B3_LAMBDA_LR TRAINING_FILL_PATHS; do
    if ! grep -qE "^export ${name}=" "$FROZEN_ENV"; then
      echo "$name must be an uncommented export in $FROZEN_ENV before the final stage (DECISIONS D-12, D-12a)" >&2
      exit 2
    fi
  done
fi

VENV_PATH="${VENV_PATH:-$HOME/venvs/thesis_corrected}"
MAX_CONCURRENT="${MAX_CONCURRENT:-6}"
# NB6 and SHAP are training- and validation-side diagnostics of the profile stage (D-19, F-19): on unless switched off.
RUN_NB6="${RUN_NB6:-1}"
RUN_SHAP="${RUN_SHAP:-1}"
SMOKE="${SMOKE:-0}"
if [[ -z "${DATA_DIR:-}" ]]; then
  echo "DATA_DIR is required, e.g. export DATA_DIR=/home/<user>/datasets" >&2
  exit 2
fi
if [[ "$SMOKE" == "1" ]]; then
  echo "SMOKE=1 is disabled: truncating real market days violates the corrected full-day objectives." >&2
  echo "Run the synthetic/unit smoke tests locally instead." >&2
  exit 2
fi

cd "$PROJECT_DIR"
PROFILE_DIR="results/profiling/b3_dual"
# D-19: the profile stage's record of what the final stage takes from it (record_profile_stage.py).
PROFILE_RECORD="results/profiling/profile_stage.sha256"
PROFILE_COMMIT="results/profiling/profile_stage.commit"
PROFILE_SUBMITTED="results/profiling/profile_stage.submitted"
RECORDED="models/ppo_b1_doge.zip models/ppo_b1plus_doge.zip models/ppo_b2_doge.zip models/ppo_b3_doge.zip models/vecnorm_b1.pkl models/vecnorm_b1plus.pkl models/vecnorm_b2.pkl models/vecnorm_b3.pkl results/b2_alpha_sweep_summary.csv results/b2_best_alpha.txt results/b3_cvar_lagrangian_history.csv results/b3_training_costs.csv results/b3_training_feasibility.json results/formula_manifest.csv results/formula_train_params.json"
# The training code B2, B1+ and B3 share (record_profile_stage.TRAINING_CODE_PATHS).
TRAINING_CODE=(cvar_mm ':(exclude)cvar_mm/gym/statistical_inference.py' scripts/pipeline/common.py scripts/pipeline/train_formula_b2.py scripts/pipeline/train_formula_b3.py)
if [[ "$PIPELINE_STAGE" == "final" ]]; then
  for required in "$PROFILE_DIR/inputs.sha256" "$PROFILE_DIR/selected_lambda_lr.txt" "$PROFILE_DIR/selection.json"; do
    if [[ ! -f "$required" ]]; then
      echo "Missing $required: run PIPELINE_STAGE=profile first" >&2
      exit 2
    fi
  done
  if ! sha256sum --check --quiet "$PROFILE_DIR/inputs.sha256"; then
    echo "B1, its normalizer, d or the calibration differ from those eta was profiled on" >&2
    exit 2
  fi
  # D-19, D-20, D-22: the promoted final B2, B1+, the promoted B3 and the B2 selection, with the B1, calibration
  # and manifest they came from, are exactly the profile stage's.
  if [[ ! -f "$PROFILE_RECORD" ]]; then
    echo "Missing $PROFILE_RECORD: run PIPELINE_STAGE=profile first (D-19)" >&2
    exit 2
  fi
  if [[ "$(cut -c67- "$PROFILE_RECORD" | LC_ALL=C sort | paste -sd ' ' -)" != "$RECORDED" ]]; then
    echo "$PROFILE_RECORD must list exactly: $RECORDED (D-19)" >&2
    exit 2
  fi
  if ! sha256sum --check --quiet "$PROFILE_RECORD"; then
    echo "B1, final B2, B1+, B3, the B2 selection or the calibration differ from the profile stage's record (D-19, D-20)" >&2
    exit 2
  fi
  # D-19, D-20: the final stage evaluates with the training code the profile stage trained B1, B2, B1+ and B3 with.
  profile_commit="$(tr -d '[:space:]' < "$PROFILE_COMMIT" 2>/dev/null || true)"
  if [[ -z "$profile_commit" ]] || ! git cat-file -e "${profile_commit}^{commit}" 2>/dev/null; then
    echo "Missing or unknown $PROFILE_COMMIT: run PIPELINE_STAGE=profile first (D-19)" >&2
    exit 2
  fi
  if ! git diff --quiet "$profile_commit" HEAD -- "${TRAINING_CODE[@]}" \
     || [[ -n "$(git status --porcelain --untracked-files=all -- "${TRAINING_CODE[@]}")" ]]; then
    git diff --stat "$profile_commit" -- "${TRAINING_CODE[@]}" >&2 || true
    echo "The training code changed since the profile stage ($profile_commit): rerun PIPELINE_STAGE=profile (D-11, D-18, D-19)" >&2
    exit 2
  fi
  # D-19: the frozen configuration is the profile stage's too; only B3_LAMBDA_LR and the time limits may change.
  frozen_exports() {
    sed -nE 's/^export ([A-Za-z_][A-Za-z0-9_]*=.*)$/\1/p' | grep -vE '^(B3_LAMBDA_LR|[A-Za-z0-9_]*_TIME)=' | LC_ALL=C sort
  }
  if [[ "$(git show "${profile_commit}:configs/final_pipeline.env" | frozen_exports)" != "$(frozen_exports < "$FROZEN_ENV")" ]]; then
    echo "The frozen configuration changed since the profile stage ($profile_commit): rerun PIPELINE_STAGE=profile (D-19)" >&2
    exit 2
  fi
  # Item 7b (M1): the eta selection the final stage evaluates was made by D-12a, from the 10-update histories it
  # hashed, and B3_LAMBDA_LR and N match it. B3_LAMBDA_LR stays the author's explicit sign-off on the selected eta
  # (2026-10-08): a positive number numerically equal to the choice, so "2" signs off 2.0. Read with the venv's
  # Python (standard library only); any failed check exits 2.
  if [[ ! -x "$VENV_PATH/bin/python" ]]; then
    echo "VENV_PATH does not contain an executable Python: $VENV_PATH (the final stage reads $PROFILE_DIR/selection.json with it)" >&2
    exit 2
  fi
  if ! "$VENV_PATH/bin/python" -I - "$PROFILE_DIR" "${B3_LAMBDA_LR:-}" "${TRAINING_FILL_PATHS:-}" <<'PY'
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

directory, eta_text, paths_text = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
path = (directory / "selection.json").as_posix()
selection = json.loads(Path(path).read_text(encoding="utf-8"))
NUMBER = re.compile(r"(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def stop(message):
    print(message, file=sys.stderr)
    sys.exit(2)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


if selection.get("rule") != "D-12a":
    stop(f"{path}: rule is {selection.get('rule')!r}, not 'D-12a': eta was not chosen by the training-only rule (D-12a)")
updates = selection.get("updates_per_eta")
if type(updates) is not int or updates != 10:
    stop(f"{path}: updates_per_eta is {updates!r}, not 10 (D-12a scores updates 8-10; k = 10, D-19)")
grid, selected = selection.get("eta_grid"), selection.get("selected_lambda_lr")
if not isinstance(grid, list) or not grid or not all(positive(eta) for eta in grid):
    stop(f"{path}: eta_grid is {grid!r}, not a list of positive numbers (D-12a)")
if not positive(selected) or selected not in grid:
    stop(f"{path}: selected_lambda_lr is {selected!r}, not an eta of the grid {grid} (D-12a)")
chosen = (directory / "selected_lambda_lr.txt").read_text(encoding="utf-8").strip()
if not NUMBER.fullmatch(chosen) or float(chosen) != selected:
    stop(f"{directory.as_posix()}/selected_lambda_lr.txt holds '{chosen}', {path} {selected}: rerun profile_b3_dual.py --select (D-12a, D-20)")
if not NUMBER.fullmatch(eta_text) or not 0 < float(eta_text) < math.inf:
    stop(f"B3_LAMBDA_LR='{eta_text}' is not a positive number: set it to the profiled choice {selected} (D-12a)")
if float(eta_text) != selected:
    stop(f"B3_LAMBDA_LR={eta_text} differs from the profiled choice {selected} in {path} (D-12a)")
recorded = selection.get("training_fill_paths")
if not re.fullmatch(r"[1-9][0-9]*", paths_text) or int(paths_text) < 2 or int(paths_text) & (int(paths_text) - 1):
    stop(f"TRAINING_FILL_PATHS='{paths_text}' is not a power of two >= 2 (D-12)")
if type(recorded) is not int or int(paths_text) != recorded:
    stop(f"TRAINING_FILL_PATHS={paths_text} differs from the N of the eta runs, {recorded!r} in {path} (D-12, D-20)")
# Item-3 hand-off 1: the histories the choice was made from, one per eta, unchanged since, each with 10 updates.
lines = selection.get("histories_sha256")
pairs = [str(line).partition("  ") for line in lines] if isinstance(lines, list) else []
listed = {relative: digest for digest, _, relative in pairs}
expected = sorted((directory / f"history_eta{float(eta)}.csv").as_posix() for eta in grid)
if len(listed) != len(pairs) or sorted(listed) != expected:
    stop(f"{path}: histories_sha256 must list one history per eta of {grid}, i.e. {expected} (D-12a)")
for relative, digest in sorted(listed.items()):
    history = Path(relative)
    if not history.is_file() or hashlib.sha256(history.read_bytes()).hexdigest() != digest:
        stop(f"{relative} changed after the eta selection (histories_sha256 in {path}): rerun profile_b3_dual.py --select, then record_profile_stage.py (D-12a, D-20)")
    with history.open(encoding="utf-8", newline="") as handle:
        rows = len(list(csv.reader(handle))) - 1
    if rows != 10:
        stop(f"{relative} has {rows} rows, not the 10 updates D-12a scores (D-12a, D-19)")
PY
  then
    exit 2
  fi
fi
if [[ "$PIPELINE_STAGE" == "profile" ]]; then
  # D-19: the profile stage runs one committed training code and frozen configuration. The record job checks
  # every job of the stage against the commit written below, and the final stage checks it again.
  if ! git rev-parse --verify --quiet HEAD >/dev/null 2>&1; then
    echo "PROJECT_DIR must be a git checkout with a commit: the profile stage records it (D-19)" >&2
    exit 2
  fi
  uncommitted="$(git status --porcelain --untracked-files=all -- "${TRAINING_CODE[@]}" configs/final_pipeline.env)"
  if [[ -n "$uncommitted" ]]; then
    echo "Commit the training code and configs/final_pipeline.env before the profile stage (D-19):" >&2
    echo "$uncommitted" >&2
    exit 2
  fi
fi
mkdir -p logs results/eval_parts results/job_metadata

RUNNER="scripts/pipeline/run_python_job.sh"

submit_job() {
  local name="$1"
  local script="$2"
  local args="${3:-}"
  local dep="${4:-}"
  local array="${5:-}"
  local time_limit="${6:-06:00:00}"

  local cmd=(sbatch --parsable --job-name "$name" --time "$time_limit")
  if [[ -n "$dep" ]]; then
    cmd+=(--dependency "$dep")
  fi
  if [[ -n "$array" ]]; then
    cmd+=(--array "$array")
  fi
  cmd+=(--export=ALL,PROJECT_DIR="$PROJECT_DIR",DATA_DIR="$DATA_DIR",VENV_PATH="$VENV_PATH",SCRIPT_PATH="$script",SCRIPT_ARGS="$args",ABLATION_TIMESTEPS="${ABLATION_TIMESTEPS:-}",B2_ALPHAS="$B2_ALPHAS",EVALUATION_ROLLOUTS="${EVALUATION_ROLLOUTS:-}",EVALUATION_FILL_SEEDS="$EVALUATION_FILL_SEEDS",TRAINING_SEED="${TRAINING_SEED:-42}",CVAR_ROLLOUTS_PER_SEED="${CVAR_ROLLOUTS_PER_SEED:-}",B3_LAMBDA_LR="${B3_LAMBDA_LR:-}",FIXED_CLOCK_SECONDS="${FIXED_CLOCK_SECONDS:-60}",FORMULA_GAMMA_MIN="${FORMULA_GAMMA_MIN:-}",FORMULA_GAMMA_MAX="${FORMULA_GAMMA_MAX:-}",FORMULA_SKEW_TICKS_MAX="${FORMULA_SKEW_TICKS_MAX:-}")
  cmd+=("$RUNNER")
  # M2: a failed sbatch, or one that returns no numeric job ID, stops the submission here (exit 2, naming the
  # job), so no dependant gets an empty afterok:. Errors go to stderr: stdout is the job ID.
  local out line
  if ! out="$("${cmd[@]}")"; then
    echo "sbatch failed for job $name; nothing after it was submitted, and the jobs before it are still queued (M2)" >&2
    exit 2
  fi
  line="${out##*$'\n'}"
  line="${line%$'\r'}"
  if [[ ! "$line" =~ ^([0-9]+)(;.*)?$ ]]; then
    echo "sbatch returned no numeric job ID for job $name (output: '$out'); nothing after it was submitted, and the jobs before it are still queued (M2)" >&2
    exit 2
  fi
  echo "${BASH_REMATCH[1]}"
}

TEST_ARRAY="0-22%${MAX_CONCURRENT}"
VALIDATION_ARRAY="0-1%2"
AGG_PARTIAL_ARG=""

echo "Submitting formula-AS pipeline stage '$PIPELINE_STAGE' from $PROJECT_DIR"
echo "DATA_DIR=$DATA_DIR"

if [[ "$PIPELINE_STAGE" == "profile" ]]; then
  # D-19: an earlier record must not stand in for this stage if one of its jobs fails.
  if [[ -f "$PROFILE_RECORD" || -f "$PROFILE_COMMIT" || -f "$PROFILE_SUBMITTED" ]]; then
    superseded="results/profiling/superseded/$(date -u +%Y%m%dT%H%M%S.%NZ)"
    mkdir -p "$(dirname "$superseded")"
    mkdir "$superseded"
    for recorded_file in "$PROFILE_RECORD" "$PROFILE_COMMIT" "$PROFILE_SUBMITTED"; do
      if [[ -f "$recorded_file" ]]; then
        mv "$recorded_file" "$superseded/"
      fi
    done
    echo "Moved the previous profile-stage record to $superseded"
  fi
  # D-19: the commit and the time this stage was submitted at, which the record job checks every job against.
  mkdir -p results/profiling
  printf '%s %s\n' "$(git rev-parse HEAD)" "$(date -u +%Y%m%dT%H%M%SZ)" > "$PROFILE_SUBMITTED"
  # D-19: every training- and validation-side job, and no job that reads Jan 7-29 (prep only hashes those
  # files). k = 10 is fixed a priori, and the eta rule reads Jan 1-6 only (D-12a).
  prepare_job=$(submit_job "prep_formula" "scripts/pipeline/prepare_formula_pipeline.py" "--force" "" "" "01:00:00")
  b1_train=$(submit_job "b1_train" "scripts/pipeline/train_formula_b1.py" "" "afterok:${prepare_job}" "" "$B1_TIME")
  cvar_cal=$(submit_job "b3_cvarcal" "scripts/pipeline/calibrate_cvar_b3.py" "" "afterok:${b1_train}" "" "$CVAR_CAL_TIME")
  b3_dual=$(submit_job "b3_dual" "scripts/pipeline/profile_b3_dual.py" "" "afterok:${cvar_cal}" "0-2" "$DUAL_PROFILE_TIME")
  b3_dual_select=$(submit_job "b3_dual_select" "scripts/pipeline/profile_b3_dual.py" "--select" "afterok:${b3_dual}" "" "00:30:00")

  # B2 (F-05, D-11): candidates from B1_cand on Jan 1-4, validated on Jan 5-6; the final B2 from the final B1.
  # D-22: one final B2 per alpha trains from the final B1 while the candidates run, and none reads the selection;
  # b2_promote then copies the selected alpha's to the final B2's names with SHA-256 (as D-20 does for B3).
  b1_cand=$(submit_job "b1_cand" "scripts/pipeline/train_formula_b1.py" "--phase candidate" "afterok:${prepare_job}" "" "$B1_CAND_TIME")
  b2_agg_jobs=()
  for alpha in $B2_ALPHAS; do
    train_job=$(submit_job "b2_a${alpha}" "scripts/pipeline/train_formula_b2.py" "--phase candidate --alpha ${alpha}" "afterok:${b1_cand}" "" "$RISK_TRAIN_TIME")
    eval_job=$(submit_job "b2e_a${alpha}" "scripts/pipeline/evaluate_formula_model.py" "--model b2 --split validation --alpha ${alpha}" "afterok:${train_job}" "$VALIDATION_ARRAY" "$EVAL_TIME")
    agg_job=$(submit_job "b2g_a${alpha}" "scripts/pipeline/aggregate_final_results.py" "--model b2 --split validation --alpha ${alpha} ${AGG_PARTIAL_ARG}" "afterok:${eval_job}" "" "01:00:00")
    b2_agg_jobs+=("$agg_job")
  done
  b2_agg_dep=$(IFS=:; echo "afterok:${b2_agg_jobs[*]}")
  b2_select=$(submit_job "b2_select" "scripts/pipeline/aggregate_final_results.py" "--select-b2" "$b2_agg_dep" "" "01:00:00")
  b2_final_jobs=()
  for alpha in $B2_ALPHAS; do
    final_job=$(submit_job "b2_final_a${alpha}" "scripts/pipeline/train_formula_b2.py" "--phase final --alpha ${alpha}" "afterok:${b1_train}" "" "$RISK_TRAIN_TIME")
    b2_final_jobs+=("$final_job")
  done
  b2_promote_dep=$(IFS=:; echo "afterok:${b2_select}:${b2_final_jobs[*]}")
  b2_promote=$(submit_job "b2_promote" "scripts/pipeline/train_formula_b2.py" "--promote" "$b2_promote_dep" "" "00:30:00")
  # B1+ (D-18): the final B1 through the B2 path at alpha = 0; it needs the final B1 only.
  b1plus_train=$(submit_job "b1plus_train" "scripts/pipeline/train_formula_b2.py" "--phase final --control b1plus" "afterok:${b1_train}" "" "$RISK_TRAIN_TIME")
  # D-19: the record the final stage checks, once the promoted final B2, B1+ and the eta selection exist; it refuses
  # to write unless every job of this stage ran the submitted training code, the final B2 is the promoted run of the
  # selected alpha (D-22), and B2 and B1+ trained from the current inputs.
  profile_record=$(submit_job "profile_record" "scripts/pipeline/record_profile_stage.py" "" "afterok:${b2_promote}:${b1plus_train}:${b3_dual_select}" "" "00:30:00")

  # Training- and validation-side diagnostics (F-19): NB6 trains on Jan 1-4 and evaluates on Jan 5-6; SHAP reads Jan 1.
  if [[ "$RUN_NB6" == "1" ]]; then
    ablation_jobs=()
    for feature_set in base_state base_plus_vol full; do
      ablation_job=$(submit_job "abl_${feature_set}" "scripts/pipeline/run_feature_ablation.py" "--feature-set ${feature_set}" "afterok:${prepare_job}" "" "$ABLATION_TIME")
      ablation_jobs+=("$ablation_job")
    done
    ablation_dep=$(IFS=:; echo "afterok:${ablation_jobs[*]}")
    submit_job "abl_agg" "scripts/pipeline/aggregate_final_results.py" "--ablation" "$ablation_dep" "" "01:00:00" >/dev/null
  fi
  if [[ "$RUN_SHAP" == "1" ]]; then
    submit_job "b1_shap" "scripts/pipeline/run_shap_diagnostics.py" "" "afterok:${b1_train}" "" "$SHAP_TIME" >/dev/null
  fi

  echo "Submitted profile stage (training and validation days only, D-19):"
  echo "  prepare      $prepare_job"
  echo "  B1 train     $b1_train"
  echo "  B3 d         $cvar_cal"
  echo "  eta profile  $b3_dual (array, one eta per task)"
  echo "  eta select   $b3_dual_select"
  echo "  B1 cand      $b1_cand"
  echo "  B2 select    $b2_select"
  echo "  B2 finals    ${b2_final_jobs[*]} (one per alpha: ${B2_ALPHAS}; D-22)"
  echo "  B2 promote   $b2_promote"
  echo "  B1+ train    $b1plus_train"
  echo "  record       $profile_record"
  echo "Then the go/no-go review: $PROFILE_DIR/summary.md (eta; its run is the final B3, D-20), results/b2_alpha_sweep_summary.csv and"
  echo "b2_best_alpha.txt (B2 selection), results/b2_promotion.json (D-22), the final B2 and B1+ training metadata,"
  echo "the NB6 results, and"
  echo "$PROFILE_RECORD. Do not pull or edit training code until the final stage is submitted. If you rerun"
  echo "a job, rerun every job that depends on it, then record_profile_stage.py (it refuses a stale lineage)."
  echo "Set B3_LAMBDA_LR (from $PROFILE_DIR) in $FROZEN_ENV (k = 10 is frozen, D-19), commit it,"
  echo "and submit PIPELINE_STAGE=final."
  exit 0
fi

# Final stage (D-19, D-20): B0 and every Jan 7-29 evaluation, aggregation, comparison and inference. Nothing is
# trained: B1, final B2, B1+ and B3 (the promoted eta run) come from the profile stage, checked above.
b0_job=$(submit_job "b0_fair" "scripts/pipeline/run_b0_fair.py" "" "" "" "$B0_TIME")
b1_eval=$(submit_job "b1_eval" "scripts/pipeline/evaluate_formula_model.py" "--model b1" "" "$TEST_ARRAY" "$EVAL_TIME")
b1_agg=$(submit_job "b1_agg" "scripts/pipeline/aggregate_final_results.py" "--model b1 ${AGG_PARTIAL_ARG}" "afterok:${b1_eval}" "" "01:00:00")
b2_eval=$(submit_job "b2_eval" "scripts/pipeline/evaluate_formula_model.py" "--model b2" "" "$TEST_ARRAY" "$EVAL_TIME")
b2_agg=$(submit_job "b2_agg" "scripts/pipeline/aggregate_final_results.py" "--model b2 ${AGG_PARTIAL_ARG}" "afterok:${b2_eval}" "" "01:00:00")
b1plus_eval=$(submit_job "b1plus_eval" "scripts/pipeline/evaluate_formula_model.py" "--model b1plus" "" "$TEST_ARRAY" "$EVAL_TIME")
b1plus_agg=$(submit_job "b1plus_agg" "scripts/pipeline/aggregate_final_results.py" "--model b1plus ${AGG_PARTIAL_ARG}" "afterok:${b1plus_eval}" "" "01:00:00")

# D-20: B3 is evaluated like B1, B2 and B1+; each task first checks the promoted B3 against the record.
b3_eval=$(submit_job "b3_eval" "scripts/pipeline/evaluate_formula_model.py" "--model b3" "" "$TEST_ARRAY" "$EVAL_TIME")
b3_agg=$(submit_job "b3_agg" "scripts/pipeline/aggregate_final_results.py" "--model b3 ${AGG_PARTIAL_ARG}" "afterok:${b3_eval}" "" "01:00:00")

# The comparison and the inference wait for the B1+ control too (D-18).
compare_dep="afterok:${b0_job}:${b1_agg}:${b2_agg}:${b3_agg}:${b1plus_agg}"
comparison=$(submit_job "final_compare" "scripts/pipeline/aggregate_final_results.py" "--comparison" "$compare_dep" "" "01:00:00")
inference=$(submit_job "paired_infer" "scripts/pipeline/run_paired_inference.py" "" "$compare_dep" "" "01:00:00")

echo "Submitted final stage:"
echo "  B0           $b0_job"
echo "  B1 eval      $b1_eval"
echo "  B2 eval      $b2_eval"
echo "  B1+ eval     $b1plus_eval"
echo "  B3 eval      $b3_eval"
echo "  comparison   $comparison"
echo "  inference    $inference"
echo
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  sacct -u \$USER --format=JobID,JobName,State,ExitCode,Elapsed -X"
