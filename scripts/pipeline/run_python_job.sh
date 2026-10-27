#!/bin/bash
# Generic Snellius runner for one Python pipeline script.
# Submit with:
#   sbatch --export=ALL,PROJECT_DIR=/home/user/thesis,DATA_DIR=/home/user/datasets,VENV_PATH=/home/user/venvs/thesis_corrected,SCRIPT_PATH=scripts/pipeline/train_formula_b1.py,SCRIPT_ARGS="--phase candidate" scripts/pipeline/run_python_job.sh

#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --job-name=formula_job
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if [[ -z "${PROJECT_DIR:-}" ]]; then
  echo "PROJECT_DIR is required" >&2
  exit 2
fi
if [[ -z "${DATA_DIR:-}" ]]; then
  echo "DATA_DIR is required" >&2
  exit 2
fi
if [[ -z "${VENV_PATH:-}" ]]; then
  echo "VENV_PATH is required" >&2
  exit 2
fi
if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  echo "VENV_PATH does not contain an executable Python: $VENV_PATH" >&2
  exit 2
fi
if [[ -z "${SCRIPT_PATH:-}" ]]; then
  echo "SCRIPT_PATH is required" >&2
  exit 2
fi

mkdir -p "$PROJECT_DIR/logs"

# Do not inherit an interactive Conda/Python activation into the batch job.
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL
unset CONDA_EXE CONDA_PYTHON_EXE _CE_CONDA _CE_M
unset PYTHONPATH PYTHONHOME

module load 2023
module load Miniconda3/23.5.2-0

# Use the validated thesis virtual environment explicitly.
# Do not inherit Python packages from ~/.local.
export PYTHONNOUSERSITE=1
export VIRTUAL_ENV="$VENV_PATH"
export PATH="$VENV_PATH/bin:$PATH"
PYTHON="$VENV_PATH/bin/python"

cd "$PROJECT_DIR"

echo "[$(date)] Host      : $(hostname)"
echo "[$(date)] Job       : ${SLURM_JOB_ID:-local}"
echo "[$(date)] Array task: ${SLURM_ARRAY_TASK_ID:-none}"
echo "[$(date)] Script    : $SCRIPT_PATH ${SCRIPT_ARGS:-}"
echo "[$(date)] DATA_DIR  : $DATA_DIR"
echo "[$(date)] VENV_PATH : $VENV_PATH"
echo "[$(date)] Python    : $PYTHON"
echo "[$(date)] Git HEAD  : $(git rev-parse HEAD 2>/dev/null || echo unknown)"

"$PYTHON" "$SCRIPT_PATH" ${SCRIPT_ARGS:-}

echo "[$(date)] Finished $SCRIPT_PATH"
