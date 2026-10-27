#!/bin/bash
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE _CE_CONDA _CE_M
unset PYTHONPATH PYTHONHOME
module load 2023
module load Miniconda3/23.5.2-0
# The repository checkout, the folder of the Tardis files and the Python environment: export them before
# sbatch, which passes the environment to the job. The logs go to logs/ in the folder you submit from.
: "${PROJECT_DIR:?export PROJECT_DIR, the repository checkout}"
: "${DATA_DIR:?export DATA_DIR, the folder of the Tardis files}"
: "${VENV_PATH:?export VENV_PATH, the Python virtual environment}"
export PROJECT_DIR DATA_DIR VENV_PATH
export PYTHONNOUSERSITE=1
export VIRTUAL_ENV="$VENV_PATH"
export PATH="$VENV_PATH/bin:$PATH"
cd "$PROJECT_DIR"
source configs/final_pipeline.env
echo "[$(date -u +%FT%TZ)] host $(hostname) job ${SLURM_JOB_ID:-local} commit $(git rev-parse HEAD)"
echo "[$(date -u +%FT%TZ)] args: $*"
"$VENV_PATH/bin/python" -m hummingbot_adapter.shadow_replay "$@"
echo "[$(date -u +%FT%TZ)] finished"
