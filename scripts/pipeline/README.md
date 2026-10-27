# The thesis pipeline (SURF Snellius, Slurm)

These scripts ran the thesis's corrected pipeline on Snellius compute nodes. They train the four PPO models (B1, B2,
B3 and the B1⁺ control), tune and run the analytical baseline B0, evaluate every model on the 23 test days and run the
inference. [docs/model.md](../../docs/model.md) specifies what each model computes, and
[docs/design_decisions.md](../../docs/design_decisions.md) gives the reasons (F- and D-numbers below).

The thesis results come from commit cf3096d, where this folder was `scripts/snellius/` and the package `procs/`
([docs/reproduce.md](../../docs/reproduce.md)). The scripts here are the same code under the public names.

- **Data split:** Jan 1–4 candidate training, Jan 5–6 validation, Jan 1–6 final refit, Jan 7–29 retrospective
  out-of-sample test (F-02).
- **B2:** α is selected on the validation days only; the candidates use the Jan 1–4 calibration and start from
  B1_cand (F-05, D-11).
- **Final B2, B3 and B1⁺:** all three start from the final B1 with its frozen normalizer and run k = 10 PPO
  iterations (D-11, D-18, D-19).
- **B3** constrains the RU-CVaR₀.₂ of the full-day maximum drawdown with the D-12 optimizer.
- **Two stages (D-19, D-20):** every training- and validation-side job first, B3 included (the selected η-profiling
  run is the final B3); B0 and every Jan 7–29 job only after the author's go/no-go review.

## Frozen configuration

Every frozen value lives in `configs/final_pipeline.env`, and `submit_final_pipeline.sh` sources that file. If a
value already set in your shell differs from the file, the script exits with status 1; the comparison is exact text,
so `15` and `15.0` differ. These exit with status 2 instead:
- k: `B2_UPDATES` or `B3_UPDATES` other than exactly 10, in your shell or in the file (D-19);
- `B1_PASSES` other than exactly 1 (F-16);
- `POLICY_STD` other than exactly 0.3, the fixed exploration σ₀ of every PPO policy (D-21);
- `FINE_TUNE_LR` other than exactly 0.003 (`3e-3` included), the learning rate of every fine-tune (D-23);
- a timestep budget exported in your shell, even an empty one: B1 and B1_cand train exactly one pass and refuse it
  (F-16).

A frozen value changes only through a new design decision.

## Running the pipeline

Set up the shell on a Snellius login node. `<PROJECT_DIR>` is your checkout of this repository, `<DATA_DIR>` the
folder with the 29 market data files ([data/README.md](../../data/README.md)) and `<VENV>` your Python environment.

```bash
cd <PROJECT_DIR> && git pull
source <VENV>/bin/activate
export PYTHONNOUSERSITE=1; unset PYTHONPATH
export PROJECT_DIR=<PROJECT_DIR> DATA_DIR=<DATA_DIR> VENV_PATH=<VENV>
source configs/final_pipeline.env
```

### Stage 1: profile (training and validation days only)

```bash
PIPELINE_STAGE=profile bash scripts/pipeline/submit_final_pipeline.sh
```

This submits every training- and validation-side job:
1. the prepare job: the manifest and the Jan 1–6 and Jan 1–4 calibrations;
2. B1, after prep. B1, B1_cand and the NB6 ablations explore with σ fixed at `POLICY_STD` = 0.3 in action units
   (D-21);
3. the B3 threshold d = 0.8 × CVaR₀.₂ of B1's full-day MaxDD on Jan 1–6, after B1, from stochastic rollouts; the
   deterministic CVaR is recorded, not used (D-17). It refuses a B1 whose σ is not `POLICY_STD`, and
   `cvar_threshold_b3.json` records `b1_pinned_share`, the share of B1's deterministic decisions at an action bound;
4. `profile_b3_dual.py`, one array task per η ∈ {0.5, 2, 8}, each running 10 B3 updates on Jan 1–6 with training seed
   42. Each run has the final B3's configuration (D-20), so each task saves it as a B3 run;
5. the η selection (D-12a): an η is excluded if λ reaches λ_max or Ĥ − d changes sign more than twice; among the rest
   the smallest mean |Ĥ_u − d|/d over updates 8–10 wins, ties going to the smaller η (a tie by construction, when the
   λ used is 0 at every update for every η, chooses 0.5). If every η is excluded, the job writes `summary.md`, fails and
   selects nothing; there is no fallback η. The selected run is copied to `models/ppo_b3_doge.zip`,
   `models/vecnorm_b3.pkl` and `results/b3_*`, with its SHA-256 in `selection.json`;
6. B1_cand (Jan 1–4, the Jan 1–4 calibration), after prep;
7. the B2 candidates α ∈ {0.1, 1, 10} from B1_cand, each validated on Jan 5–6, then the B2 selection (F-05);
8. one final B2 per α (Jan 1–6, from the final B1), in parallel with the candidates; then `b2_promote` copies the
   selected α's model and normalizer to `models/ppo_b2_doge.zip` and `models/vecnorm_b2.pkl` and records their SHA-256
   in `results/b2_promotion.json` (D-22);
9. B1⁺ (D-18), the final B1 through the B2 code path at α = 0;
10. the NB6 feature ablation (`RUN_NB6=1`: trains on Jan 1–4, evaluates on Jan 5–6) and SHAP (`RUN_SHAP=1`: B1 on
    Jan 1). Both are on by default.

**No job reads Jan 7–29.** No job loads market data from a Jan 7–29 file or reads a file derived from such data; the
prepare job only hashes those files' bytes for the manifest.

**Time limits** are runtime settings, not methodology (D-19): explicit `HH:MM:SS` values in the runtime block of
`configs/final_pipeline.env`, with no defaults. The thesis run used 24 h for B1, B1_cand, each NB6 job and B0, 48 h for
the d calibration, 72 h for each η task and each B2 or B1⁺ job, and 8 h for SHAP and each evaluation task. If `sbatch`
fails or answers without a job ID, the script stops at once (exit 2) and names the job; cancel the jobs it already
submitted before you resubmit.

### The go/no-go review

Before anything of stage 2 runs, the author reviewed:
- `results/profiling/b3_dual/summary.md`: B1's σ per action dimension, d with the stochastic and deterministic
  CVaR₀.₂, and per η the λ and Ĥ/d series, the sign changes, the exclusion reason, the epochs completed and KL stops,
  the Adam damping ratio, the action shift against B1, the critic's value loss and explained variance, the seconds per
  update and the peak memory;
- `results/b2_alpha_sweep_summary.csv`, `b2_best_alpha.txt` and `results/b2_promotion.json`;
- the final B2s' and B1⁺'s training metadata in `results/job_metadata/` (learning rate 0.003 and the per-update
  diagnostics);
- `results/formula_feature_ablation_results.csv`;
- the profile record, `results/profiling/profile_stage.sha256` and `profile_stage.commit`. The submission refuses
  uncommitted or untracked training code and records its commit and time in `profile_stage.submitted`. The record
  job runs after `b2_promote`, B1⁺ and the η selection and writes nothing unless the lineage holds: every job ran the
  submitted training code; the final B2 is the promoted run of the selected α; final B2, B1⁺ and B3 started from the
  same B1, normalizer, calibration and manifest, with k = 10, seed 42 and one N.

Do not pull or edit the training code while the profile stage runs. If you rerun a job, rerun every job that depends
on it, then the record job, which refuses a stale lineage.

In the thesis run, the selected η (8.0) was then written into `configs/final_pipeline.env` as `B3_LAMBDA_LR`, the
author's sign-off; the file in this repository holds that value.

### Stage 2: final (Jan 7–29)

```bash
PIPELINE_STAGE=final bash scripts/pipeline/submit_final_pipeline.sh
```

Before it submits anything, the script checks:
- η is exported in the frozen file, and k, σ₀ and the fine-tune learning rate equal their frozen values;
- `sha256sum -c results/profiling/b3_dual/inputs.sha256` passes: B1, its normalizer, d, the calibration and the
  manifest are exactly the files η was profiled on;
- `results/profiling/profile_stage.sha256` lists exactly its fifteen files and `sha256sum -c` passes: B1 and final B2
  with their normalizers, B1⁺ and its normalizer, `b2_best_alpha.txt`, `b2_alpha_sweep_summary.csv`, the calibration,
  the manifest, and the promoted B3 (model, normalizer, history, costs and feasibility);
- the training code is unchanged since `profile_stage.commit`, with nothing uncommitted or untracked. The training
  code is `cvar_mm/` except `cvar_mm/gym/statistical_inference.py`, plus `scripts/pipeline/common.py`,
  `train_formula_b2.py` and `train_formula_b3.py`;
- the frozen values equal that commit's, except `B3_LAMBDA_LR` and the time limits;
- the η selection in `results/profiling/b3_dual/selection.json` is the D-12a one, with 10 updates per η, η equal to
  `B3_LAMBDA_LR`, and unchanged histories.

Every failed check exits 2 before any job is submitted. It then submits B0 (γ tuned on Jan 1–6, evaluated on
Jan 7–29), the evaluations of B1, final B2, B1⁺ and B3 on Jan 7–29 (one array task per day; each B3 task first checks
B3 against the record) and their aggregations, then the comparison tables and the paired inference. Nothing in this
stage trains.

## Optional settings (not frozen)

```bash
MAX_CONCURRENT=6           # test-day array concurrency
RUN_NB6=1                  # feature ablations (stage 1); 0 skips them
RUN_SHAP=1                 # SHAP diagnostics (stage 1); 0 skips them
```

## Tests

The test suites need no market data; the tests that do skip when `DATA_DIR` is unset.

```bash
python -m pytest -q -rs tests
python -m pytest -q -rs hummingbot_adapter/tests
```

Do not truncate real replay days: that would invalidate the full-day B2 and B3 objectives.

## Main outputs

```text
results/formula_manifest.csv, results/formula_train_params.{json,csv}
models/ppo_b1_doge.zip, models/vecnorm_b1.pkl                    final B1
models/ppo_b1_cand_doge.zip, models/vecnorm_b1_cand.pkl          B1_cand (Jan 1-4)
results/cvar_threshold_d.txt, results/cvar_threshold_b3.json     d (D-17)
results/profiling/b3_dual/                                       eta profiling, selection.json, selected_lambda_lr.txt, summary.md
models/ppo_b3_doge.zip, models/vecnorm_b3.pkl                    final B3: the selected eta run (D-20)
results/b3_cvar_lagrangian_history.csv, b3_training_costs.csv, b3_training_feasibility.json
results/b2_alpha{alpha}_validation_results.csv, b2_alpha_sweep_summary.csv, b2_best_alpha.txt
models/ppo_b2_doge.zip, models/vecnorm_b2.pkl, results/b2_promotion.json      final B2 (D-22)
models/ppo_b1plus_doge.zip, models/vecnorm_b1plus.pkl            B1+ control (D-18)
results/profiling/profile_stage.sha256, profile_stage.commit, profile_stage.submitted   the profile record
results/b0_gamma_fixed.txt, b0_train_calibration_params.csv      B0
results/b{0,1,2,1plus,3}_test_results.csv, b{0,1,2,1plus,3}_test_episodes.csv
results/comparison_table.csv, overall_summary_table.csv
results/paired_day_inference_primary.csv, paired_day_inference.csv
results/formula_feature_ablation_results.csv, results/ablation/, results/b1_shap_feature_importance.csv
results/job_metadata/*.json
```

Each evaluated day also writes two part files under `results/eval_parts/<split>/<model>/`. The thesis's own outputs
are published in `results/final/` and the models as release assets (the [README](../../README.md)).

## Calibration profiling (historical)

The frozen calibration (F-03) was chosen with `scripts/pipeline/profile_formula_calibration.py`. It compares four
execution-proxy and start-rule candidates on Jan 1–4 only, writes under `results/profiling/calibration/`, and never
writes the canonical calibration files. It is kept as evidence; the calibration is never re-selected from P&L, Sharpe,
MaxDD or January 5–29.

## Monitoring

```bash
squeue -u "$USER"
sacct -u "$USER" --format=JobID,JobName,State,ExitCode,Elapsed -X
tail -f logs/*.out logs/*.err
```
