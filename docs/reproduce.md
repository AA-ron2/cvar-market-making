# Reproducing the thesis

## What the published numbers are

The thesis results come from the corrected pipeline run on SURF Snellius on 2026-10-14 and 2026-10-15. The profile
stage ran at commit fe5f9fe and the final stage at commit cf3096d of the private research repository (tag
`thesis-final-cf3096d`). Their outputs are published here as produced:
- the result tables in [`results/final/`](../results/final/README.md);
- the four final models and their normalizers as assets of the GitHub release v1.0.0.

A few files differ from their originals in local paths only, and three model files in one metadata field; each is
listed in [`results/final/REDACTIONS.md`](../results/final/REDACTIONS.md) with both hashes.

The code in this repository is cf3096d's thesis code with two renames: the package `procs` is `cvar_mm`, and
`scripts/snellius` is `scripts/pipeline`. Dead modules were removed, and personal paths and documentation were
cleaned. The nine modules that compute the thesis results differ from cf3096d only in their import paths (and a
licence notice in one of them). The test suites check the code; the published numbers were not recomputed from this
tree.

Code comments and docstrings cite the private repository's documents by their old names: "DECISIONS F-xx" and
"DECISIONS D-xx" are the entries of [`docs/design_decisions.md`](design_decisions.md), and "MODEL_SPEC §n" is
[`docs/model.md`](model.md), whose sections keep the same numbers. WORK_ORDER, AGENTS, PROGRESS and DESIGN_SYSTEM are
internal working documents of the private repository and are not published.

## Environment

- Python 3.11. `pyproject.toml` lists the dependencies as ranges: `pip install -e ".[test]"`.
- `requirements-lock.txt` is the pip freeze of the thesis run on Snellius (Linux x86_64, Python 3.11.4, CUDA build of
  torch 2.11.0). For the same versions on a CPU machine, follow the tests workflow
  ([`.github/workflows/tests.yml`](../.github/workflows/tests.yml)):

  ```bash
  python -m pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cpu
  grep -v -E '^(torch|torchvision|torchaudio|triton)==|^nvidia-' requirements-lock.txt > constraints.txt
  python -m pip install -c constraints.txt -e ".[test]"
  ```

- The tests need no market data: `python -m pytest -q -rs tests` and
  `python -m pytest -q -rs hummingbot_adapter/tests`. The tests that need data skip when `DATA_DIR` (the market data
  files) or `HB_ARTIFACTS` (the forward test's frozen inputs) is unset. Some tests run `bash`, `git` and `sha256sum`.

## Data

Binance Spot DOGEUSDT order-book snapshots (the top 25 levels per side), 2025-01-01 to 2025-01-29: 29 files named
`binance_book_snapshot_25_YYYY-MM-DD_DOGEUSDT.csv` ([data/README.md](../data/README.md)). The data is licensed
third-party data and is not included. `results/final/profiling/formula_manifest.csv` holds the SHA-256 of each file the
thesis used.

## The two pipeline stages

The pipeline runs on a Slurm cluster ([scripts/pipeline/README.md](../scripts/pipeline/README.md) has the details):

1. **Profile** (training and validation days only): `PIPELINE_STAGE=profile bash scripts/pipeline/submit_final_pipeline.sh`.
   It runs the calibration, B1 and B1_cand, the threshold d, the η profiling and selection (the selected run is the
   final B3), the B2 candidates with their validation and selection, the final B2s and the promotion, B1⁺, the NB6
   ablation and SHAP, and then writes the profile record.
2. **The go/no-go review** of the profile stage's outputs, before any test-day output exists. In the thesis run the
   review found no defect, B3 did not reach d within its 10 updates (D-24), and the selected η = 8.0 was signed off
   in `configs/final_pipeline.env` (`B3_LAMBDA_LR`).
3. **Final** (Jan 7–29): `PIPELINE_STAGE=final bash scripts/pipeline/submit_final_pipeline.sh`. It checks the profile
   record and the training code, then runs B0, the evaluation of B1, B2, B1⁺ and B3, the comparison and the paired
   inference. It trains nothing.

Run both stages from this repository: the final stage checks the training code against the profile stage's commit in
the same checkout. The thesis's own record names cf3096d's ancestor fe5f9fe and the old paths, so it cannot stand in
for a profile stage of this tree. The training code, the frozen values and the seeds are the thesis's, so a rerun
follows the same procedure; bit-identical outputs are expected only with the same package versions and hardware.

## Compute

About 10,541 core-hours on SURF Snellius (CPU nodes, rome partition) for the corrected pipeline, 3,707 of them for the
final stage. The longest jobs are the η-profiling tasks and the B2 and B1⁺ fine-tunes, with 10 updates each on Jan 1–6
and 16 fill paths: each update of the three η runs took about 1.1–1.6 hours (the "seconds" column of
`results/final/profiling/b3_dual/summary.md`). A final-stage evaluation task (one test day, 2 fill seeds × 20 paths)
takes about 2 h 10 min to 2 h 47 min.

## The trained models

The release v1.0.0 assets are the final B1, B2, B1⁺ and B3 models (Stable Baselines 3 zips) and their VecNormalize
normalizers. Download them into one folder and check them:

```bash
python scripts/analysis/verify_release.py --assets <folder>
```

It checks the files against `results/final/release_assets.sha256`, checks `results/final/` against its `SHA256SUMS`,
and reports, for every file the thesis's profile record lists, whether it is identical to the thesis output or a
documented change. The release's ninth asset, `SHA256SUMS`, holds the SHA-256 of the eight files (it is identical to
`results/final/release_assets.sha256`): download it into the same folder, and `sha256sum -c SHA256SUMS`, run there,
checks the assets alone.

`ppo_b3_doge.zip` and the four normalizers are byte-identical to the thesis files. In `ppo_b1_doge.zip`,
`ppo_b2_doge.zip` and `ppo_b1plus_doge.zip` one metadata field, the training-log path (`tensorboard_log`), was set to
null; every other member of the archives, the weights included, is unchanged (`results/final/REDACTIONS.md` gives the
hashes of every member). The models load with `PPO.load(path, device="cpu")` and the normalizers with `pickle.load`,
without this repository's code, given NumPy ≥ 2.0 and stable-baselines3 2.8.

## Where each thesis table comes from

Numbers and short titles as in the thesis PDF. Paths are relative to `results/final/`; `results/final/README.md`
describes every file.

| Thesis | Files |
|---|---|
| Section 3.1 "Order-book data" and Figure 3.1 "Data split": the 29 files and their roles | `profiling/formula_manifest.csv` |
| Section 3.2 "Calibration": σ, A and κ on Jan 1–6, and on Jan 1–4 for the B2 candidates | `profiling/formula_train_params.json` (both fits), `profiling/formula_train_params.csv` (Jan 1–6 per day) |
| Section 6.1 "Training and selection (Jan 1–6)": threshold d, B2 selection, η selection | `profiling/cvar_threshold_d.txt`, `profiling/cvar_threshold_b3.json`, `profiling/b2_alpha*_validation_results.csv`, `profiling/b2_alpha_sweep_summary.csv`, `profiling/b2_best_alpha.txt`, `profiling/b2_promotion.json`, `profiling/b3_dual/selection.json`, `profiling/b3_dual/selected_lambda_lr.txt`, `profiling/b3_dual/summary.md` |
| Figure 6.1 "B3's constraint statistic and multiplier during training" | `profiling/b3_dual/history_eta0.5.csv`, `history_eta2.0.csv`, `history_eta8.0.csv` |
| Table 6.1 "Movement of the fine-tunes away from B1" | B3's row: `profiling/b3_cvar_lagrangian_history.csv`; the B2 and B1⁺ rows come from their training metadata, which is not published |
| Section 6.1 "Training and selection (Jan 1–6)": B3's training feasibility | `profiling/b3_training_feasibility.json`, `profiling/b3_training_costs.csv` |
| Section 6.1 "Training and selection (Jan 1–6)": SHAP and the feature ablation | `diagnostics/b1_shap_feature_importance.csv`, `diagnostics/formula_feature_ablation_results.csv`, `diagnostics/ablation/` |
| Table 6.2 "Retrospective out-of-sample results" | `main/overall_summary_table.csv`, `main/comparison_table.csv` (days best), `main/b{0,1,1plus,2,3}_test_results.csv`; the 920-path CVaR from `main/b*_test_episodes.csv` |
| Section 4.2 "B0 and B1" and Section 6.2 "Retrospective out-of-sample comparison (Jan 7–29)": B0's γ | `main/b0_gamma_fixed.txt`, `main/b0_train_calibration_params.csv` |
| Figure 6.2 "Daily maximum drawdown on the test days" | `main/b{0,1,1plus,2,3}_test_results.csv`; the Hodges–Lehmann estimates from `main/paired_day_inference.csv` |
| Table 6.3 "Primary inference on the tail of daily drawdown" | `main/paired_day_inference_primary.csv` |
| Table A.1 "Size of the primary tests" | `inference_size_study/size_study_table.tex`, `size_study.csv`, `size_study_metadata.json` |
| Table A.2 "Secondary family of paired tests" | `main/paired_day_inference.csv` |
| Table A.3 "Daily maximum drawdown on the test days" | `main/b{0,1,1plus,2,3}_test_results.csv` |
| Appendix B "Hummingbot forward validation (paper-trading deployment test)": Table B.1 "Hummingbot forward validation per bot and day", Figure B.1 "Inventory per bot and prospective day" and the readings in the text | `forward_test/`: `appendix_table.tex` (Table B.1, from `qa_inventory.csv`, `qb_expected_fills.csv`, `qb_markouts.csv`, `qc_break_even.csv`, `qd_operations.csv` and `shadow_vs_paper.csv`), `fig_forward_inventory.pdf` and `.png` (Figure B.1), `pnl_split.csv` and `qb_fill_classes.csv` (the two exploratory readings) |
| Provenance of the profile stage | `profiling/profile_stage.sha256`, `profile_stage.commit`, `profile_stage.submitted`, `profiling/b3_dual/inputs*.sha256` |

The README's results table and `docs/figures/cvar_by_model.png` are generated from these files by
`scripts/analysis/make_readme_assets.py`.
