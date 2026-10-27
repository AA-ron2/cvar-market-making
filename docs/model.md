# Model: equations and code

What the code computes, and where. This describes the code of the thesis's final run (commit cf3096d, where the
package was called `procs`). [design_decisions.md](design_decisions.md) gives the reasons (the F- and D-numbers), and
`configs/final_pipeline.env` holds the frozen values. Functions are named rather than given line numbers.

## 1. Notation and units

| Symbol | Meaning | Units | Where |
|---|---|---|---|
| S_k | Mid = (best ask + best bid)/2 at snapshot k | USDT per DOGE | `cvar_mm/gym/data_loader.py` |
| Δt_k | Exchange-timestamp gap to the next snapshot | s | `data_loader.py` (raw µs / 10⁶) |
| q_k | Inventory; order size 1; q_max = 10 | DOGE | `cvar_mm/gym/trading_environment.py` |
| X_k | Cash | USDT | state vector |
| W_k = X_k + q_k S_k | Mark-to-market wealth | USDT | `cvar_mm/gym/sb3_wrapper.py`, metrics |
| T, τ = T − t | Day length (Σ Δt of the current day, refreshed on every reset), time to the end of the day | s | `trading_environment.py` |
| σ | Arithmetic-BM volatility of the mid | USDT/√s | `cvar_mm/gym/calibration.py` |
| A, κ | Execution intensity Λ(δ) = A·e^{−κδ} | 1/s; 1/USDT | `calibration.py` |
| γ | A–S risk aversion, a policy *control* for B1–B3 | 1/USDT | `cvar_mm/gym/formula_as.py` |
| φ | Running inventory penalty in the reward; 0 in replay training (D-10) | USDT/(DOGE²·s) | `cvar_mm/gym/experiment_config.py` |
| tick | 1 × 10⁻⁵ | USDT | `experiment_config.py` |

## 2. Data and clock

- Files: CSVs of Binance Spot DOGEUSDT order-book snapshots (the top 25 levels per side),
  `binance_book_snapshot_25_YYYY-MM-DD_DOGEUSDT.csv`, 2025-01-01 to 2025-01-29 ([data/README.md](../data/README.md)).
- One episode is one UTC day file; one environment step is one snapshot row.
- The clock is the exchange timestamp (F-01). On rows with Δt = 0 the mid is updated, there are no fills, and the
  action has no effect.

## 3. Calibration (`calibration.py`; settings in `configs/final_pipeline.env`)

- **σ̂:** within each day, 600 s blocks give √(ΣΔS² / ΣΔt); the median within the day, then the median across days.
- **Λ̂(δ_j):** for the depths δ_j ∈ {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20} ticks, exposure windows start every 15 s and
  are censored at 15 s. A *hit* is the best opposite price reaching mid ± δ_j (`best_price_reach`, a touch proxy: the
  best-price-reach execution proxy). Λ̂_j = hits / exposure seconds, pooled over bid and ask.
- **Fit:** weighted least squares of ln Λ̂_j = ln A − κ δ_j (`fit_exponential_intensity`).
- **Jan 1–6:** σ = 2.7568397609849377e-05, A = 0.10877437260020707, κ = 12066.410319238006 (log R² 0.989).
- **Jan 1–4** (B1_cand, the B2 candidates, their validation and the NB6 ablation): σ = 3.126206742299518e-05,
  A = 0.10894483870858916, κ = 12011.124427515675.
- `FirstPassageCalibrationConfig`'s defaults are these F-03 settings, and the pipeline passes every field from
  `configs/final_pipeline.env` (`scripts/pipeline/common.py`, `calibration_config_from_env`). The legacy wrappers
  `calibrate_from_arrays` and `calibrate_as_parameters`, which no pipeline job calls, keep their old settings
  explicitly.
- Every job that simulates records the calibration it used (`experiment_config` in its job metadata).
- `submit_final_pipeline.sh` sources `configs/final_pipeline.env` and stops if the caller's environment disagrees with
  any value in it.

## 4. Quote map (`formula_as.py`, `avellaneda_stoikov_quotes` and `map_formula_action`)

Action a ∈ [−1, 1]²:

$$\gamma = \exp\!\big(\ln\gamma_{\min} + \tfrac{1}{2}(a_1+1)(\ln\gamma_{\max}-\ln\gamma_{\min})\big),\qquad \text{skew} = 5\,a_2\cdot\text{tick}$$

$$r = S - q\gamma\sigma^2\tau,\qquad h = \tfrac{1}{2}\gamma\sigma^2\tau + \tfrac{1}{\gamma}\ln\!\Big(1+\tfrac{\gamma}{\kappa}\Big)$$

- bid = ⌊(r + skew − h)/tick⌋·tick and ask = ⌈(r + skew + h)/tick⌉·tick, with γ ∈ [0.01, 0.9].
- Then the clamp: bid ≤ S − ½tick and ask ≥ S + ½tick; if ask ≤ bid, ask = bid + tick.
- Depths: δ_b = S − bid, δ_a = ask − S (float32, the environment's action dtype).
- With κ ≈ 12,066, (1/γ)ln(1 + γ/κ) ≈ 1/κ ≈ 8.29 ticks for every admissible γ, so γ moves the quotes only through the
  inventory lean qγσ²τ and the time term.
- **B0** (`cvar_mm/gym/helpers/fast_rollout.py`) calls the same function with a fixed γ_B0 and skew = 0, clamp
  included. γ_B0 is tuned on the training days by `calibration.tune_gamma` with a seeded TPE sampler
  (γ_B0 = 0.8908584807525282). B0 executes with the environment's own fill model, so for the same γ, B0 and the
  B1–B3 environment give identical quotes, fills and metrics (`tests/test_b0_parity.py`).

## 5. Simulator step k (`trading_environment.py`)

1. Δt = Δt_{k+1}.
2. Fills: an independent Bernoulli per side, p = 1 − exp(−A e^{−κδ} Δt)
   (`cvar_mm/stochastic_processes/fill_probability_models.py`); at most one fill per side.
3. Fills that would push |q| above q_max are suppressed (`_enforce_inventory_limits`).
4. q ← q + 1_bid − 1_ask; X ← X + ask·1_ask − bid·1_bid.
5. Advance to S_{k+1} and mark W at S_{k+1}.

## 6. Observation (`formula_as.py`), normalized by VecNormalize (clip 10)

| Feature | Definition | Window |
|---|---|---|
| inventory_norm | clip(q/q_max, −1, 1) | current |
| time_remaining_frac | τ/T | current |
| log_mid_rel | ln(S/S₀), S₀ = the day's first mid | current |
| rolling_vol_ticks | √(ΣΔS²/ΣΔt) / tick | last 100 snapshots |
| momentum_ticks | (ΣΔS/ΣΔt) / tick | last 100 snapshots |
| spread_ticks | level-1 spread / tick | current snapshot |
| lob_imbalance | quantity imbalance, levels 1–5 | current snapshot |

The window is 100 *snapshots*, not seconds. Every feature uses past and current data only.

## 7. Policy and PPO settings (`experiment_config.py`, `scripts/pipeline/common.py`, `scripts/pipeline/train_formula_b*.py`)

- SB3 `PPO("MlpPolicy")` with SB3's default actor and critic networks (2 × 64, tanh): 9,541 parameters, as the
  published models confirm. The only `policy_kwargs` is `log_std_init` = log σ₀ (D-21).
- B1, B1_cand and the NB6 ablations: discount 0.999, GAE λ 0.95, clip 0.2, entropy coefficient 0.01, learning rate
  3e-4, 10 epochs, minibatch 512, n_steps 2,048, VecNormalize on observations and rewards; exactly one pass over the
  phase's days (`B1_PASSES = 1`, F-16).
- **Exploration noise (D-21).** Every PPO policy explores with a fixed σ₀ = `POLICY_STD` = 0.3 in [−1, 1] action units.
  `common.fixed_std_ppo` builds B1, B1_cand and the NB6 ablations with `log_std_init` = log σ₀ and
  `requires_grad=False` on log σ, so only the policy mean and the critic train and the entropy coefficient has no
  effect. In float32, log 0.3 is −1.2039728164672852, so σ = 0.2999999964.
- **Fine-tunes** (the B2 candidates and finals, B3 through the η profiler, B1⁺; D-15 to D-15c, D-23,
  `common.risk_aware_ppo_kwargs`): B1's discount, GAE λ, clip and entropy coefficient; learning rate
  `FINE_TUNE_LR` = 3e-3; per iteration 10 epochs of 32 minibatches (⌈n_steps·N/32⌉ transitions) with
  target_kl = 0.02, so at most 320 Adam steps (SB3 stops the remaining epochs once the approximate KL exceeds
  1.5 × target_kl).
- **Warm start** (`common.warm_start_from`), identical for all fine-tunes: B1's frozen VecNormalize with
  `training=False` and `norm_reward=True`, so they see B1's normalized rewards r/s (s = √(ret_rms.var + ε), recorded
  as `reward_scale_s`); only B1's policy and critic weights are copied, and Adam starts fresh (D-15c); log σ is frozen
  at B1's value σ₀, and a source with another σ is refused.
- **Fine-tune batches:** n_steps = one balanced full-day batch, `normalize_advantage=False` (the callbacks standardize
  per batch, §8); each iteration collects n_steps × N transitions from N = `TRAINING_FILL_PATHS` = 16 fill paths per
  day, sharing the replayed mid with independent fill draws.
- **Update budget:** B2 (all phases), B3 and B1⁺ each run exactly k = 10 PPO iterations, fixed a priori (D-19).
- **B1_cand** trains on Jan 1–4 with the Jan 1–4 calibration; the B2 candidates start from it. The final B2, B3 and
  B1⁺ start from the final B1 (D-11).
- **B1⁺ (D-18)** runs the final-B2 code at α = 0 (`train_formula_b2.py --phase final --control b1plus`), with the
  final B2's warm start, N, k, seed and PPO settings. α = 0 is never a B2 candidate. With equal seeds and λ ≡ 0, B1⁺
  and B3 coincide bit for bit.

## 8. Objectives

**Reward (B1–B3):** r_k = ΔW_k − φ·q²_{k+1}·Δt_k (`cvar_mm/rewards/reward_funcs.py`) with φ = 0 (D-10), so r_k = ΔW_k
exactly (`tests/test_reward_phi.py`). The φ = 0.01 in `BMExperimentConfig` belongs to the Brownian-motion validation
only.

**B1:** max_θ J(θ) = E[Σ_k β^k r_k].

**B2** (`cvar_mm/gym/drawdown_objectives.py`): max_θ J(θ) − α·E[Z], with the full-day maximum drawdown

$$Z = \max_k\big(\max_{u\le k} W_u - W_k\big),$$

read at the terminal step before the auto-reset. Z̄_{−i} is the mean Z of the other fill paths of the same replay day,
the leave-one-out baseline (D-12).

**Advantages in B1's units** (D-15a, `drawdown_objectives.standardized_risk_advantages`, shared by B2 and B3). Let A_t
be SB3's GAE advantages of B1's normalized rewards r/s, and μ_A, σ_A their mean and population standard deviation over
the whole batch. PPO optimizes

$$\hat A_t = \frac{A_t - \mu_A}{\sigma_A} - \frac{w\,(c_i - \bar c_{-i})}{s\,\sigma_A}$$

for every step t of complete trajectory i, with w = α and c = Z for B2.
- Every policy step follows the gradient of E[P&L] − w·E[c], scaled by 1/(s·σ_A): the risk-to-P&L weight is exactly w,
  in USDT per USDT, and Â does not depend on the reward unit.
- At w = 0 (B1⁺, or B3 while λ = 0) an update is a B1-style update of the mean and the critic.
- The critic's targets are the GAE returns of r/s, fixed before the callback edits the advantages, so the critic stays
  P&L-only, in B1's units.

**B3** (`cvar_mm/gym/cvar_lagrangian.py`):

$$\max_\theta\ \min_{\lambda\ge 0}\ J(\theta) - \lambda\big(H_\rho(\theta,\nu) - d\big),\qquad H_\rho = \nu + \tfrac{1}{\rho}\,\mathbb{E}\big[(Z-\nu)^+\big],\quad \rho = 0.2$$

- **Score term:** the formula above with w = λ (the λ before this batch's dual step) and c_i = (Z_i − ν)⁺/ρ, with the
  same per-day leave-one-out baseline and helper as B2.
- **ν (D-12):** the batch's empirical (1 − ρ)-quantile of Z (`empirical_ru_cvar`, method "higher"), used in the score
  term and in Ĥ = ν + mean((Z − ν)⁺)/ρ.
- **λ (D-12):** after each batch, λ ← clip(λ + η_λ(Ĥ − d)/d, 0, λ_max), with λ₀ = 0 and λ_max = 500. η_λ =
  `B3_LAMBDA_LR` = 8.0, selected on Jan 1–6 by the D-12a rule (`scripts/pipeline/profile_b3_dual.py`): 10 B3 updates
  for each η ∈ {0.5, 2, 8}; an η is excluded if λ reaches λ_max or Ĥ_u − d changes sign more than twice; among the
  rest, the smallest mean |Ĥ_u − d|/d over updates 8–10 wins, ties going to the smaller η.
- **Paths (D-12):** N = 16 fill paths per day for B2 and B3 alike, so a batch holds 96 complete trajectories on
  Jan 1–6.
- **Unbiasedness:** the baseline excludes path i, so the B2 score estimator is unbiased, and so is B3's at a fixed ν.
  The applied term is divided by s·σ_A, with σ_A estimated on the same batch, so the update is unbiased up to that
  shared positive batch-level scale (an O(1/n) effect, as with any advantage normalization); the ratio w between the
  risk and P&L terms is unaffected. With ν set to the same batch's quantile, the B3 estimator is the plug-in (Danskin)
  estimator: consistent, with an O(n^-1/2) finite-batch bias (Tamar, Glassner & Mannor 2015).
- **History:** `b3_cvar_lagrangian_history.csv` has one row per update with the λ used, λ after the update, ν, Ĥ,
  Ĥ − d, the mean Z and the diagnostics below.
- **Diagnostics (B2 and B3 alike, `drawdown_objectives.UPDATE_DIAGNOSTICS`):** `reward_scale_s`; `advantage_mean` and
  `advantage_std`; `reward_clip_fraction` (expected 0); `adam_steps`, `minibatches_per_epoch`, `epochs_completed` and
  `kl_stopped`; SB3's `approx_kl`, `clip_fraction`, `value_loss` and `explained_variance`; `mean_log_std` (constant);
  `adam_damping_ratio`; `action_shift_vs_b1` per action dimension (the mean |clip(μ_now) − clip(μ_B1)| on a fixed sample
  of first-rollout observations; it draws from no generator training uses); and the cumulative `elapsed_seconds`.
- **Training feasibility:** `b3_training_feasibility.json` compares the pooled CVaR₀.₂ of Z over the last
  min(5, k − 1) rollouts with d, and reports the mean per-batch Ĥ. Rollout 1 is the warm-start B1 and is excluded.
  The per-path costs are in `b3_training_costs.csv`.
- **Threshold d (F-06, D-17; `scripts/pipeline/calibrate_cvar_b3.py`).** d = 0.8 × the empirical CVaR₀.₂ of the final
  B1's full-day MaxDD on Jan 1–6 from **stochastic** rollouts with B1's σ (the sampling that produces Ĥ): fill seeds 42
  and 43 × 20 paths per day, 240 in all, with torch seeded per (day, fill seed). A second, deterministic pass is
  recorded (`cvar_deterministic`), not used. `cvar_threshold_b3.json` holds the estimator, d, both CVaRs and ν, B1's σ,
  `b1_pinned_share` (the share of B1's deterministic decisions at an action bound), the seeds and every path's Z.
- **Nesting:** B3 with ρ = 1, ν = 0 and λ ≡ α equals B2 exactly (D-11); both pass the same A, s and coefficients
  through the same helper (`tests/test_full_day_ru_cvar.py`).

## 9. Metrics (B1–B3: `cvar_mm/gym/helpers/generate_trajectory_stats.py`; B0: `fast_rollout.py`)

The two implementations use the same definitions and give identical values for the same γ and fill seed.

| Metric | Definition |
|---|---|
| Final P&L | W_T − W_0 (USDT) |
| Sharpe | mean / population std of the wealth changes on a fixed 60 s clock; not annualized |
| Sortino | Falces Marín et al. (2022), eq. (20) (D-13): the mean of all 60 s wealth changes divided by the population std of the negative changes about their own mean; 0 with fewer than two negative changes (`tests/test_sortino_definition.py`) |
| Max DD | Z on the full snapshot path (USDT) |
| P&L-to-MAP | Final P&L / time-weighted mean \|q\| |
| Mean \|q\|, near-cap fraction | time-weighted; near cap = \|q\| ≥ 0.8 q_max |

- **Per-day value:** the mean over rollouts × fill seeds (`scripts/pipeline/evaluate_formula_model.py`), with
  deterministic (mean) actions for B1, B1⁺, B2 and B3 (F-17).
- **Per-path values:** one row per full-day path (day × fill seed × rollout) in `b{0,1,2,3}_test_episodes.csv` and
  `b1plus_test_episodes.csv`.
- **CVaR₀.₂ of the daily MaxDD:** reported per model in `overall_summary_table.csv` over the 23 seed-averaged days
  ("Max DD CVaR0.2 (days)") and over all full-day paths ("Max DD CVaR0.2 (episodes)"), with `empirical_ru_cvar`.
- `comparison_table.csv` (days best) counts B0–B3 only, so B1⁺ cannot change their counts.

## 10. Inference (`cvar_mm/gym/statistical_inference.py`, `scripts/pipeline/run_paired_inference.py`)

Two pre-specified families, each with its own Holm correction (D-14, D-14a), and descriptive rows outside both (D-16,
D-18). The trading day is the unit of inference. A contrast is (reference, challenger), and its effect is challenger −
reference.

**Statistic (D-14).** Δ̂ = CVaR₀.₂ of the challenger's 23 seed-averaged daily MaxDD values minus the reference's,
computed with `empirical_ru_cvar` on the sorted values.

**`paired_day_inference_primary.csv`.**
- **`primary`:** {B3 vs B1, B3 vs B2}.
  - The within-day label-swap randomization test (`paired_randomization_test_cvar`, D-14a): each of B = 10,000 draws
    (numpy `default_rng(42)`) gives every day an independent fair coin; on the marked days the two models' values
    swap, and Δ*_b is recomputed. The two-sided p = (1 + #{|Δ*_b| ≥ |Δ̂|})/(B + 1).
  - H0: within each day the two models' outcomes are exchangeable. A rejection with Δ̂ < 0 is evidence that B3's daily
    drawdown distribution differs from the reference's, with a lower upper-tail mean; it is not a test of equal CVaRs
    alone.
  - Holm over the two primary p-values (`holm_p`).
  - Interval: the paired circular block bootstrap (block 3, B = 10,000, centred), descriptive and labelled unadjusted,
    reported with its simulated coverage at n = 23.
- **`sensitivity`** (no Holm): the bootstrap interval at blocks 1, 2 and 4, and the path-level CVaR over all full-day
  paths (920 per model), tested by the same randomization test with each day's paths swapped together.
- **`descriptive`** (no Holm, not multiplicity-adjusted): B3 vs B0 (D-16), and B3 vs B1⁺ and B1⁺ vs B1 (D-18).

**`paired_day_inference.csv`.**
- **`secondary`:** (B1, B0), (B1, B2), (B1, B3) and (B2, B3) × 7 metrics. Each row has the Pratt–Wilcoxon p of the
  paired day differences, with Holm over the 28 rows; the Hodges–Lehmann effect with its exact signed-rank 95 %
  interval (at n = 23, [W(74), W(203)] of the 276 Walsh averages, exact coverage 0.9516); and the circular
  block-bootstrap interval of the mean difference (block 5).
- **`descriptive`** (no Holm): B3 vs B1⁺ and B1⁺ vs B1 on the same metrics.

**Size study (D-14a)**, `scripts/analysis/cvar_test_size_study.py`: a NumPy simulation with no market data that runs the
two production tests on eight null designs (cross-day AR(1) φ ∈ {0, 0.5} × within-day correlation r ∈ {0.5, 0.9} ×
lognormal σ ∈ {0.5, 1.0}), with n = 23, ρ = 0.2, 2,000 replications per design and 2,000 randomization and 2,000
bootstrap draws (block 3) per replication. Randomization rejection rates were 0.023–0.053, bootstrap rejection rates
0.095–0.126 and bootstrap interval coverage 0.874–0.905 (`results/final/inference_size_study/`).

## 11. Known gaps

**Simplifications of the simulator,** stated in the thesis:
- no fees;
- no adverse selection: fills are independent of S_{k+1} − S_k;
- no queue position and no latency; the fill model is calibrated on a touch proxy (`best_price_reach`), not on
  observed queue-aware fills;
- no terminal liquidation cost;
- unit order size, with P&L linear in size.

**Caveats of the estimators:**
- *Action clipping.* Sampled actions are clipped to ±1 before execution, so a mean beyond ±1 pins the executed action
  and removes training noise without changing the deterministic evaluation action. It is measured, not corrected:
  `b1_pinned_share` in `cvar_threshold_b3.json` and `action_shift_vs_b1` per update.
- *Finite samples of d and Ĥ.* The plug-in CVaR (`empirical_ru_cvar`) is the minimum over ν of the empirical RU
  function, so it is biased low by O(1/n). Ĥ₁ uses 96 paths and d uses 240, so Ĥ₁ ≈ 1.25·d holds in expectation only up
  to that bias difference. d's 240 paths are 40 seeded streams replayed on six days, so their effective size is
  somewhat below 240.
- *Power.* With 23 test days and ρ = 0.2, the day-level CVaR rests on about 4.6 days.

**Issues found in the review and fixed before the final run** (the corrected pipeline produced every published
result):

| Issue | Fix |
|---|---|
| φ = 0.01 made the inventory penalty dominate P&L | replay φ = 0 (D-10) |
| B2 and B3 defaulted to one PPO iteration, and B3's only iteration had λ = 0 | k = 10 iterations, fixed a priori (D-11, D-19) |
| B2 trained from scratch while B3 started from B1 | B1_cand; B2 starts from B1_cand or B1 with the frozen normalizer; equal k (D-11) |
| B3's ν and λ steps were on the wrong scale, with 6 values of Z per update | batch-quantile ν, normalized λ step, 16 fill paths, per-day leave-one-out, η profiling (D-12, D-12a) |
| B0's quotes lacked the mid clamp | one shared quote function; B0 uses the environment's fill model; parity tested |
| No CVaR₀.₂ evaluation metric, and no pre-specified inference family | CVaR₀.₂ at day and path level, B3's training feasibility, the D-14 families with the D-14a test |
| The frozen calibration lived only in the shell environment | the submit script sources and enforces `configs/final_pipeline.env`; two stages |
| `with_suffix(".zip")` broke model hashes; the fill-seed default was "42" | explicit `<stem>.zip`; seeds "42 43"; data tests skip without `DATA_DIR` |
| τ used day 1's length in multi-day training; the critic's reward scale changed at the warm start; Optuna was unseeded | T and n_steps refreshed per reset; seeded TPE; B2 and B3 train in B1's units (D-15a) |
| B1's entropy weight on raw-reward advantages made log σ overflow within the first fine-tune iteration | bounded schedule, B1's units, frozen log σ, fresh Adam (D-15 to D-15c) |
