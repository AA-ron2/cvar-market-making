# Design decisions

The methodological choices behind the thesis results, condensed from the thesis's decision log. **F-01 to F-23** are
the frozen rules that the final run follows. **D-10 to D-24** record how they were reached in the review of the pipeline
before the final run. Each D-decision was taken before any output for the test days (2025-01-07 to 2025-01-29)
existed. Every entry gives the decision, the reason and the date.

The frozen values are in `configs/final_pipeline.env`, and [model.md](model.md) gives the equations. Module paths use
the public names (`cvar_mm`, `scripts/pipeline`); the thesis run used `procs` and `scripts/snellius`
([reproduce.md](reproduce.md)).

## Frozen rules

### F-01 · Exchange timestamps are the clock

- **Decision.** The exchange `timestamp` (int64 µs, read as exact text) is the clock. Out-of-order rows are
  stable-sorted as whole rows, and only when needed. Duplicate timestamps are kept, and Δt comes from the raw
  microseconds.
- **Reason.** Parsing through float64 or using the local receipt time distorts Δt, which drives the fills and the
  volatility. Three January days hold real inversions (up to 21 ms); all 29 days pass the loader's validation.
- **Date.** Recorded 2026-10-05 (the rule predates the log).

### F-02 · Data split

- **Decision.** Jan 1–4 candidate training, Jan 5–6 validation, Jan 1–6 final refit, Jan 7–29 retrospective
  out-of-sample test (23 days).
- **Reason.** Every model choice uses training or validation days only, so the test days evaluate frozen procedures.
  Earlier, pre-correction versions of the pipeline were evaluated on Jan 7–29, so the thesis calls them retrospective
  out-of-sample, not an untouched hold-out.
- **Date.** Recorded 2026-10-05.

### F-03 · Calibration of the execution intensity and the volatility

- **Decision.** The `best_price_reach` touch proxy; exposure windows on a fixed 15 s stride, censored at 15 s; depths
  {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20} ticks; at least 10 hits per depth; σ from 600 s blocks. On Jan 1–6:
  σ = 2.7568 × 10⁻⁵ USDT/√s, A = 0.10877 s⁻¹, κ = 12,066 per USDT (log R² 0.989).
- **Reason.** Chosen among four execution-proxy and start-rule candidates on Jan 1–4 only
  (`scripts/pipeline/profile_formula_calibration.py`), never on P&L or later days. `best_price_reach` is a touch
  proxy (the best-price-reach execution proxy), not an observed queue-aware fill.
- **Date.** Recorded 2026-10-05.

### F-04 · Action space

- **Decision.** The policy's two actions set γ ∈ [0.01, 0.9] (log-linear) and a quote skew in [−5, +5] ticks; the
  Avellaneda–Stoikov formula builds the quotes, with the mid clamp.
- **Reason.** Every model quotes within one family, so B0's policy lies inside the PPO action space and the
  comparison isolates how γ and the skew are chosen.
- **Date.** Recorded 2026-10-05.

### F-05 · B2 selection

- **Decision.** α ∈ {0.1, 1, 10}. The candidates train on Jan 1–4 and are validated on Jan 5–6, with the fill seeds
  averaged within each day. A candidate is feasible if its mean validation Sharpe is above 0; the feasible ones rank
  by (MaxDD, −Sharpe, α), with (−Sharpe, MaxDD, α) as the fallback when none is feasible. The selected α's final B2
  trains on Jan 1–6.
- **Reason.** The penalty weight is chosen on validation days only, drawdown first, as the penalty intends.
- **Date.** Recorded 2026-10-05.

### F-06 · B3's constraint

- **Decision.** B3 constrains the Rockafellar–Uryasev CVaR at tail mass ρ = 0.2 of the full-day maximum drawdown Z,
  with d = 0.8 × B1's training CVaR₀.₂ from stochastic B1 rollouts (D-17). B3 starts from B1 with B1's frozen
  normalizer.
- **Reason.** The threshold is a 20 % cut from B1's own tail risk on the training days, measured with the sampling
  that B3 trains on, and uses training-only information.
- **Date.** Recorded 2026-10-05; the estimator of d is D-17 (2026-10-06).

### F-07 · Evaluation

- **Decision.** Fill seeds 42 and 43 × 20 rollouts per test day, averaged within the day; Sharpe and Sortino on 60 s
  wealth changes; MaxDD on the full path; the trading day is the unit of inference.
- **Reason.** The snapshots within a day are not independent samples; the day is the replicate.
- **Date.** Recorded 2026-10-05.

### F-08 · Inference tools

- **Decision.** Paired day differences, Pratt–Wilcoxon tests, the circular block bootstrap and Holm corrections.
- **Reason.** Every model runs on the same replayed days and fill seeds, so the comparisons are paired, and n = 23
  calls for distribution-free tests. The primary test was later replaced by a randomization test (F-13).
- **Date.** Recorded 2026-10-05.

### F-09 · Reward: mark-to-market P&L

- **Decision.** B1–B3 train on ΔW: the running inventory penalty φ is 0 (D-10).
- **Reason.** φ = 0.01 made the penalty dominate P&L by orders of magnitude; risk enters only through B2 and B3.
- **Date.** 2026-10-05.

### F-10 · Shared warm start and equal budget

- **Decision.** The B2 candidates start from B1_cand (Jan 1–4); the final B2 and B3 start from the final B1
  (Jan 1–6); B2 and B3 get the same k = 10 risk-aware PPO iterations (D-11, D-19).
- **Reason.** A shared start and budget let the B3-vs-B2 contrast identify the effect of the objective.
- **Date.** 2026-10-05; k fixed on 2026-10-07.

### F-11 · B3's optimizer

- **Decision.** ν is the batch's (1 − ρ)-quantile of Z; λ ← clip(λ + η_λ(Ĥ − d)/d, 0, 500), λ₀ = 0; N fill paths per
  training day with a per-day leave-one-out baseline in B2 and B3 alike; η_λ by the training-only rule D-12a (D-12,
  D-12a).
- **Reason.** The exact inner minimization and a dimensionless dual step that can act within the budget.
- **Date.** 2026-10-05; the η rule 2026-10-06.

### F-12 · Sortino

- **Decision.** The reported Sortino is Falces Marín et al. (2022), eq. (20), with the definition stated (D-13).
- **Reason.** Comparability with that study.
- **Date.** 2026-10-05.

### F-13 · Primary inference

- **Decision.** Δ = CVaR₀.₂ of the seed-averaged daily MaxDD for {B3 vs B1, B3 vs B2}, tested by a within-day
  label-swap randomization test (10,000 draws, seed 42) with Holm over the two; the paired circular block-bootstrap
  interval (block 3) is descriptive and labelled unadjusted; the secondary metrics form a separate Holm family (D-14,
  D-14a).
- **Reason.** The research question is about tail drawdown risk, and the randomization test holds its size at n = 23.
- **Date.** 2026-10-05; the test 2026-10-06.

### F-14 · How B2 and B3 fine-tune from B1

- **Decision.** B1's frozen reward normalizer, batch-standardized P&L advantages with the risk term in the same units,
  fresh Adam moments, no critic rescale, log σ frozen at B1's value; 10 epochs of 32 minibatches with
  target_kl = 0.02; identical for B2, B3, the η profiler and B1⁺; learning rate 3e-3 (D-15 to D-15c, D-23).
- **Reason.** Each update follows E[P&L] − w·E[cost] with the weight w in USDT per USDT, and the policy can move within
  k = 10 bounded updates.
- **Date.** 2026-10-06; the learning rate 2026-10-14.

### F-15 · B0 versus B3

- **Decision.** B3 vs B0 on the primary statistic is a descriptive row, outside both Holm families (D-16).
- **Reason.** No research question requires it, and no claim rests on it.
- **Date.** 2026-10-06.

### F-16 · One pass for B1 and B1_cand

- **Decision.** B1 and B1_cand each train exactly one pass over their days (`B1_PASSES = 1`); the pipeline does not
  use a timestep budget.
- **Reason.** A final B1 trained longer than B1_cand would make the B2 candidates start from a less-trained policy
  than the final B2, so the selection would compare candidates that do not represent it.
- **Date.** 2026-10-06.

### F-17 · Deterministic evaluation, stochastic training statistics

- **Decision.** The evaluation uses the mean action for B1, B1⁺, B2 and B3; d and B3's Ĥ use stochastic actions with
  B1's σ.
- **Reason.** The thesis evaluates the policies' mean actions, while the training-side risk statistics measure the
  sampling that training acts on (D-15b, D-17).
- **Date.** 2026-10-06.

### F-18 · The B1⁺ control

- **Decision.** B1⁺ (B1 plus k iterations at α = 0) is evaluated and reported in descriptive rows only (D-18).
- **Reason.** It separates the extra fine-tuning from the risk term.
- **Date.** 2026-10-06.

### F-19 · k = 10 a priori; two stages

- **Decision.** k = 10, fixed a priori. Every training- and validation-side job runs in `PIPELINE_STAGE=profile`; B0
  and every Jan 7–29 evaluation, aggregation and inference run in `PIPELINE_STAGE=final`, after the author's go/no-go
  review. The Slurm time limits are explicit values in `configs/final_pipeline.env` (D-19).
- **Reason.** No test-day output exists before the review, and the compute plan does not depend on timings.
- **Date.** 2026-10-07.

### F-20 · The final B3 is the selected η run

- **Decision.** The η-profiling run that D-12a selects is promoted by copy, with its SHA-256 recorded, to the final B3;
  the final stage evaluates B3 and trains nothing (D-20).
- **Reason.** Every η run already has the final B3's configuration.
- **Date.** 2026-10-08.

### F-21 · Fixed exploration noise

- **Decision.** Every PPO policy explores with a fixed σ₀ = 0.3 in [−1, 1] action units (`POLICY_STD`); B2, B3, the η
  profiler and B1⁺ inherit it, and d is computed with it (D-21).
- **Reason.** A learned σ made B1's stochastic policy nearly bang-bang.
- **Date.** 2026-10-13.

### F-22 · Parallel final B2s

- **Decision.** Each α's final B2 trains on Jan 1–6 in parallel with the candidates; the selected one is promoted by
  copy with its SHA-256 (D-22).
- **Reason.** Equivalent to the sequential refit, and faster.
- **Date.** 2026-10-13.

### F-23 · Fine-tune learning rate

- **Decision.** Every fine-tune (the B2 candidates and finals, B3 through the η profiler, B1⁺) trains at 3e-3
  (`FINE_TUNE_LR`); B1, B1_cand and the NB6 ablations keep 3e-4 (D-23).
- **Reason.** At 3e-4 the fine-tunes did not move within k = 10 updates.
- **Date.** 2026-10-14.

## Decisions taken in the review (D-10 to D-24)

### D-10 · Running inventory penalty φ = 0

- **Decision.** B1–B3 train on mark-to-market P&L (φ = 0). The φ = 0.01 of the Brownian-motion validation stage stays
  there.
- **Reason.** φ = 0.01 came from the Brownian-motion setup (prices near 100) and is about seven orders of magnitude too
  large here: its daily penalty exceeds the maximal expected spread revenue (about 0.57 USDT per day) once the
  time-average q² passes 6.6 × 10⁻⁴, so the best policy stays flat. The A–S-consistent ½γσ² is below 0.2 % of daily
  P&L at this calibration, so φ = 0 and risk enters only through B2 and B3. In the A–S formula γ is a control, not
  the agent's preference.
- **Date.** 2026-10-05.

### D-11 · Shared initialization and equal update budget for B2 and B3

- **Decision.** The B2 candidates start from B1_cand, trained on Jan 1–4 with the Jan 1–4 calibration; the final B2
  and B3 start from the final B1 with its frozen normalizer and run the same number k of risk-aware iterations.
  Amendment (2026-10-06): B1 and B1_cand train exactly one pass (F-16).
- **Reason.** B2 is the ρ = 1, fixed-multiplier member of the B3 family: at ρ = 1, ν = 0 and λ = α their advantages
  are identical. With a shared start and budget, the B3-vs-B2 difference identifies the effect of the objective,
  conditional on the B1 checkpoint. Starting the candidates from the final B1 would leak Jan 5–6 into their selection.
- **Date.** 2026-10-05.

### D-12 · B3's inner (ν) and dual (λ) updates and sample size

- **Decision.** ν = the batch's empirical (1 − ρ)-quantile of Z; λ ← clip(λ + η_λ(Ĥ − d)/d, 0, λ_max) with λ₀ = 0 and
  λ_max = 500; N = 16 fill paths per training day; a per-day leave-one-out baseline, in B2 and B3 alike.
- **Reason.** The quantile is the exact inner minimization of the RU function (Rockafellar & Uryasev 2000); the raw
  dual step could not reach a useful λ within any feasible number of updates; 16 paths give about 19 values of Z in the
  tail per batch instead of about 1. With ν taken from the same batch, the B3 score estimator is the plug-in (Danskin)
  estimator: consistent, with an O(n^-1/2) finite-batch bias (Tamar, Glassner & Mannor 2015).
- **Date.** 2026-10-05.

### D-12a · Training-only rule for η_λ

- **Decision.** Grid η ∈ {0.5, 2, 8}; for each, 10 B3 updates on Jan 1–6 (seed 42). An η is excluded if λ reaches
  λ_max or Ĥ − d changes sign more than twice; among the rest, the smallest mean |Ĥ_u − d|/d over updates 8–10 wins,
  ties going to the smaller η. If every η is excluded, nothing is selected and there is no fallback.
- **Reason.** D-12's first rule (5 updates, the largest η without a λ_max hit) passed every η by construction, because
  λ cannot grow far in 5 steps. The new rule picks the step size that brings the training-time constraint statistic
  closest to its budget by the end of the profiled horizon, from training days only.
- **Date.** 2026-10-06.

### D-13 · Sortino definition

- **Decision.** Report Falces Marín et al. (2022), eq. (20): the mean of the 60 s wealth changes divided by the
  population standard deviation of the negative changes about their own mean.
- **Reason.** Comparability with that study. It is not the Sortino & Price (1994) downside deviation, and Methods
  states the difference.
- **Date.** 2026-10-05.

### D-14 · Primary inference on CVaR₀.₂ of the daily MaxDD

- **Decision.** Primary statistic Δ = CVaR₀.₂ of the 23 seed-averaged daily MaxDD values, challenger minus reference,
  for {B3 vs B1, B3 vs B2}, with Holm over the two. Sensitivity rows: bootstrap blocks 1, 2 and 4, and the
  path-level CVaR over all full-day paths. Secondary family: the seven other metrics for (B1, B0), (B1, B2), (B1, B3)
  and (B2, B3), with Pratt–Wilcoxon p-values, Hodges–Lehmann effects and block-5 bootstrap intervals of the mean, and
  their own Holm over 28 rows.
- **Reason.** The research question concerns tail drawdown risk, and Wilcoxon or Hodges–Lehmann do not apply to a
  difference of tail functionals. With 23 days the day-level CVaR rests on about 4.6 days, so power is low; this is
  a stated limitation.
- **Date.** 2026-10-05.

### D-14a · A randomization test as the primary test

- **Decision.** The primary p-value comes from a within-day label-swap randomization test (10,000 draws, seed 42,
  two-sided, with the +1 correction). The paired circular block-bootstrap interval (block 3) stays as a descriptive,
  unadjusted interval, reported with its simulated coverage. A size study runs both tests on eight null designs.
- **Reason.** In simulated null designs at n = 23 the centred bootstrap test rejected about 11 % of the time at a
  nominal 5 %, while the randomization test is exact under within-day exchangeability. The size study
  (`scripts/analysis/cvar_test_size_study.py`, `results/final/inference_size_study/`) gave randomization rejection
  rates of 0.023–0.053 (all ≤ 0.06), bootstrap rejection rates of 0.095–0.126 and bootstrap interval coverage of
  0.874–0.905. A rejection with Δ̂ < 0 is evidence that B3's daily drawdown distribution differs from the reference's,
  with a lower upper-tail mean, not a test of equal CVaRs alone.
- **Date.** 2026-10-06.

### D-15 · How B2 and B3 fine-tune from B1: the update schedule

- **Decision.** Each B2 and B3 iteration runs 10 epochs of 32 minibatches with target_kl = 0.02, so at most 320 Adam
  steps; the same for B2, B3, the η profiler and B1⁺. (The entropy part of D-15 was withdrawn the same day in favour of
  D-15a and D-15b.)
- **Reason.** With the default minibatch of 512 transitions, one iteration was about 1.4 × 10⁶ Adam steps, about 15
  times B1's whole optimization, and B1's entropy weight on raw-reward advantages drove log σ to overflow inside the
  first update.
- **Date.** 2026-10-06.

### D-15a · Fine-tuning in B1's units

- **Decision.** B2 and B3 train on B1's normalized rewards (its frozen normalizer, `norm_reward=True`), with
  advantages Â = (A − μ_A)/σ_A − w·(c_i − c̄_{−i})/(s·σ_A), standardized once per batch; the critic stays in B1's units.
- **Reason.** The risk-to-P&L weight is then exactly w, in USDT per USDT, with no new hyperparameter (s and σ_A are
  measured). Raw-unit advantages with B1's optimizer state left the policy at B1 in a check.
- **Date.** 2026-10-06.

### D-15b · Freeze log σ during fine-tuning

- **Decision.** In B2, B3, the η profiler and B1⁺, log σ is frozen at B1's value; only the policy mean and the critic
  train.
- **Reason.** Otherwise B2 and B3 could lower their training-time drawdown by shrinking the exploration noise, which
  costs no P&L and does not exist in the deterministic evaluation.
- **Date.** 2026-10-06.

### D-15c · Fresh Adam moments at the warm start

- **Decision.** The warm start copies B1's policy and critic weights only; the optimizer starts fresh.
- **Reason.** B1's second moments were built on minibatches about 4,400 times smaller than a fine-tune's, which would
  damp each step by up to 1/66 for most of the k iterations.
- **Date.** 2026-10-06.

### D-16 · B0 versus B3 as a descriptive row

- **Decision.** Report B3 vs B0 on the primary statistic, with the randomization p and the bootstrap interval, as a
  descriptive row outside both Holm families.
- **Reason.** No research question requires it, but the comparison with the analytical baseline is expected; no claim
  of superiority rests on it.
- **Date.** 2026-10-06.

### D-17 · The estimator of B3's threshold d

- **Decision.** d = 0.8 × CVaR₀.₂ of the final B1's full-day MaxDD from stochastic rollouts with B1's σ: fill seeds 42
  and 43 × 20 rollouts per seed on each of Jan 1–6, with torch seeded per (day, fill seed). The deterministic CVaR is
  recorded, not used.
- **Reason.** B3's Ĥ is computed on stochastic training rollouts, so d has to measure the same sampling for 0.8 to mean
  a 20 % cut from B3's own starting point. The plug-in CVaR is biased low by O(1/n), so E[Ĥ₁] sits slightly below
  1.25·E[d]; this is reported, not corrected.
- **Date.** 2026-10-06.

### D-18 · The B1⁺ control: B1 fine-tuned without a risk term

- **Decision.** B1⁺ is the final B1 fine-tuned for the same k iterations through the B2 code path at α = 0, evaluated
  on Jan 7–29 like the other models and reported as descriptive rows (B3 vs B1⁺, B1⁺ vs B1), outside both Holm
  families. It never enters the B2 selection.
- **Reason.** It separates the effect of the extra fine-tuning from that of the risk term in B3 vs B1. D-24 qualifies
  it: B1⁺ matches B2 and B3 in procedure, k and learning rate, but not in accepted Adam steps.
- **Date.** 2026-10-06.

### D-19 · k = 10 fixed a priori; training before the test stage

- **Decision.** k = 10 for B2, B3 and B1⁺, frozen. `PIPELINE_STAGE=profile` runs every training- and validation-side
  job; `PIPELINE_STAGE=final` runs B0 and every Jan 7–29 evaluation, comparison and inference after the author's
  go/no-go review. The final stage refuses a change of the training code or of a frozen value since the profile
  stage (only η and the time limits may change). Since D-20 the final stage trains nothing.
- **Reason.** Sizing k from timings made every B2 job wait for the profiling; fixing it a priori removes that wait
  and keeps every test-day output behind the review.
- **Date.** 2026-10-07.

### D-20 · The final B3 is the selected η-profiling run

- **Decision.** Each η-profiling run is saved as a B3; the run that D-12a selects is promoted by copy, with its
  SHA-256 recorded in the profile record, to the final B3.
- **Reason.** Every η run already has the final B3's configuration (the final B1 and its normalizer, d, N = 16,
  k = 10, seed 42, Jan 1–6), so retraining would repeat one of them (20–35 h); the evaluated B3 is the run whose
  criterion was computed.
- **Date.** 2026-10-08.

### D-21 · Fixed exploration σ₀ = 0.3 for every PPO policy

- **Decision.** B1, B1_cand and the NB6 ablations train with log σ fixed at log 0.3 in action units; B2, B3, the η
  profiler and B1⁺ inherit it, and d uses it. σ₀ was fixed a priori.
- **Reason.** In the 2026-10-12 profile stage B1 learned σ = [6.50, 2.09] with actions clipped at ±1: about 88 % (γ) and
  63 % (skew) of the sampled actions were clipped, so the stochastic policy was nearly bang-bang and d, B2's penalty and
  B3's constraint acted mostly on noise. That profile stage was rerun.
- **Date.** 2026-10-13.

### D-22 · Parallel final B2s, the selected one promoted by copy

- **Decision.** Each α's final B2 trains on Jan 1–6 from the final B1 with the same seed, inputs and k, in parallel
  with the candidate sweep; the selected α's model and normalizer are copied to the final B2 names with their SHA-256.
- **Reason.** Equivalent to the sequential refit (tested) and about 13 h faster; the unchosen finals are never
  evaluated.
- **Date.** 2026-10-13.

### D-23 · Fine-tune learning rate 3e-3

- **Decision.** B2, B3 (through the η profiler) and B1⁺ train at a learning rate of 3e-3, ten times B1's 3e-4; every
  other setting is unchanged.
- **Reason.** At 3e-4 the fine-tunes did not move (approx_kl about 1e-3 per update, 5 % of target_kl, no KL stop), so
  go/no-go check 2b was not met: the fine-tunes must move, so that Ĥ responds to λ. At 3e-3 the trust region bounds
  each update (about 0.07 action units). That profile stage was rerun.
- **Date.** 2026-10-14.

### D-24 · The third profile stage is final; B3's non-attainment is a finding

- **Decision.** B3 (η = 8.0) did not reach d within k = 10 updates (Ĥ/d between 1.21 and 1.36 while λ rose to 21.9), so
  go/no-go check 2b was not met. The review found no defect (crash, code bug or broken critic), so the final stage
  ran with no further training iteration.
- **Reason.** Changing the training after seeing this would tune on the outcome. The test days evaluate the procedure
  as specified, not a policy that was feasible on its training days. B1⁺ is matched in procedure but not in accepted
  Adam steps: each of its 10 updates stopped on the KL limit in epoch 0. D-24 records a finding, not a rule, so it
  has no F-row.
- **Date.** 2026-10-15.

## References

- Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance* 8(3),
  217–224.
- Falces Marín, J., Díaz Pardo de Vera, D. & Lopez Gonzalo, E. (2022). *PLOS ONE* 17(12), e0277042.
- Kingma, D. P. & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
- Rockafellar, R. T. & Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk* 2(3), 21–41.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A. & Klimov, O. (2017). Proximal policy optimization algorithms.
  arXiv:1707.06347.
- Sortino, F. & Price, L. (1994). Performance measurement in a downside risk framework. *Journal of Investing* 3(3),
  59–64.
- Stooke, A., Achiam, J. & Abbeel, P. (2020). Responsive safety in reinforcement learning by PID Lagrangian methods.
  *ICML*.
- Tamar, A., Glassner, Y. & Mannor, S. (2015). Optimizing the CVaR via sampling. *AAAI*.
