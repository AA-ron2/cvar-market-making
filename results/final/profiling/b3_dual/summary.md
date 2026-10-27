# B3 dual step size: D-12a summary

**Selected: η_λ = 8.0.** The line for `configs/final_pipeline.env` is `export B3_LAMBDA_LR=8.0`.

Jan 1–6 only (Jan 7–29 never loaded); 10 B3 updates per η ∈ {0.5, 2, 8}; N = 16 fill paths; training seed 42; λ_max = 500. Written by `profile_b3_dual.py --select`, 2026-10-15T07:27:35.478281+00:00 (git fe5f9feabb460a2f7ec646f79ef7a11269a69955).

## B1 and the threshold (D-17)

B1's policy noise, fixed at σ₀ = POLICY_STD in B1 (D-21) and frozen during B3 (D-15b), in PPO action units on [−1, 1]:

| Action | log σ | σ |
|---|---|---|
| γ | -1.204 | 0.3 |
| skew | -1.204 | 0.3 |

- d = 0.0729883, i.e. 0.8 × the stochastic CVaR₀.₂ of B1's full-day MaxDD.
- CVaR₀.₂ stochastic = 0.0912354 (used for d); deterministic = 0.0915705 (recorded, not used); deterministic − stochastic = 0.000335104 (0.367 % of the stochastic value).

## Rule (D-12a)

eta_lambda in {0.5, 2, 8}, 10 B3 updates each on Jan 1-6 with training seed 42. Exclude an eta if lambda after any update reaches lambda_max, or if H_u - d changes sign more than twice over the 10 updates (an exact zero is not a sign). Among the rest choose the smallest mean |H_u - d|/d over updates 8-10; ties go to the smaller eta. If the lambda used at every update is 0 for every eta, the runs coincide and their criteria tie by construction. If every eta is excluded, stop: there is no fallback (DECISIONS D-12a).

## Per η

| η | max λ | sign changes of Ĥ − d | criterion: mean \|Ĥ − d\|/d, updates 8–10 | kept | exclusion reason | KL stops | mean s / update | peak RSS (GiB) |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 1.465 | 0 | 0.283793 | yes | — | 1 of 10 | 5425.1 | 10.074 |
| 2 | 5.841 | 0 | 0.277488 | yes | — | 1 of 10 | 5360.2 | 10.135 |
| 8 | 21.92 | 0 | 0.249165 | yes | — | 2 of 10 | 5253.4 | 10.138 |

### η = 0.5

| update | λ used | λ after | Ĥ/d | epochs completed | KL stop | adam_damping_ratio | action shift vs B1, γ / skew | value_loss | explained_variance | seconds | peak RSS so far (GiB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0.1454 | 1.2908 | 0 | yes | 1.000 | 0.0580 / 0.0427 | 0.3027 | 0.993 | 3993.5 | 9.526 |
| 2 | 0.1454 | 0.2797 | 1.2686 | 10 | no | 0.811 | 0.0612 / 0.0447 | 0.2483 | 0.992 | 5612.9 | 10.040 |
| 3 | 0.2797 | 0.4556 | 1.3518 | 10 | no | 0.836 | 0.0567 / 0.0445 | 0.2412 | 0.992 | 5572.9 | 10.074 |
| 4 | 0.4556 | 0.5967 | 1.2823 | 10 | no | 0.801 | 0.0612 / 0.0476 | 0.2537 | 0.991 | 5585.1 | 10.074 |
| 5 | 0.5967 | 0.7492 | 1.3050 | 10 | no | 0.808 | 0.0554 / 0.0489 | 0.2596 | 0.991 | 5582.7 | 10.074 |
| 6 | 0.7492 | 0.9198 | 1.3411 | 10 | no | 0.797 | 0.0497 / 0.0484 | 0.2585 | 0.990 | 5575.7 | 10.074 |
| 7 | 0.9198 | 1.039 | 1.2380 | 10 | no | 0.787 | 0.0515 / 0.0483 | 0.2665 | 0.990 | 5588.4 | 10.074 |
| 8 | 1.039 | 1.152 | 1.2272 | 10 | no | 0.788 | 0.0553 / 0.0500 | 0.2585 | 0.989 | 5581.3 | 10.074 |
| 9 | 1.152 | 1.293 | 1.2815 | 10 | no | 0.796 | 0.0572 / 0.0542 | 0.2601 | 0.989 | 5578.4 | 10.074 |
| 10 | 1.293 | 1.465 | 1.3427 | 10 | no | 0.791 | 0.0550 / 0.0535 | 0.2605 | 0.988 | 5579.6 | 10.074 |

### η = 2

| update | λ used | λ after | Ĥ/d | epochs completed | KL stop | adam_damping_ratio | action shift vs B1, γ / skew | value_loss | explained_variance | seconds | peak RSS so far (GiB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0.5815 | 1.2908 | 0 | yes | 1.000 | 0.0580 / 0.0427 | 0.3027 | 0.993 | 3960.5 | 9.519 |
| 2 | 0.5815 | 1.119 | 1.2686 | 10 | no | 0.813 | 0.0620 / 0.0396 | 0.2739 | 0.992 | 5498.5 | 9.950 |
| 3 | 1.119 | 1.801 | 1.3412 | 10 | no | 0.799 | 0.0608 / 0.0405 | 0.2353 | 0.992 | 5520.7 | 9.950 |
| 4 | 1.801 | 2.373 | 1.2857 | 10 | no | 0.802 | 0.0640 / 0.0433 | 0.2581 | 0.992 | 5522.0 | 9.950 |
| 5 | 2.373 | 2.972 | 1.2995 | 10 | no | 0.790 | 0.0704 / 0.0453 | 0.2553 | 0.991 | 5511.1 | 9.950 |
| 6 | 2.972 | 3.678 | 1.3533 | 10 | no | 0.792 | 0.0716 / 0.0419 | 0.2552 | 0.991 | 5520.0 | 10.135 |
| 7 | 3.678 | 4.176 | 1.2488 | 10 | no | 0.771 | 0.0762 / 0.0379 | 0.2566 | 0.990 | 5513.8 | 10.135 |
| 8 | 4.176 | 4.656 | 1.2402 | 10 | no | 0.791 | 0.0794 / 0.0384 | 0.2522 | 0.990 | 5525.0 | 10.135 |
| 9 | 4.656 | 5.197 | 1.2703 | 10 | no | 0.798 | 0.0824 / 0.0396 | 0.2467 | 0.990 | 5503.7 | 10.135 |
| 10 | 5.197 | 5.841 | 1.3220 | 10 | no | 0.774 | 0.0835 / 0.0402 | 0.247 | 0.989 | 5526.7 | 10.135 |

### η = 8

| update | λ used | λ after | Ĥ/d | epochs completed | KL stop | adam_damping_ratio | action shift vs B1, γ / skew | value_loss | explained_variance | seconds | peak RSS so far (GiB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 2.326 | 1.2908 | 0 | yes | 1.000 | 0.0580 / 0.0427 | 0.3027 | 0.993 | 3988.1 | 9.557 |
| 2 | 2.326 | 4.475 | 1.2686 | 0 | yes | 1.413 | 0.0547 / 0.0394 | 0.2842 | 0.992 | 3988.6 | 9.675 |
| 3 | 4.475 | 7.316 | 1.3552 | 10 | no | 0.790 | 0.0643 / 0.0399 | 0.3146 | 0.992 | 5587.8 | 10.012 |
| 4 | 7.316 | 9.679 | 1.2953 | 10 | no | 0.804 | 0.0592 / 0.0365 | 0.2465 | 0.992 | 5583.1 | 10.012 |
| 5 | 9.679 | 11.89 | 1.2763 | 10 | no | 0.794 | 0.0587 / 0.0374 | 0.2378 | 0.991 | 5583.5 | 10.038 |
| 6 | 11.89 | 14.18 | 1.2865 | 10 | no | 0.782 | 0.0485 / 0.0430 | 0.2413 | 0.991 | 5556.8 | 10.138 |
| 7 | 14.18 | 15.94 | 1.2193 | 10 | no | 0.789 | 0.0488 / 0.0441 | 0.2512 | 0.990 | 5562.2 | 10.138 |
| 8 | 15.94 | 17.63 | 1.2120 | 10 | no | 0.780 | 0.0467 / 0.0444 | 0.2517 | 0.989 | 5562.5 | 10.138 |
| 9 | 17.63 | 19.72 | 1.2606 | 10 | no | 0.788 | 0.0391 / 0.0421 | 0.2542 | 0.989 | 5567.2 | 10.138 |
| 10 | 19.72 | 21.92 | 1.2749 | 10 | no | 0.785 | 0.0465 / 0.0432 | 0.2523 | 0.988 | 5554.3 | 10.138 |

## Memory cross-check

`peak_rss_so_far_gib` is the process's running maximum of `ru_maxrss` after each update. With one η per Slurm array task (the layout `submit_final_pipeline.sh` submits), an η's last value is that task's peak; if the three η ran in one process (no `--eta-index` and no `SLURM_ARRAY_TASK_ID`), each value is the maximum since that process started, so a later η's values include the earlier η's peaks. Cross-check it against Slurm's own record:

    sacct -j <jobid> --format=JobID,MaxRSS

Each task's job ID is in its log name, `logs/b3_dual_<jobid>.out`; the array job ID that `submit_final_pipeline.sh` prints as "eta profile" covers all three (task i is ETA_GRID[i]). Compare with the `.batch` step, and with the `--mem` the jobs request (`scripts/snellius/run_python_job.sh`).

## Next step

- Set `B3_LAMBDA_LR=8.0` in `configs/final_pipeline.env` (k = 10 is frozen, D-19), commit, and submit `PIPELINE_STAGE=final` after the go/no-go review of the whole profile stage (D-19).
