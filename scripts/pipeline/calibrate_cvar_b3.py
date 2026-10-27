"""Calibrate the frozen B3 threshold d from the final B1 on Jan 1-6 (F-06, D-17).

d = 0.8 x the empirical CVaR_0.20 of B1's full-day MaxDD from stochastic
rollouts: actions sampled with B1's own sigma, exactly as PPO's rollouts sample
them, so d and B3's H measure the same policy at warm start. A deterministic
(mean-action) pass on the same fill draws gives the evaluation-side CVaR, which
is recorded, not used. B1 must explore with sigma0 = POLICY_STD (D-21); the
deterministic pass also records the share of decisions pinned at a bound of the
action box (b1_pinned_share, a diagnostic).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO

if __package__:
    from .common import (
        b1_artifact_stems,
        env_int,
        load_context,
        load_train_arrays,
        parse_seed_list,
        print_context_banner,
        require_policy_std,
        seeded_result_name,
        write_metadata,
    )
else:
    from common import (  # type: ignore[no-redef]
        b1_artifact_stems,
        env_int,
        load_context,
        load_train_arrays,
        parse_seed_list,
        print_context_banner,
        require_policy_std,
        seeded_result_name,
        write_metadata,
    )
from cvar_mm.gym.cvar_lagrangian import (
    B3_POLICY_SAMPLING_SEED_RULE,
    B3_TAIL_MASS,
    B3_THRESHOLD_ESTIMATOR,
    B3_THRESHOLD_MULTIPLIER,
    b3_threshold_from_b1,
    collect_b1_full_day_maxdrawdowns,
    empirical_ru_cvar,
)


class PinnedDecisionCounter:
    """B1 as the D-17 collector uses it (``device`` and ``predict``), counting the deterministic pass's decisions
    whose mean action is pinned at a bound of the action box [-1, 1] (D-21; a diagnostic, never a gate).

    ``predict(deterministic=True)`` returns clip(mu) to the box, so |mu_i| >= 1 exactly where the returned action
    is -1 or +1. Actions pass through unchanged and both passes get this one object, so d and both CVaRs are B1's.
    """

    def __init__(self, model):
        space = model.action_space
        if not (np.all(space.low == -1.0) and np.all(space.high == 1.0)):
            raise ValueError(f"b1_pinned_share needs the action box [-1, 1], got [{space.low}, {space.high}] (D-21)")
        self.model = model
        self.device = model.device
        self.pinned = np.zeros(space.shape, dtype=np.int64)
        self.decisions = 0

    def predict(self, observation, deterministic: bool):
        actions, state = self.model.predict(observation, deterministic=deterministic)
        if deterministic:
            decisions = np.asarray(actions).reshape(-1, self.pinned.size)
            self.pinned += np.count_nonzero(np.abs(decisions) >= 1.0, axis=0)
            self.decisions += decisions.shape[0]
        return actions, state

    def share(self) -> list[float]:
        """Per action dimension, the share of deterministic decisions with |mu_i| >= 1."""

        if self.decisions == 0:
            raise RuntimeError("The deterministic pass made no decision, so b1_pinned_share is undefined (D-21)")
        return (self.pinned / self.decisions).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the frozen B3 full-day CVaR threshold from B1 (D-17).")
    parser.add_argument("--rollouts-per-seed", type=int, default=None)
    parser.add_argument("--fill-seeds", default=None)
    parser.add_argument(
        "--training-seed", type=int, default=int(os.environ.get("TRAINING_SEED", 42))
    )
    args = parser.parse_args()

    ctx = load_context()
    print_context_banner(ctx, "calibrate_cvar_b3")
    train_S, train_dt, train_dates, train_features = load_train_arrays(ctx)
    b1_model_stem, b1_vecnorm_stem = b1_artifact_stems("final", args.training_seed)  # B3's warm start
    model_path = ctx.cfg.model_path(f"{b1_model_stem}.zip")
    vecnorm_path = ctx.cfg.vecnorm_path(b1_vecnorm_stem)
    model_b1 = PPO.load(str(model_path), device="cpu")
    require_policy_std(model_b1.policy.log_std, source=f"B1 {model_path}")  # D-21, before the two passes
    counted_b1 = PinnedDecisionCounter(model_b1)

    fill_seeds = parse_seed_list(args.fill_seeds)
    rollouts = (  # an explicit --rollouts-per-seed 0 reaches the collector's check
        args.rollouts_per_seed
        if args.rollouts_per_seed is not None
        else env_int("CVAR_ROLLOUTS_PER_SEED", ctx.cfg.evaluation_rollouts)
    )
    rollout_kwargs = dict(
        daily_midprices=train_S,
        daily_dt_arrays=train_dt,
        daily_market_features=train_features,
        model=counted_b1,
        vecnorm_path=vecnorm_path,
        config=ctx.cfg,
        formula_kwargs=ctx.formula_kwargs,
        fill_seeds=fill_seeds,
        rollouts_per_seed=rollouts,
    )
    seconds_per_pass = {}
    start = time.perf_counter()
    stochastic, stochastic_records = collect_b1_full_day_maxdrawdowns(**rollout_kwargs, deterministic=False)
    seconds_per_pass["stochastic"] = time.perf_counter() - start
    start = time.perf_counter()
    deterministic, deterministic_records = collect_b1_full_day_maxdrawdowns(**rollout_kwargs, deterministic=True)
    seconds_per_pass["deterministic"] = time.perf_counter() - start

    # D-17: d from the stochastic pass, the sampling that produces B3's H.
    d, estimate = b3_threshold_from_b1(stochastic)
    if not d > 0:
        raise ValueError(f"d = {d}: the CVaR_0.2 of B1's stochastic full-day MaxDD is zero; B3 needs d > 0")
    deterministic_estimate = empirical_ru_cvar(deterministic, rho=B3_TAIL_MASS)  # recorded, not used
    b1_log_std = model_b1.policy.log_std.detach().cpu().numpy().copy()

    summary = {
        "estimator": B3_THRESHOLD_ESTIMATOR,
        "d": d,
        "rho": B3_TAIL_MASS,
        "threshold_multiplier": B3_THRESHOLD_MULTIPLIER,
        "cvar_stochastic": estimate.value,
        "nu_stochastic": estimate.nu,
        "cvar_deterministic": deterministic_estimate.value,
        "nu_deterministic": deterministic_estimate.nu,
        "b1_log_std": b1_log_std.tolist(),
        "b1_std": np.exp(b1_log_std).tolist(),  # as B3's frozen_std
        "b1_pinned_share": counted_b1.share(),  # D-21: deterministic decisions with |mu_i| >= 1, per action
        "fill_seeds": fill_seeds,
        "rollouts_per_seed": rollouts,
        "paths_per_estimator": int(stochastic.size),
        "policy_sampling_seed": B3_POLICY_SAMPLING_SEED_RULE,
        "training_seed": args.training_seed,
        "constraint_scope": "full_market_day_trajectory_maxdrawdown",
        "reference_model": "corrected_b1",
        "reference_model_path": str(model_path),
        "reference_vecnorm_path": str(vecnorm_path),
        "train_dates": train_dates,
    }
    txt_path = ctx.cfg.result_path(seeded_result_name("cvar_threshold_d.txt", args.training_seed))
    json_path = ctx.cfg.result_path(seeded_result_name("cvar_threshold_b3.json", args.training_seed))
    txt_path.write_text(f"{d}\n{estimate.value}\n{estimate.nu}\n", encoding="utf-8")
    # The profile stage hashes this file, so it holds no timings: a rerun on the same inputs is byte-identical.
    json_path.write_text(
        json.dumps({**summary, "trajectory_records": stochastic_records + deterministic_records}, indent=2),
        encoding="utf-8",
    )
    write_metadata(
        "calibrate_cvar_b3",
        {
            **summary,
            "threshold_path": txt_path,
            "threshold_json": json_path,
            "seconds_per_pass": seconds_per_pass,
            "torch_version": torch.__version__,
            "stable_baselines3_version": stable_baselines3.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "experiment_config": ctx.cfg,  # the calibrated A and kappa of B1's rollouts (item 8); not in the d JSON
        },
    )
    print(
        f"B1 full-day CVaR_{B3_TAIL_MASS:.0%}: stochastic {estimate.value:.6g} (nu {estimate.nu:.6g}), "
        f"deterministic {deterministic_estimate.value:.6g} (nu {deterministic_estimate.nu:.6g}); d = {d:.6g}"
    )
    print(
        f"Seconds per pass: stochastic {seconds_per_pass['stochastic']:.0f}, "
        f"deterministic {seconds_per_pass['deterministic']:.0f}"
    )
    print(f"Saved B3 CVaR threshold -> {txt_path}")
    print(f"Saved B3 CVaR threshold metadata -> {json_path}")


if __name__ == "__main__":
    # One policy call per step on 20 paths is too small to gain from threads; one thread also keeps the
    # float results independent of the node's core count. Set here, not in main(), which tests call.
    torch.set_num_threads(1)
    main()
