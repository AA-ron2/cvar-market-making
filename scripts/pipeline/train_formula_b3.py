from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

if __package__:
    from .common import (
        archive_once,
        assert_complete_day_coverage,
        b1_artifact_stems,
        b3_inputs,
        b3_run_paths,
        file_sha256,
        load_context,
        load_train_arrays,
        manifest_path,
        minimum_complete_day_steps,
        policy_std,
        print_context_banner,
        required_env,
        risk_aware_ppo_kwargs,
        risk_aware_updates,
        seeded_result_name,
        seeded_stem,
        train_params_json_path,
        training_code_state,
        training_fill_paths,
        warm_start_from,
        write_metadata,
    )
else:
    from common import (  # type: ignore[no-redef]
        archive_once,
        assert_complete_day_coverage,
        b1_artifact_stems,
        b3_inputs,
        b3_run_paths,
        file_sha256,
        load_context,
        load_train_arrays,
        manifest_path,
        minimum_complete_day_steps,
        policy_std,
        print_context_banner,
        required_env,
        risk_aware_ppo_kwargs,
        risk_aware_updates,
        seeded_result_name,
        seeded_stem,
        train_params_json_path,
        training_code_state,
        training_fill_paths,
        warm_start_from,
        write_metadata,
    )
from cvar_mm.gym.cvar_lagrangian import (
    B3_LAMBDA_MAX,
    B3_TAIL_MASS,
    B3_THRESHOLD_ESTIMATOR,
    B3_THRESHOLD_MULTIPLIER,
    FullDayRUCVaRCallback,
    training_feasibility,
)
from cvar_mm.gym.drawdown_objectives import ACTION_SHIFT_COLUMNS, UPDATE_DIAGNOSTICS
from cvar_mm.gym.notebook_support import build_formula_multi_day_replay_env, freeze_vecnorm
from cvar_mm.gym.sb3_wrapper import StableBaselinesTradingEnvironment
from cvar_mm.rewards import CjMmCriterion


def load_b3_threshold(ctx, seed: int) -> float:
    """d from calibrate_cvar_b3.py: 0.8 x CVaR_0.2 of stochastic B1 rollouts (F-06, D-17).

    Fails closed on a d file pair that the D-17 calibration did not write, such
    as a pre-D-17 (deterministic) d or a txt whose d differs from its JSON.
    """

    threshold_path = ctx.cfg.result_path(seeded_result_name("cvar_threshold_d.txt", seed))
    if not threshold_path.exists():
        raise FileNotFoundError("Missing cvar_threshold_d.txt. Run calibrate_cvar_b3.py first.")
    threshold_metadata = json.loads(
        ctx.cfg.result_path(seeded_result_name("cvar_threshold_b3.json", seed)).read_text(encoding="utf-8")
    )
    if threshold_metadata.get("rho") != B3_TAIL_MASS:
        raise ValueError("Threshold metadata does not use the frozen rho=0.20")
    d = float(threshold_path.read_text(encoding="utf-8").splitlines()[0])
    recorded = {key: threshold_metadata.get(key) for key in ("estimator", "threshold_multiplier", "d")}
    expected = {"estimator": B3_THRESHOLD_ESTIMATOR, "threshold_multiplier": B3_THRESHOLD_MULTIPLIER, "d": d}
    if recorded != expected:
        raise ValueError(
            f"{threshold_path.name} is not a D-17 threshold: its JSON records {recorded}, expected {expected}. "
            "Rerun calibrate_cvar_b3.py (DECISIONS D-17)."
        )
    return d


def update_diagnostics_columns(callback) -> dict[str, list]:
    """The D-15 per-update diagnostics as history columns (B3 history and eta profiling histories), in the
    order of drawdown_objectives.UPDATE_DIAGNOSTIC_COLUMNS: D-21's action_shift_vs_b1 gives one column per action."""

    columns: dict[str, list] = {}
    for key in UPDATE_DIAGNOSTICS:
        values = [row[key] for row in callback.diagnostics]
        if key == "action_shift_vs_b1":
            columns.update({column: [value[i] for value in values] for i, column in enumerate(ACTION_SHIFT_COLUMNS)})
        else:
            columns[key] = values
    return columns


def b3_provenance_at_start(ctx, seed: int) -> dict:
    """What a B3 run loads and the code it starts from, taken when it starts (D-19, D-20): write_metadata hashes
    the payload's files only when the job ends, days later, so record_profile_stage.py checks these instead."""

    return {
        **training_code_state(),
        "input_sha256": {name: file_sha256(path) for name, path in b3_inputs(ctx.cfg, seed).items()},
    }


def run_b3(
    ctx,
    train_S,
    train_dt,
    train_dates,
    train_features,
    *,
    threshold_d: float,
    lambda_lr: float,
    updates: int,
    fill_paths: int,
    seed: int,
    tensorboard_log: str | None,
    extra_callbacks: Sequence[BaseCallback] = (),
) -> SimpleNamespace:
    """Warm-start B3 from the final B1 and run exactly ``updates`` PPO iterations (D-11, D-12, D-15).

    ``extra_callbacks`` only observe (the eta profiler's memory probe); they run after B3's callback.
    """

    b1_model_stem, b1_vecnorm_stem = b1_artifact_stems("final", seed)
    b1_model_path = ctx.cfg.model_path(f"{b1_model_stem}.zip")
    b1_vecnorm_path = ctx.cfg.vecnorm_path(b1_vecnorm_stem)
    complete_batch_steps = minimum_complete_day_steps(train_S)

    # D-12(c): N fill paths per day share the replayed mid and draw independent fills.
    env = build_formula_multi_day_replay_env(
        train_S,
        train_dt,
        ctx.cfg,
        daily_market_features=train_features,
        reward_fn=CjMmCriterion(per_step_inventory_aversion=ctx.cfg.phi),
        mode="sequential",
        num_trajectories=fill_paths,
        **ctx.formula_kwargs,
    )
    sb3_env = StableBaselinesTradingEnvironment(env)
    # D-15a: B1's frozen normalizer, rewards included (B1's s and clip_reward).
    vecnorm = freeze_vecnorm(b1_vecnorm_path, sb3_env, ctx.cfg, norm_reward=True)
    # SB3 counts n_steps * num_envs per iteration, so this is exactly k iterations.
    total_timesteps = complete_batch_steps * sb3_env.num_envs * updates

    ppo_kwargs = risk_aware_ppo_kwargs(ctx.cfg, complete_batch_steps, sb3_env.num_envs)  # D-15, D-23
    model = PPO(
        "MlpPolicy",
        vecnorm,
        **ppo_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=1,
        device="cpu",
        seed=seed,
    )
    reward_scale = warm_start_from(model, b1_model_path, vecnorm)
    frozen_log_std = model.policy.log_std.detach().cpu().numpy().copy()
    print(f"B3 warm-started from {b1_model_path} with its frozen normalizer {b1_vecnorm_path}")
    callback = FullDayRUCVaRCallback(
        dd_threshold=threshold_d,
        expected_day_indices=tuple(range(len(train_dates))),
        reward_scale=reward_scale,
        rho=B3_TAIL_MASS,
        lambda_learning_rate=lambda_lr,
        verbose=1,
    )
    # SB3 wraps only a list in a CallbackList; without extras learn gets B3's callback exactly as before.
    model.learn(total_timesteps=total_timesteps, callback=[callback, *extra_callbacks] if extra_callbacks else callback)
    return SimpleNamespace(
        model=model,
        vecnorm=vecnorm,
        callback=callback,
        coverage=assert_complete_day_coverage(env, train_dates),
        ppo_kwargs=ppo_kwargs,
        total_timesteps=total_timesteps,
        warm_start_model=b1_model_path,
        warm_start_vecnorm=b1_vecnorm_path,
        reward_scale_s=reward_scale,
        frozen_log_std=frozen_log_std,
    )


def save_b3_run(
    ctx,
    run: SimpleNamespace,
    paths: dict[str, Path],
    *,
    threshold_d: float,
    lambda_lr: float,
    updates: int,
    fill_paths: int,
    seed: int,
    train_dates: list[str],
    provenance_at_start: dict,
    unequal_updates_override: bool = False,
) -> dict:
    """Save one B3 run's files at ``paths`` (common.B3_RUN_FILES) and return its metadata payload.

    The same for train_formula_b3.py and for every eta-profiling run, one of which becomes the final B3 (D-20).
    """

    callback = run.callback
    run.model.save(str(paths["model_path"]))
    run.vecnorm.save(str(paths["vecnorm_path"]))
    pd.DataFrame(
        {
            "rollout": range(1, len(callback.lambda_history) + 1),  # as in the costs and feasibility files
            "lambda_used": callback.lambda_used_history,
            "lambda": callback.lambda_history,
            "nu": callback.nu_history,
            "ru_cvar": callback.cvar_history,
            "violation": callback.violation_history,
            "mean_full_day_maxdd": callback.mean_maxdd_history,
            **update_diagnostics_columns(callback),  # D-15, incl. elapsed_seconds
            "threshold_d": threshold_d,
        }
    ).to_csv(paths["history_path"], index=False)
    pd.DataFrame([
        {
            "update": update,
            "day_index": segment.day_index,
            "path": segment.env_index,
            "max_drawdown": segment.max_drawdown,
        }
        for update, segments in enumerate(callback.segment_history, start=1)
        for segment in segments
    ]).to_csv(paths["training_costs_path"], index=False)
    feasibility = training_feasibility(
        [[segment.max_drawdown for segment in segments] for segments in callback.segment_history], threshold_d
    )
    paths["training_feasibility_path"].write_text(json.dumps(feasibility, indent=2), encoding="utf-8")
    return {
        "model": "b3",
        "result_type": "formula_as_parameter_ppo_full_day_ru_cvar",
        **paths,  # write_metadata hashes these files when the job ends
        "training_feasibility": feasibility,
        "threshold_d": threshold_d,
        "constraint_scope": "full_market_day_trajectory_maxdrawdown",
        "rho": B3_TAIL_MASS,
        "nu_rule": "batch empirical (1 - rho)-quantile (D-12(a))",
        "lambda_step": "clip(lambda + eta * (H - d) / d, 0, lambda_max) (D-12(b))",
        "fine_tuning": "D-15/D-15a/D-15b/D-15c",
        "advantage": "(A - mu_A)/sigma_A - lambda * (c_i - leave-one-out day mean) / (s * sigma_A), "
                     "c = (Z - nu)+ / rho (D-15a)",
        "lambda_learning_rate": lambda_lr,
        "lambda_max": B3_LAMBDA_MAX,
        "training_fill_paths": fill_paths,
        "score_baseline": "per_day_leave_one_out",
        "risk_aware_updates": updates,
        "declared_updates": {name: os.environ.get(name) for name in ("B2_UPDATES", "B3_UPDATES")},
        "unequal_updates_override": unequal_updates_override,
        "warm_start_model": run.warm_start_model,
        "warm_start_vecnorm": run.warm_start_vecnorm,
        "normalizer_frozen_during_b3": True,
        "reward_scale_s": run.reward_scale_s,
        "frozen_log_std": run.frozen_log_std.tolist(),
        "frozen_std": np.exp(run.frozen_log_std).tolist(),
        "policy_std": policy_std(),  # D-21: the warm start checked frozen_std against it
        "learning_rate": float(run.model.learning_rate),  # D-23: PPO's (Adam's) rate, FINE_TUNE_LR; not lambda's eta
        "adam_state": "fresh (D-15c)",
        "training_diagnostics": callback.diagnostics,
        "total_timesteps": run.total_timesteps,
        "training_seed": seed,
        "train_dates": train_dates,
        "training_day_coverage": run.coverage,
        "formula_kwargs": ctx.formula_kwargs,
        "experiment_config": ctx.cfg,  # the calibrated A and kappa it trained with (item 8)
        "ppo_kwargs": run.ppo_kwargs,
        "train_params": train_params_json_path(ctx.cfg),
        "manifest": manifest_path(ctx.cfg),
        "provenance_at_start": provenance_at_start,
    }


def main() -> None:
    """Train one B3 under the final names. The pipeline no longer runs this (D-20): the final B3 is the
    eta-profiling run D-12a selects, which profile_b3_dual.py saves with save_b3_run and --select promotes."""

    parser = argparse.ArgumentParser(description="Train formula-based B3 with full-day RU-CVaR PPO.")
    parser.add_argument("--allow-unequal-updates", action="store_true")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("TRAINING_SEED", 42)))
    args = parser.parse_args()

    ctx = load_context()
    print_context_banner(ctx, "train_formula_b3")
    train_S, train_dt, train_dates, train_features = load_train_arrays(ctx)
    updates = risk_aware_updates("b3", allow_unequal=args.allow_unequal_updates)
    if updates < 2:
        raise ValueError("B3_UPDATES must be at least 2: rollout 1 is the warm-start B1 (lambda_0 = 0)")
    lambda_lr = float(required_env("B3_LAMBDA_LR"))  # chosen by profile_b3_dual.py (D-12a)
    fill_paths = training_fill_paths()
    d = load_b3_threshold(ctx, args.seed)
    provenance_at_start = b3_provenance_at_start(ctx, args.seed)

    run = run_b3(
        ctx,
        train_S,
        train_dt,
        train_dates,
        train_features,
        threshold_d=d,
        lambda_lr=lambda_lr,
        updates=updates,
        fill_paths=fill_paths,
        seed=args.seed,
        tensorboard_log=str(ctx.cfg.repo_root / "tb_logs" / "b3"),
    )

    paths = b3_run_paths(ctx.cfg, args.seed)
    archive_once(
        paths["model_path"],
        ctx.cfg.model_path(f'{seeded_stem("ppo_b3_doge_direct_depth", args.seed)}.zip'),
    )
    archive_once(
        paths["vecnorm_path"],
        ctx.cfg.vecnorm_path(seeded_stem("vecnorm_b3_direct_depth", args.seed)),
    )
    payload = save_b3_run(
        ctx,
        run,
        paths,
        threshold_d=d,
        lambda_lr=lambda_lr,
        updates=updates,
        fill_paths=fill_paths,
        seed=args.seed,
        train_dates=train_dates,
        provenance_at_start=provenance_at_start,
        unequal_updates_override=args.allow_unequal_updates,
    )
    write_metadata("train_formula_b3", payload)
    print(f"Saved B3 model -> {paths['model_path']}")
    print(f"Saved B3 VecNormalize -> {paths['vecnorm_path']}")
    print(f"Saved B3 CVaR history -> {paths['history_path']}")
    feasibility = payload["training_feasibility"]
    print(
        f"B3 training feasibility (last {len(feasibility['rollouts'])} rollouts) -> "
        f"{paths['training_feasibility_path']}: {feasibility}"
    )


if __name__ == "__main__":
    main()
