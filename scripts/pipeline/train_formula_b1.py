from __future__ import annotations

import argparse
import os
import re

if __package__:
    from .common import (
        archive_once,
        assert_complete_day_coverage,
        b1_artifact_stems,
        fixed_std_ppo,
        fixed_std_record,
        load_context,
        minimum_complete_day_steps,
        phase_training_inputs,
        print_context_banner,
        require_complete_day_budget,
        required_env,
        seeded_stem,
        write_metadata,
    )
else:
    from common import (  # type: ignore[no-redef]
        archive_once,
        assert_complete_day_coverage,
        b1_artifact_stems,
        fixed_std_ppo,
        fixed_std_record,
        load_context,
        minimum_complete_day_steps,
        phase_training_inputs,
        print_context_banner,
        require_complete_day_budget,
        required_env,
        seeded_stem,
        write_metadata,
    )
from cvar_mm.gym.notebook_support import build_formula_multi_day_replay_env, make_vecnorm
from cvar_mm.gym.sb3_wrapper import StableBaselinesTradingEnvironment
from cvar_mm.rewards import CjMmCriterion


def b1_passes() -> int:
    """B1_PASSES from configs/final_pipeline.env: the passes B1 and B1_cand train over their days (F-16: 1)."""

    raw = required_env("B1_PASSES")
    if not re.fullmatch(r"[1-9][0-9]*", raw):
        raise ValueError(f"B1_PASSES must be a positive integer, got {raw!r} (F-16)")
    return int(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train formula-based B1 PPO.")
    parser.add_argument("--phase", choices=("candidate", "final"), default="final")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("TRAINING_SEED", 42)))
    args = parser.parse_args()
    if "TOTAL_TIMESTEPS" in os.environ:
        raise RuntimeError(
            "TOTAL_TIMESTEPS is set: B1 and B1_cand train exactly B1_PASSES passes over their days and refuse "
            "it (F-16); unset it"
        )
    passes = b1_passes()

    candidate = args.phase == "candidate"
    job_name = "train_formula_b1_cand" if candidate else "train_formula_b1"
    ctx = load_context(require_prepared=candidate)
    print_context_banner(ctx, job_name)
    train_S, train_dt, train_dates, train_features, training_cfg, formula_kwargs = (
        phase_training_inputs(ctx, args.phase)
    )
    required_transitions = minimum_complete_day_steps(train_S)
    total_timesteps = passes * required_transitions  # F-16: a pass is every transition of every day once
    require_complete_day_budget(total_timesteps, train_S)

    env = build_formula_multi_day_replay_env(
        train_S,
        train_dt,
        training_cfg,
        daily_market_features=train_features,
        reward_fn=CjMmCriterion(per_step_inventory_aversion=training_cfg.phi),
        mode="sequential",
        **formula_kwargs,
    )
    sb3_env = StableBaselinesTradingEnvironment(env)
    vecnorm = make_vecnorm(sb3_env, training_cfg, training=True, norm_reward=True)

    model = fixed_std_ppo(  # D-21: sigma0 = POLICY_STD, fixed
        vecnorm,
        training_cfg.ppo_kwargs(),
        tensorboard_log=str(ctx.cfg.repo_root / "tb_logs" / ("b1_cand" if candidate else "b1")),
        verbose=1,
        device="cpu",
        seed=args.seed,
    )
    model.learn(total_timesteps=total_timesteps)
    coverage = assert_complete_day_coverage(env, train_dates)

    model_stem, vecnorm_stem = b1_artifact_stems(args.phase, args.seed)
    model_path = ctx.cfg.model_path(f"{model_stem}.zip")
    vecnorm_path = ctx.cfg.vecnorm_path(vecnorm_stem)
    if not candidate:  # pre-correction direct-depth artifacts exist only for the final B1
        archive_once(
            model_path,
            ctx.cfg.model_path(f'{seeded_stem("ppo_b1_doge_direct_depth", args.seed)}.zip'),
        )
        archive_once(
            vecnorm_path,
            ctx.cfg.vecnorm_path(seeded_stem("vecnorm_b1_direct_depth", args.seed)),
        )
    model.save(str(model_path))
    vecnorm.save(str(vecnorm_path))

    write_metadata(
        job_name,
        {
            "model": "b1",
            "phase": args.phase,
            "result_type": "formula_as_parameter_ppo",
            "model_path": model_path,
            "vecnorm_path": vecnorm_path,
            "total_timesteps": total_timesteps,
            "b1_passes": passes,
            "training_seed": args.seed,
            "train_dates": train_dates,
            "minimum_complete_day_transitions": required_transitions,
            "training_day_coverage": coverage,
            "calibration_scope": "candidate_train_only" if candidate else "all_development_days",
            "training_calibration": {
                "A": training_cfg.A,
                "kappa": training_cfg.kappa,
                "sigma": formula_kwargs["sigma"],
            },
            "formula_kwargs": formula_kwargs,
            "experiment_config": training_cfg,  # the calibrated A and kappa it trained with (item 8)
            "ppo_kwargs": training_cfg.ppo_kwargs(),
            **fixed_std_record(model),  # D-21
        },
    )
    print(f"Saved B1 model -> {model_path}")
    print(f"Saved B1 VecNormalize -> {vecnorm_path}")


if __name__ == "__main__":
    main()

