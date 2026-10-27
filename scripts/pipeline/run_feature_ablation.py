from __future__ import annotations

import argparse

# common puts the project root on sys.path, so it must precede the cvar_mm imports.
if __package__:
    from .common import (
        assert_complete_day_coverage,
        candidate_config_and_formula,
        env_int,
        fixed_std_ppo,
        fixed_std_record,
        load_context,
        minimum_complete_day_steps,
        print_context_banner,
        require_complete_day_budget,
        write_metadata,
    )
else:
    from common import (  # type: ignore[no-redef]
        assert_complete_day_coverage,
        candidate_config_and_formula,
        env_int,
        fixed_std_ppo,
        fixed_std_record,
        load_context,
        minimum_complete_day_steps,
        print_context_banner,
        require_complete_day_budget,
        write_metadata,
    )
from cvar_mm.gym.formula_as import FORMULA_FEATURE_NAMES
from cvar_mm.gym.notebook_support import (
    build_formula_multi_day_replay_env,
    evaluate_formula_rl_per_day,
    make_vecnorm,
)
from cvar_mm.gym.sb3_wrapper import StableBaselinesTradingEnvironment
from cvar_mm.rewards import CjMmCriterion

FEATURE_SETS = {
    "base_state": ("inventory_norm", "time_remaining_frac", "log_mid_rel"),
    "base_plus_vol": ("inventory_norm", "time_remaining_frac", "log_mid_rel", "rolling_vol_ticks"),
    "full": tuple(FORMULA_FEATURE_NAMES),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one formula-AS feature ablation job.")
    parser.add_argument("--feature-set", choices=tuple(FEATURE_SETS), required=True)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ctx = load_context()
    print_context_banner(ctx, f"run_feature_ablation_{args.feature_set}")
    features = FEATURE_SETS[args.feature_set]
    candidate_cfg, candidate_formula_kwargs = candidate_config_and_formula(ctx)

    train_rows = ctx.candidate_train_manifest
    val_rows = ctx.validation_manifest
    train_S, train_dt, train_features = [], [], []
    val_S, val_dt, val_dates, val_features = [], [], [], []
    from cvar_mm.gym.data_loader import load_single_day_with_features

    for row in train_rows.itertuples(index=False):
        S, dt, _, feats = load_single_day_with_features(row.input_path, tick_size=ctx.cfg.tick_size)
        train_S.append(S)
        train_dt.append(dt)
        train_features.append(feats)
    for row in val_rows.itertuples(index=False):
        S, dt, _, feats = load_single_day_with_features(row.input_path, tick_size=ctx.cfg.tick_size)
        val_S.append(S)
        val_dt.append(dt)
        val_dates.append(str(row.date))
        val_features.append(feats)

    required_transitions = minimum_complete_day_steps(train_S)
    total_timesteps = args.total_timesteps or env_int(
        "ABLATION_TIMESTEPS", required_transitions
    )
    require_complete_day_budget(total_timesteps, train_S)

    env = build_formula_multi_day_replay_env(
        train_S,
        train_dt,
        candidate_cfg,
        daily_market_features=train_features,
        reward_fn=CjMmCriterion(per_step_inventory_aversion=candidate_cfg.phi),
        mode="sequential",
        enabled_features=features,
        **candidate_formula_kwargs,
    )
    sb3_env = StableBaselinesTradingEnvironment(env)
    vecnorm = make_vecnorm(sb3_env, candidate_cfg, training=True, norm_reward=True)
    model = fixed_std_ppo(  # D-21: sigma0 = POLICY_STD, fixed, as in B1
        vecnorm,
        ctx.cfg.ppo_kwargs(),
        tensorboard_log=str(ctx.cfg.repo_root / "tb_logs" / f"ablation_{args.feature_set}"),
        verbose=1,
        device="cpu",
        seed=args.seed,
    )
    model.learn(total_timesteps=total_timesteps)
    coverage = assert_complete_day_coverage(env, train_rows["date"].astype(str).tolist())

    model_stem = f"ppo_ablation_{args.feature_set}"
    vecnorm_stem = f"vecnorm_ablation_{args.feature_set}"
    model.save(str(ctx.cfg.model_path(f"{model_stem}.zip")))
    vecnorm.save(str(ctx.cfg.vecnorm_path(vecnorm_stem)))

    df = evaluate_formula_rl_per_day(
        model=model,
        vecnorm_path=ctx.cfg.vecnorm_path(vecnorm_stem),
        test_S=val_S,
        test_dt=val_dt,
        test_dates=val_dates,
        test_market_features=val_features,
        config=candidate_cfg,
        seed=args.seed,
        num_rollouts=ctx.cfg.evaluation_rollouts,
        enabled_features=features,
        **candidate_formula_kwargs,
    )
    out_dir = ctx.cfg.results_dir / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.feature_set}_validation_results.csv"
    df.to_csv(out_path)
    write_metadata(
        f"feature_ablation_{args.feature_set}",
        {
            "feature_set": args.feature_set,
            "features": features,
            "total_timesteps": total_timesteps,
            "output_path": out_path,
            "model_path": ctx.cfg.model_path(f"{model_stem}.zip"),
            "vecnorm_path": ctx.cfg.vecnorm_path(vecnorm_stem),
            "candidate_train_dates": train_rows["date"].astype(str).tolist(),
            "validation_dates": val_dates,
            "training_day_coverage": coverage,
            "interpretation": "exploratory model-specification diagnostic; not confirmatory inference",
            "calibration_scope": "candidate_train_only",
            "experiment_config": candidate_cfg,  # the Jan 1-4 A and kappa it trained and validated with (item 8)
            "formula_kwargs": candidate_formula_kwargs,
            **fixed_std_record(model),  # D-21
        },
    )
    print(f"Saved ablation validation -> {out_path}")


if __name__ == "__main__":
    main()

