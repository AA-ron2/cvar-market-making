from __future__ import annotations

import argparse
import os

import pandas as pd
from stable_baselines3 import PPO

# common puts the project root on sys.path, so it must precede the cvar_mm imports.
if __package__:
    from .common import (
        candidate_config_and_formula,
        eval_episode_path,
        eval_part_path,
        final_model_stems,
        load_context,
        load_test_day,
        load_validation_day,
        model_key,
        parse_seed_list,
        print_context_banner,
        seeded_stem,
        write_metadata,
    )
    from .record_profile_stage import require_recorded
else:
    from common import (  # type: ignore[no-redef]
        candidate_config_and_formula,
        eval_episode_path,
        eval_part_path,
        final_model_stems,
        load_context,
        load_test_day,
        load_validation_day,
        model_key,
        parse_seed_list,
        print_context_banner,
        seeded_stem,
        write_metadata,
    )
    from record_profile_stage import require_recorded  # type: ignore[no-redef]
from cvar_mm.gym.notebook_support import evaluate_formula_rl_per_day


def _day_index_from_args(value: int | None) -> int:
    if value is not None:
        return value
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is None:
        raise ValueError("--day-index is required when SLURM_ARRAY_TASK_ID is not set.")
    return int(task_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one formula-AS model on one held-out test day.")
    parser.add_argument("--model", choices=("b1", "b1plus", "b2", "b3"), required=True)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--day-index", type=int, default=None)
    parser.add_argument(
        "--split",
        choices=("validation", "retrospective_holdout"),
        default="retrospective_holdout",
    )
    parser.add_argument("--seed", type=int, default=None, help="Deprecated single fill-seed alias.")
    parser.add_argument("--fill-seeds", default=None)
    parser.add_argument(
        "--training-seed", type=int, default=int(os.environ.get("TRAINING_SEED", 42))
    )
    args = parser.parse_args()

    day_index = _day_index_from_args(args.day_index)
    if args.model == "b2" and args.split == "validation" and args.alpha is None:
        raise ValueError("--alpha is required for B2 validation candidates")
    if args.model == "b2" and args.split == "retrospective_holdout" and args.alpha is not None:
        raise ValueError("Retrospective B2 evaluation must use the selected final model without --alpha")
    if args.model == "b1plus" and (args.split != "retrospective_holdout" or args.alpha is not None):
        raise ValueError("B1+ (D-18) is evaluated on the retrospective holdout only, without --alpha: it trained on Jan 1-6")

    ctx = load_context()
    print_context_banner(ctx, f"evaluate_formula_{model_key(args.model, args.alpha)}_day{day_index}")

    if args.model == "b2" and args.split == "validation":
        model_stem = seeded_stem(f"ppo_b2_candidate_alpha{args.alpha}_doge", args.training_seed)
        vecnorm_path = ctx.cfg.vecnorm_path(seeded_stem(
            f"vecnorm_b2_candidate_alpha{args.alpha}", args.training_seed
        ))
        evaluation_cfg, formula_kwargs = candidate_config_and_formula(ctx)
    else:
        # A key without final stems raises, so a new model cannot fall through to another's files.
        model_stem, vecnorm_stem = final_model_stems(args.model, args.training_seed)
        vecnorm_path = ctx.cfg.vecnorm_path(vecnorm_stem)
        evaluation_cfg, formula_kwargs = ctx.cfg, ctx.formula_kwargs

    # Explicit .zip: dotted stems (alpha0.1) would otherwise lose their suffix or hit a stale file.
    model_path = ctx.cfg.model_path(f"{model_stem}.zip")
    if args.model == "b3":  # D-20: B3 is the promoted eta run; evaluate only what the profile stage recorded
        require_recorded(ctx.cfg, [model_path, vecnorm_path])
    loader = load_validation_day if args.split == "validation" else load_test_day
    S, dt, date, features = loader(ctx, day_index)
    model = PPO.load(str(model_path), device="cpu")
    fill_seeds = [args.seed] if args.seed is not None else parse_seed_list(args.fill_seeds)
    seed_frames, episode_frames = [], []
    for fill_seed in fill_seeds:
        seed_frame, episodes = evaluate_formula_rl_per_day(
            model=model,
            vecnorm_path=vecnorm_path,
            test_S=[S],
            test_dt=[dt],
            test_dates=[date],
            test_market_features=[features],
            config=evaluation_cfg,
            seed=fill_seed,
            num_rollouts=ctx.cfg.evaluation_rollouts,
            return_episodes=True,
            **formula_kwargs,
        )
        seed_frame["Fill Seed"] = fill_seed
        seed_frames.append(seed_frame.reset_index())
        episode_frames.append(episodes.assign(**{"Fill Seed": fill_seed}))
    all_seed_rows = pd.concat(seed_frames, ignore_index=True)
    numeric = all_seed_rows.select_dtypes(include="number").drop(columns=["Fill Seed"])
    row = numeric.mean().to_dict()
    row["Rollouts"] = float(ctx.cfg.evaluation_rollouts * len(fill_seeds))
    df = pd.DataFrame([row], index=pd.Index([date], name="Day"))
    df["Fill Seeds"] = ",".join(str(seed) for seed in fill_seeds)
    df["Training Seed"] = args.training_seed
    df["Model"] = args.model.upper()
    if args.alpha is not None:
        df["Alpha"] = args.alpha
    df["Result Type"] = "formula_as_parameter_ppo"

    out_path = eval_part_path(
        ctx.cfg,
        model=args.model,
        alpha=args.alpha,
        date=date,
        split=args.split,
        training_seed=args.training_seed,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    # One row per full-day path (fill seed x rollout) for the episode-level CVaR (K6).
    episodes_path = eval_episode_path(
        ctx.cfg,
        model=args.model,
        alpha=args.alpha,
        date=date,
        split=args.split,
        training_seed=args.training_seed,
    )
    episodes = pd.concat(episode_frames, ignore_index=True)
    leading = ["Day", "Fill Seed", "Rollout"]
    episodes[leading + [column for column in episodes.columns if column not in leading]].to_csv(
        episodes_path, index=False
    )

    write_metadata(
        f"evaluate_formula_{model_key(args.model, args.alpha)}",
        {
            "model": args.model,
            "alpha": args.alpha,
            "day_index": day_index,
            "date": date,
            "split": args.split,
            "output_path": out_path,
            "episodes_path": episodes_path,
            "model_path": model_path,
            "vecnorm_path": vecnorm_path,
            "rollouts": ctx.cfg.evaluation_rollouts,
            "fill_seeds": fill_seeds,
            "training_seed": args.training_seed,
            "result_type": "formula_as_parameter_ppo",
            "experiment_config": evaluation_cfg,  # the calibrated A and kappa it simulated with (item 8)
            "formula_kwargs": formula_kwargs,
        },
    )
    print(f"Saved evaluation part -> {out_path}")
    print(df.to_string())


if __name__ == "__main__":
    main()

