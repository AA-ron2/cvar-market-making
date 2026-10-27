from __future__ import annotations

import argparse
import os

import numpy as np
from stable_baselines3 import PPO

if __package__:
    from .common import (
        archive_once,
        assert_complete_day_coverage,
        b1_artifact_stems,
        b2_final_paths,
        b2_final_run_stems,
        b2_promotion_path,
        file_sha256,
        final_model_stems,
        load_context,
        manifest_path,
        minimum_complete_day_steps,
        phase_training_inputs,
        policy_std,
        print_context_banner,
        promote_b2,
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
        b2_final_paths,
        b2_final_run_stems,
        b2_promotion_path,
        file_sha256,
        final_model_stems,
        load_context,
        manifest_path,
        minimum_complete_day_steps,
        phase_training_inputs,
        policy_std,
        print_context_banner,
        promote_b2,
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
from cvar_mm.gym.drawdown_objectives import FullDayMaxDrawdownPPOCallback
from cvar_mm.gym.model_selection import require_b2_alpha
from cvar_mm.gym.notebook_support import build_formula_multi_day_replay_env, freeze_vecnorm
from cvar_mm.gym.sb3_wrapper import StableBaselinesTradingEnvironment
from cvar_mm.rewards import CjMmCriterion


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train formula-based B2 PPO for one drawdown alpha, or B1+; or promote the selected final B2."
    )
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--phase", choices=("candidate", "final"), default=None, help="default: candidate")
    parser.add_argument(
        "--control",
        choices=("b1plus",),
        default=None,
        help="b1plus: the B1+ control (D-18), the final B1 fine-tuned through this B2 path at alpha = 0",
    )
    parser.add_argument("--allow-unequal-updates", action="store_true")
    parser.add_argument(
        "--promote",
        action="store_true",
        help="D-22: copy the final B2 of the alpha b2_best_alpha.txt selects to the final B2's names",
    )
    parser.add_argument("--seed", type=int, default=int(os.environ.get("TRAINING_SEED", 42)))
    args = parser.parse_args()
    if args.promote:
        if (args.alpha, args.phase, args.control, args.allow_unequal_updates) != (None, None, None, False):
            raise ValueError("--promote takes no other option: it promotes the alpha b2_best_alpha.txt selects (D-22)")
        promote(args.seed)
        return
    args.phase = args.phase or "candidate"
    control = args.control is not None
    if control and (args.phase != "final" or args.alpha is not None or args.allow_unequal_updates):
        raise ValueError(
            "--control b1plus is the final phase at alpha = 0 with B2's k (D-18): "
            "use --phase final, without --alpha or --allow-unequal-updates"
        )

    ctx = load_context(require_prepared=args.phase == "candidate")
    selection_path = None
    if control:
        args.alpha = 0.0  # w = 0 (D-18). B1+ never reads the B2 selection, so it needs only the final B1 (D-19).
    else:
        if args.alpha is None:
            if args.phase != "final":
                raise ValueError("--alpha is required for candidate training")
            selection_path = ctx.cfg.result_path(seeded_result_name("b2_best_alpha.txt", args.seed))
            args.alpha = float(selection_path.read_text(encoding="utf-8"))
        require_b2_alpha(args.alpha)  # F-05; alpha = 0 is the B1+ control (D-18)
    label = "b1plus" if control else f"b2_alpha{args.alpha}"
    model_name = "B1+" if control else "B2"
    print_context_banner(ctx, f"train_formula_{label}")
    train_S, train_dt, train_dates, train_features, training_cfg, formula_kwargs = (
        phase_training_inputs(ctx, args.phase)
    )
    # D-11: candidates start from B1_cand (Jan 1-4), the final B2 and B1+ (D-18) from the final B1.
    source_model_stem, source_vecnorm_stem = b1_artifact_stems(args.phase, args.seed)
    source_model_path = ctx.cfg.model_path(f"{source_model_stem}.zip")
    source_vecnorm_path = ctx.cfg.vecnorm_path(source_vecnorm_stem)
    complete_batch_steps = minimum_complete_day_steps(train_S)
    updates = risk_aware_updates("b2", allow_unequal=args.allow_unequal_updates)
    fill_paths = training_fill_paths()

    # D-12(c): N fill paths per day share the replayed mid and draw independent fills.
    env = build_formula_multi_day_replay_env(
        train_S,
        train_dt,
        training_cfg,
        daily_market_features=train_features,
        reward_fn=CjMmCriterion(per_step_inventory_aversion=training_cfg.phi),
        mode="sequential",
        num_trajectories=fill_paths,
        **formula_kwargs,
    )
    sb3_env = StableBaselinesTradingEnvironment(env)
    # D-15a: B1's frozen normalizer, rewards included (B1's s and clip_reward).
    vecnorm = freeze_vecnorm(source_vecnorm_path, sb3_env, training_cfg, norm_reward=True)
    # SB3 counts n_steps * num_envs per iteration, so this is exactly k iterations.
    total_timesteps = complete_batch_steps * sb3_env.num_envs * updates

    if control:
        model_stem, vecnorm_stem = final_model_stems("b1plus", args.seed)
    elif args.phase == "candidate":
        model_stem = seeded_stem(f"ppo_b2_candidate_alpha{args.alpha}_doge", args.seed)
        vecnorm_stem = seeded_stem(f"vecnorm_b2_candidate_alpha{args.alpha}", args.seed)
    elif selection_path is None:  # D-22: one of the parallel final B2s; b2_promote copies the selected one
        model_stem, vecnorm_stem = b2_final_run_stems(args.alpha, args.seed)
    else:  # the sequential refit of the selected alpha under the final names; the pipeline runs D-22's instead
        model_stem, vecnorm_stem = final_model_stems("b2", args.seed)
    ppo_kwargs = risk_aware_ppo_kwargs(training_cfg, complete_batch_steps, sb3_env.num_envs)  # D-15, D-23
    model = PPO(
        "MlpPolicy",
        vecnorm,
        **ppo_kwargs,
        tensorboard_log=str(ctx.cfg.repo_root / "tb_logs" / label),
        verbose=1,
        device="cpu",
        seed=args.seed,
    )
    reward_scale = warm_start_from(model, source_model_path, vecnorm)
    frozen_log_std = model.policy.log_std.detach().cpu().numpy().copy()
    # D-19 provenance, taken now: what this job loaded and the code it starts from. write_metadata hashes the
    # payload's files only when the job ends, days later; record_profile_stage.py checks these instead.
    loaded = {
        "warm_start_model": source_model_path,
        "warm_start_vecnorm": source_vecnorm_path,
        "train_params": train_params_json_path(ctx.cfg),
        "manifest": manifest_path(ctx.cfg),
    }
    if selection_path is not None:
        loaded["b2_selection"] = selection_path
    provenance_at_start = {
        **training_code_state(),
        "input_sha256": {name: file_sha256(path) for name, path in loaded.items()},
    }
    print(f"{model_name} warm-started from {source_model_path} with its frozen normalizer {source_vecnorm_path}")
    risk_callback = FullDayMaxDrawdownPPOCallback(
        alpha=args.alpha,
        expected_day_indices=tuple(range(len(train_dates))),
        reward_scale=reward_scale,
    )
    model.learn(total_timesteps=total_timesteps, callback=risk_callback)
    coverage = assert_complete_day_coverage(env, train_dates)

    model_path = ctx.cfg.model_path(f"{model_stem}.zip")
    vecnorm_path = ctx.cfg.vecnorm_path(vecnorm_stem)
    if not control:  # B1+ is new, with no pre-correction (direct-depth) artifact to keep
        archive_once(model_path, ctx.cfg.model_path(f"{model_stem}_direct_depth.zip"))
        archive_once(vecnorm_path, ctx.cfg.vecnorm_path(f"{vecnorm_stem}_direct_depth"))
    model.save(str(model_path))
    vecnorm.save(str(vecnorm_path))

    payload = {
        "model": "b2",
        "alpha": args.alpha,
        "phase": args.phase,
        "training_seed": args.seed,
        "result_type": "formula_as_parameter_ppo",
        "model_path": model_path,
        "vecnorm_path": vecnorm_path,
        "total_timesteps": total_timesteps,
        "risk_aware_updates": updates,
        "declared_updates": {name: os.environ.get(name) for name in ("B2_UPDATES", "B3_UPDATES")},
        "unequal_updates_override": args.allow_unequal_updates,
        "train_dates": train_dates,
        "minimum_complete_day_transitions": complete_batch_steps,
        "training_day_coverage": coverage,
        "objective": "J_beta(theta) - alpha * E[full_day_trajectory_MaxDD]",
        "fine_tuning": "D-15/D-15a/D-15b/D-15c",
        "advantage": "(A - mu_A)/sigma_A - alpha * (Z_i - leave-one-out day mean) / (s * sigma_A) (D-15a)",
        "training_fill_paths": fill_paths,
        "score_baseline": "per_day_leave_one_out",
        "formula_kwargs": formula_kwargs,
        "experiment_config": training_cfg,  # the calibrated A and kappa it trained with (item 8)
        "calibration_scope": "candidate_train_only" if args.phase == "candidate" else "all_development_days",
        "warm_start_model": source_model_path,
        "warm_start_vecnorm": source_vecnorm_path,
        "normalizer_frozen_during_b2": True,
        "reward_scale_s": reward_scale,
        "frozen_log_std": frozen_log_std.tolist(),
        "frozen_std": np.exp(frozen_log_std).tolist(),
        "policy_std": policy_std(),  # D-21: the warm start checked frozen_std against it
        "learning_rate": float(model.learning_rate),  # D-23: PPO's (Adam's) rate, FINE_TUNE_LR
        "adam_state": "fresh (D-15c)",
        "training_diagnostics": risk_callback.diagnostics,
        "ppo_kwargs": ppo_kwargs,
        "train_params": train_params_json_path(ctx.cfg),
        "manifest": manifest_path(ctx.cfg),
        "provenance_at_start": provenance_at_start,
    }
    if selection_path is not None:
        payload["b2_selection"] = selection_path  # write_metadata hashes it
    if args.phase == "final" and not control:
        payload["final_b2"] = (
            "sequential: trained after the B2 selection, under the final names" if selection_path is not None
            else "parallel (D-22): one alpha of the sweep; b2_promote copies the selected one to the final names"
        )
    if control:
        payload.update(
            model="b1plus",
            control="B1+ (D-18): the final B1 fine-tuned through the B2 path at alpha = 0",
            objective="J_beta(theta): no risk term (alpha = 0, D-18)",
            advantage="(A - mu_A)/sigma_A: the risk term is identically 0 at w = alpha = 0 (D-15a, D-18)",
        )
    write_metadata(f"train_formula_{label}", payload)
    print(f"Saved {model_name} alpha={args.alpha} model -> {model_path}")
    print(f"Saved {model_name} alpha={args.alpha} VecNormalize -> {vecnorm_path}")


def promote(seed: int) -> None:
    """b2_promote (D-22): the selected alpha's parallel final B2 becomes the final B2, by copy with SHA-256."""

    ctx = load_context(require_prepared=True)
    print_context_banner(ctx, "promote_b2")
    promotion = promote_b2(ctx.cfg, seed)
    write_metadata("promote_b2", {
        **promotion,
        "promotion_path": b2_promotion_path(ctx.cfg, seed),
        **b2_final_paths(ctx.cfg, seed),  # write_metadata hashes the promoted model and normalizer
        "training_seed": seed,
    })
    print(f"Promoted the alpha = {promotion['alpha']} final B2: {[file['to'] for file in promotion['files'].values()]}")


if __name__ == "__main__":
    main()

