from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# common puts the project root on sys.path, so it must precede the cvar_mm imports.
if __package__:
    from .common import (
        FINAL_MODEL_STEMS,
        final_model_stems,
        load_context,
        print_context_banner,
        seeded_stem,
        write_metadata,
    )
else:
    from common import (  # type: ignore[no-redef]
        FINAL_MODEL_STEMS,
        final_model_stems,
        load_context,
        print_context_banner,
        seeded_stem,
        write_metadata,
    )
from cvar_mm.gym.formula_as import FORMULA_FEATURE_NAMES
from cvar_mm.gym.notebook_support import build_formula_replay_env, freeze_vecnorm
from cvar_mm.gym.sb3_wrapper import StableBaselinesTradingEnvironment
from cvar_mm.rewards import CjMmCriterion


def resolve_model_paths(cfg, model: str, seed: int) -> tuple[Path, Path, str]:
    """A final model's files under its seeded stems, with an explicit .zip (WORK_ORDER item 8, M5: SB3 would
    otherwise read an extension-less stale file first), and the prefix of the output name."""

    model_stem, vecnorm_stem = final_model_stems(model, seed)  # raises for any other model
    return cfg.model_path(f"{model_stem}.zip"), cfg.vecnorm_path(vecnorm_stem), seeded_stem(model, seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optional SHAP diagnostics for formula-policy PPO models."
    )
    parser.add_argument("--model", choices=sorted(FINAL_MODEL_STEMS), default="b1")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("TRAINING_SEED", 42)))
    parser.add_argument("--samples", type=int, default=150)
    parser.add_argument("--background", type=int, default=30)
    parser.add_argument("--nsamples", type=int, default=100)
    args = parser.parse_args()

    try:
        import shap
    except ImportError:
        load_context()
        write_metadata(
            f"shap_diagnostics_skipped_{args.model}",
            {"reason": "shap_not_installed", "model": args.model, "training_seed": args.seed},
        )
        print("SHAP is not installed; skipping diagnostics without failing the pipeline.")
        return

    ctx = load_context()
    print_context_banner(ctx, f"run_shap_diagnostics_{args.model}")

    model_path, vecnorm_path, tag = resolve_model_paths(ctx.cfg, args.model, args.seed)

    row = ctx.train_manifest.iloc[0]

    from cvar_mm.gym.data_loader import load_single_day_with_features

    S, dt, _, features = load_single_day_with_features(
        str(row["input_path"]),
        tick_size=ctx.cfg.tick_size,
    )

    env = build_formula_replay_env(
        S,
        dt,
        ctx.cfg,
        sigma=ctx.formula_kwargs["sigma"],
        market_features=features,
        reward_fn=CjMmCriterion(per_step_inventory_aversion=ctx.cfg.phi),
        **{k: v for k, v in ctx.formula_kwargs.items() if k != "sigma"},
    )

    sb3_env = StableBaselinesTradingEnvironment(env)

    print(f"Loading VecNormalize: {vecnorm_path}")
    eval_vn = freeze_vecnorm(
        vecnorm_path,
        sb3_env,
        ctx.cfg,
        norm_reward=False,
    )

    print(f"Loading PPO model: {model_path}")
    model = PPO.load(str(model_path), device="cpu")

    obs, _ = env.reset(seed=ctx.cfg.evaluation_seed)

    observations = []
    for _ in range(args.samples):
        observations.append(obs[0].copy())

        norm_obs = eval_vn.normalize_obs(obs)[0]
        action, _ = model.predict(norm_obs, deterministic=True)

        obs, _, terminated, _, _ = env.step(action.reshape(1, -1))
        if bool(terminated[0]):
            break

    X = np.asarray(observations, dtype=np.float32)

    if len(X) <= args.background:
        raise RuntimeError(
            f"Not enough observations collected for SHAP diagnostics: "
            f"len(X)={len(X)}, background={args.background}"
        )

    background = X[: args.background]
    explain = X[args.background :]

    def predict_fn(raw_x):
        raw_x = np.asarray(raw_x, dtype=np.float32)
        norm_x = eval_vn.normalize_obs(raw_x)
        actions = [model.predict(row, deterministic=True)[0] for row in norm_x]
        return np.asarray(actions)

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(explain, nsamples=args.nsamples)

    if isinstance(shap_values, list):
        values = np.stack(shap_values, axis=0)
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            values = np.moveaxis(values, -1, 0)

    if values.ndim != 3:
        raise RuntimeError(f"Unexpected SHAP value shape: {values.shape}")

    base_output_names = ["u_gamma", "u_skew"]
    output_names = [
        base_output_names[i] if i < len(base_output_names) else f"action_{i}"
        for i in range(values.shape[0])
    ]

    rows = []
    for output_idx, output_name in enumerate(output_names):
        mean_abs = np.abs(values[output_idx]).mean(axis=0)

        for feature, score in zip(FORMULA_FEATURE_NAMES, mean_abs):
            rows.append(
                {
                    "model": tag,
                    "output": output_name,
                    "feature": feature,
                    "mean_abs_shap": float(score),
                    "samples": int(len(X)),
                    "background": int(args.background),
                    "nsamples": int(args.nsamples),
                }
            )

    out_path = ctx.cfg.result_path(f"{tag}_shap_feature_importance.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)

    write_metadata(
        f"shap_diagnostics_{tag}",
        {
            "model": args.model,
            "training_seed": args.seed,
            "model_path": model_path,  # write_metadata hashes the files it explained
            "vecnorm_path": vecnorm_path,
            "experiment_config": ctx.cfg,  # the Jan 1-6 A and kappa of its replay (item 8)
            "output_path": str(out_path),
            "samples": len(X),
            "background": args.background,
            "nsamples": args.nsamples,
            "n_explain": int(len(explain)),
            "features": list(FORMULA_FEATURE_NAMES),
            "outputs": output_names,
        },
    )

    print(f"Saved SHAP feature importance -> {out_path}")


if __name__ == "__main__":
    main()
