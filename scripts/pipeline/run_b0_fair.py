from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from common import (
    load_context,
    parse_seed_list,
    primary_b0_gamma_range,
    print_context_banner,
    require_gamma_within_bounds,
    seeded_result_name,
    write_metadata,
)
from cvar_mm.gym.calibration import tune_gamma
from cvar_mm.gym.data_loader import load_single_day
from cvar_mm.gym.helpers.fast_rollout import fast_simulate
from cvar_mm.gym.notebook_support import stats_dict_to_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fair train-only B0 analytical A-S baseline.")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("TRAINING_SEED", 42)))
    parser.add_argument("--fill-seeds", default=None)
    args = parser.parse_args()

    ctx = load_context()
    print_context_banner(ctx, "run_b0_fair")
    gamma_range = primary_b0_gamma_range(ctx.cfg, ctx.formula_kwargs)

    gamma_candidates = []
    train_rows = []
    for row in ctx.train_manifest.itertuples(index=False):
        S, dt, _ = load_single_day(row.input_path)
        best_gamma, _ = tune_gamma(
            S,
            dt,
            sigma=float(ctx.train_params["sigma_train"]),
            kappa=float(ctx.train_params["kappa_train"]),
            A=float(ctx.train_params["A_train"]),
            tick_size=ctx.cfg.tick_size,
            Q_MAX=ctx.cfg.q_max,
            gamma_range=gamma_range,
            n_trials=ctx.cfg.as_gamma_trials,
            num_trajectories=ctx.cfg.as_gamma_num_trajectories,
            seed=args.seed,
            verbose=False,
        )
        best_gamma = require_gamma_within_bounds(best_gamma, gamma_range)
        gamma_candidates.append(best_gamma)
        train_rows.append(
            {
                "Day": row.date,
                "sigma": float(ctx.train_params["sigma_train"]),
                "A": float(ctx.train_params["A_train"]),
                "kappa": float(ctx.train_params["kappa_train"]),
                "gamma_candidate": best_gamma,
                "gamma_min": gamma_range[0],
                "gamma_max": gamma_range[1],
            }
        )
        print(f"{row.date}: gamma={best_gamma:.6f}")

    df_train = pd.DataFrame(train_rows)
    gamma_fixed = float(np.mean(gamma_candidates))
    sigma_train = float(ctx.train_params["sigma_train"])
    A_train = float(ctx.train_params["A_train"])
    kappa_train = float(ctx.train_params["kappa_train"])

    df_train.to_csv(ctx.cfg.result_path(
        seeded_result_name("b0_train_calibration_params.csv", args.seed)
    ), index=False)
    ctx.cfg.result_path(seeded_result_name("b0_gamma_fixed.txt", args.seed)).write_text(
        str(gamma_fixed), encoding="utf-8"
    )

    fill_seeds = parse_seed_list(args.fill_seeds)
    rows = []
    episode_frames = []
    for row in ctx.test_manifest.itertuples(index=False):
        S, dt, _ = load_single_day(row.input_path)
        seed_stats = [fast_simulate(
            midprices=S, dt_array=dt, gamma=gamma_fixed, sigma=sigma_train,
            kappa=kappa_train, A=A_train, terminal_time=float(dt.sum()),
            tick_size=ctx.cfg.tick_size, Q_MAX=ctx.cfg.q_max,
            num_trajectories=ctx.cfg.evaluation_rollouts, seed=fill_seed,
            use_linear_approximation=False,
            fixed_clock_seconds=ctx.cfg.fixed_clock_seconds,
        ) for fill_seed in fill_seeds]
        for fill_seed, stats in zip(fill_seeds, seed_stats):
            frame = stats_dict_to_frame(stats)
            episode_frames.append(
                frame.assign(Day=str(row.date), Rollout=np.arange(len(frame)), **{"Fill Seed": fill_seed})
            )
        def mean_metric(name: str) -> float:
            return float(np.concatenate([np.asarray(stats[name]) for stats in seed_stats]).mean())
        rows.append(
            {
                "Day": str(row.date),
                "Sharpe": mean_metric("sharpe"),
                "Sortino": mean_metric("sortino"),
                "Max DD": mean_metric("max_drawdown"),
                "P&L-to-MAP": mean_metric("pnl_to_map"),
                "Final PnL": mean_metric("total_pnl"),
                "Mean |q|": mean_metric("mean_abs_q"),
                "Near Cap Fraction": mean_metric("near_cap_fraction"),
                "Sharpe Snapshot (supplementary)": mean_metric("snapshot_sharpe"),
                "Sortino Snapshot (supplementary)": mean_metric("snapshot_sortino"),
                "Rollouts": float(ctx.cfg.evaluation_rollouts * len(fill_seeds)),
                "Fill Seeds": ",".join(str(seed) for seed in fill_seeds),
                "Training Seed": args.seed,
                "Result Type": "train_only_fair_b0",
            }
        )
        print(f"{row.date}: Sharpe={rows[-1]['Sharpe']:.4f}, MaxDD={rows[-1]['Max DD']:.6f}")

    df = pd.DataFrame(rows)
    out_path = ctx.cfg.result_path(seeded_result_name("b0_test_results.csv", args.seed))
    df.to_csv(out_path, index=False)
    # One row per full-day path (fill seed x rollout) for the episode-level CVaR (K6).
    episodes_path = ctx.cfg.result_path(seeded_result_name("b0_test_episodes.csv", args.seed))
    episodes = pd.concat(episode_frames, ignore_index=True)
    leading = ["Day", "Fill Seed", "Rollout"]
    episodes[leading + [column for column in episodes.columns if column not in leading]].to_csv(
        episodes_path, index=False
    )
    write_metadata(
        "run_b0_fair",
        {
            "output_path": out_path,
            "episodes_path": episodes_path,
            "gamma_fixed": gamma_fixed,
            "gamma_tuning_bounds": gamma_range,
            "sigma_train": sigma_train,
            "A_train": A_train,
            "kappa_train": kappa_train,
            "result_type": "train_only_fair_b0",
            "training_seed": args.seed,
            "fill_seeds": fill_seeds,
            "experiment_config": ctx.cfg,  # A and kappa equal A_train and kappa_train above (item 8)
        },
    )
    print(f"Saved B0 results -> {out_path}")


if __name__ == "__main__":
    main()

