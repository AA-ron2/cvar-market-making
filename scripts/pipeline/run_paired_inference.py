from __future__ import annotations

import argparse
import os

import pandas as pd

if __package__:
    from .common import load_context, seeded_result_name, write_metadata
else:
    from common import load_context, seeded_result_name, write_metadata  # type: ignore[no-redef]
from cvar_mm.gym.statistical_inference import (
    PRIMARY_CONTRASTS,
    PRIMARY_DESCRIPTIVE_CONTRASTS,
    RANDOMIZATION_DRAWS,
    SECONDARY_CONTRASTS,
    SECONDARY_DESCRIPTIVE_CONTRASTS,
    SECONDARY_METRICS,
    paired_inference_table,
    primary_cvar_inference_table,
)

# B1+ is the D-18 control: it enters descriptive rows only.
DAY_LEVEL_MODELS = ("B0", "B1", "B2", "B3", "B1PLUS")
PATH_LEVEL_MODELS = ("B1", "B2", "B3")  # the primary contrasts' path-level sensitivity rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Day-paired corrected-model inference (DECISIONS D-14, D-14a, D-16, D-18)."
    )
    parser.add_argument("--primary-block-length-days", type=int, default=3)
    parser.add_argument("--sensitivity-block-lengths", default="1 2 4")
    parser.add_argument("--block-length-days", type=int, default=5, help="secondary family")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--randomization-draws", type=int, default=RANDOMIZATION_DRAWS)
    parser.add_argument("--seed", type=int, default=42, help="seeds both the randomization draws and the bootstrap")
    parser.add_argument(
        "--training-seed", type=int, default=int(os.environ.get("TRAINING_SEED", 42))
    )
    args = parser.parse_args()
    ctx = load_context()

    def result(filename: str):
        return ctx.cfg.result_path(seeded_result_name(filename, args.training_seed))

    day_paths = {name: result(f"{name.lower()}_test_results.csv") for name in DAY_LEVEL_MODELS}
    episode_paths = {name: result(f"{name.lower()}_test_episodes.csv") for name in PATH_LEVEL_MODELS}
    day_frames = {name: pd.read_csv(path) for name, path in day_paths.items()}
    episode_frames = {name: pd.read_csv(path) for name, path in episode_paths.items()}
    sensitivity = tuple(int(part) for part in args.sensitivity_block_lengths.split())

    primary = primary_cvar_inference_table(
        day_frames,
        episode_frames,
        block_length=args.primary_block_length_days,
        sensitivity_block_lengths=sensitivity,
        repetitions=args.bootstrap_repetitions,
        draws=args.randomization_draws,
        seed=args.seed,
    )
    secondary = paired_inference_table(
        day_frames,
        block_length=args.block_length_days,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    primary_path = result("paired_day_inference_primary.csv")
    secondary_path = result("paired_day_inference.csv")
    primary.to_csv(primary_path, index=False)
    secondary.to_csv(secondary_path, index=False)
    write_metadata("paired_day_inference", {
        "primary_output_path": primary_path,
        "secondary_output_path": secondary_path,
        "input_paths": day_paths,
        "episode_input_paths": episode_paths,
        "primary_sampling_unit": "market_day",
        "primary_family": {
            "contrasts": PRIMARY_CONTRASTS,
            "statistic": "CVaR_0.2 of daily MaxDD over the seed-averaged days (D-14)",
            "test": "within-day label-swap randomization, two-sided, p = (1 + #{|D*| >= |D|}) / (draws + 1); "
                    "Holm over the 2 primary rows (D-14a)",
            "randomization_draws": args.randomization_draws,
            "interval": "paired circular block bootstrap CI, unadjusted 95 %, descriptive (D-14a, M3)",
            "block_length_days": args.primary_block_length_days,
            "sensitivity": {
                "block_lengths": sensitivity,
                "episode_level": "path-level CVaR over every full-day path, randomization with whole-day swaps, "
                                 "CI at the primary block length",
            },
        },
        "descriptive_rows": {
            "primary_statistic": PRIMARY_DESCRIPTIVE_CONTRASTS,
            "secondary_metrics": SECONDARY_DESCRIPTIVE_CONTRASTS,
            "note": "descriptive, not multiplicity-adjusted; outside both Holm families (D-16, D-18)",
        },
        "secondary_family": {
            "contrasts": SECONDARY_CONTRASTS,
            "metrics": SECONDARY_METRICS,
            "test": "Pratt-Wilcoxon; Hodges-Lehmann effect with its exact signed-rank 95 % CI (Bauer 1972, M4); "
                    "block-bootstrap CI of the mean; its own Holm",
            "block_length_days": args.block_length_days,
        },
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "seed": args.seed,
        "training_seed": args.training_seed,
    })
    print(f"Saved primary CVaR inference -> {primary_path}")
    print(f"Saved secondary paired inference -> {secondary_path}")


if __name__ == "__main__":
    main()
