from __future__ import annotations

import argparse

from common import (
    DEFAULT_EXPECTED_TEST_DAYS,
    DEFAULT_TRAIN_DAYS,
    load_context,
    manifest_path,
    print_context_banner,
    train_params_csv_path,
    train_params_json_path,
    write_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare formula-AS Snellius manifest and train calibration.")
    parser.add_argument("--force", action="store_true", help="Rebuild manifest and train calibration files.")
    args = parser.parse_args()

    ctx = load_context(force_prepare=args.force)
    print_context_banner(ctx, "prepare_formula_pipeline")

    if len(ctx.train_manifest) != DEFAULT_TRAIN_DAYS or len(ctx.test_manifest) != DEFAULT_EXPECTED_TEST_DAYS:
        raise ValueError(
            f"Expected {DEFAULT_TRAIN_DAYS} train and {DEFAULT_EXPECTED_TEST_DAYS} test days, "
            f"found {len(ctx.train_manifest)} and {len(ctx.test_manifest)}."
        )

    print(f"Manifest       : {manifest_path(ctx.cfg)}")
    print(f"Train params   : {train_params_json_path(ctx.cfg)}")
    print(f"Train params CSV: {train_params_csv_path(ctx.cfg)}")

    write_metadata(
        "prepare_formula_pipeline",
        {
            "manifest": manifest_path(ctx.cfg),
            "train_params_json": train_params_json_path(ctx.cfg),
            "train_params_csv": train_params_csv_path(ctx.cfg),
            "train_dates": ctx.train_manifest["date"].tolist(),
            "candidate_train_dates": ctx.candidate_train_manifest["date"].tolist(),
            "validation_dates": ctx.validation_manifest["date"].tolist(),
            "test_dates": ctx.test_manifest["date"].tolist(),
            "test_role": "frozen_retrospective_holdout_evaluation_only",
            "later_untouched_holdout_dates": ctx.later_holdout_manifest["date"].tolist(),
            "input_sha256": ctx.manifest[["date", "input_sha256"]].to_dict(orient="records"),
            "formula_kwargs": ctx.formula_kwargs,
            "experiment_config": ctx.cfg,  # with the Jan 1-6 A and kappa this job calibrated (item 8)
        },
    )


if __name__ == "__main__":
    main()

