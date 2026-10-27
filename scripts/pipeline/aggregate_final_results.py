from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

# common puts the project root on sys.path, so it must precede the cvar_mm imports.
if __package__:
    from .common import (
        b2_alpha_result_path,
        eval_episode_path,
        eval_part_path,
        load_context,
        parse_alpha_list,
        seeded_result_name,
        write_metadata,
    )
else:
    from common import (  # type: ignore[no-redef]
        b2_alpha_result_path,
        eval_episode_path,
        eval_part_path,
        load_context,
        parse_alpha_list,
        seeded_result_name,
        write_metadata,
    )
from cvar_mm.gym.cvar_lagrangian import B3_TAIL_MASS, empirical_ru_cvar
from cvar_mm.gym.model_selection import choose_b2_alpha, require_b2_alpha

SUMMARY_METRICS = ("Sharpe", "Sortino", "Max DD", "P&L-to-MAP", "Final PnL", "Mean |q|")
# The days-best table compares B0-B3. The B1+ control (D-18) gets a summary row only, so adding it
# cannot change B0-B3's counts (the author's choice of 2026-10-07).
COMPARISON_MODELS = ("B0", "B1", "B2", "B3")
CONTROL_MODELS = ("B1PLUS",)
DAYS_BEST_METRICS = {
    "Sharpe": True,
    "Sortino": True,
    "Max DD": False,
    "P&L-to-MAP": True,
}


def _read_part(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "Day" not in frame.columns:
        raise ValueError(f"Evaluation part missing Day column: {path}")
    return frame


def aggregate_model_parts(
    *,
    model: str,
    alpha: float | None = None,
    allow_partial: bool = False,
    split: str = "retrospective_holdout",
    training_seed: int = 42,
) -> pd.DataFrame:
    if model == "b1plus" and split != "retrospective_holdout":
        raise ValueError("B1+ (D-18) is evaluated on the retrospective holdout only: it trained on Jan 1-6")
    ctx = load_context()
    frames = []
    missing = []
    manifest = ctx.validation_manifest if split == "validation" else ctx.test_manifest
    for row in manifest.itertuples(index=False):
        path = eval_part_path(
            ctx.cfg,
            model=model,
            alpha=alpha,
            date=str(row.date),
            split=split,
            training_seed=training_seed,
        )
        if path.exists():
            frames.append(_read_part(path))
        else:
            missing.append((row.split_index, row.date, path))

    if missing and not allow_partial:
        missing_text = ", ".join(f"{idx}:{date}" for idx, date, _ in missing)
        raise FileNotFoundError(f"Missing evaluation parts for {model}: {missing_text}")
    if not frames:
        raise FileNotFoundError(f"No evaluation parts found for model={model}, alpha={alpha}")

    df = pd.concat(frames, ignore_index=True).sort_values("Day").reset_index(drop=True)
    if model == "b1" and split == "retrospective_holdout":
        out_path = ctx.cfg.result_path(seeded_result_name("b1_test_results.csv", training_seed))
    elif model == "b1plus" and split == "retrospective_holdout":
        out_path = ctx.cfg.result_path(seeded_result_name("b1plus_test_results.csv", training_seed))
    elif model == "b2" and split == "validation":
        if alpha is None:
            raise ValueError("alpha is required to aggregate B2 validation results")
        out_path = b2_alpha_result_path(ctx.cfg, alpha, training_seed)
    elif model == "b2" and split == "retrospective_holdout":
        out_path = ctx.cfg.result_path(seeded_result_name("b2_test_results.csv", training_seed))
    elif model == "b3" and split == "retrospective_holdout":
        out_path = ctx.cfg.result_path(seeded_result_name("b3_test_results.csv", training_seed))
    else:
        raise ValueError(f"Unknown model: {model}")
    df.to_csv(out_path, index=False)
    episodes_path = None
    if split == "retrospective_holdout":
        episodes_path = aggregate_episode_parts(
            ctx, manifest, model=model, allow_partial=allow_partial, training_seed=training_seed
        )
    write_metadata(
        f"aggregate_{model}" if alpha is None else f"aggregate_{model}_alpha{alpha}",
        {
            "model": model,
            "alpha": alpha,
            "output_path": out_path,
            "episodes_path": episodes_path,
            "rows": len(df),
            "allow_partial": allow_partial,
            "split": split,
            "training_seed": training_seed,
            "missing": [{"day_index": idx, "date": date, "path": path} for idx, date, path in missing],
            "result_type": "formula_as_parameter_ppo",
        },
    )
    print(f"Saved aggregated {model} results -> {out_path}")
    return df


def aggregate_episode_parts(
    ctx, manifest: pd.DataFrame, *, model: str, allow_partial: bool, training_seed: int
) -> Path:
    """Concatenate the per-path episode parts of the retrospective days (K6)."""

    frames, missing = [], []
    for row in manifest.itertuples(index=False):
        path = eval_episode_path(ctx.cfg, model=model, date=str(row.date), training_seed=training_seed)
        if path.exists():
            frames.append(pd.read_csv(path))
        else:
            missing.append(str(row.date))
    if (missing and not allow_partial) or not frames:
        raise FileNotFoundError(f"Missing episode parts for {model}: {', '.join(missing)}")
    episodes = pd.concat(frames, ignore_index=True)
    if episodes.groupby("Day").size().nunique() != 1:
        raise ValueError(f"{model} episode parts have unequal numbers of paths per day")
    out_path = ctx.cfg.result_path(seeded_result_name(f"{model}_test_episodes.csv", training_seed))
    episodes.to_csv(out_path, index=False)
    return out_path


def select_b2(alphas: list[float], training_seed: int = 42) -> float:
    for alpha in alphas:  # before anything is read: the B1+ control (alpha = 0) is never a candidate (D-18)
        require_b2_alpha(alpha)
    ctx = load_context()
    rows = []
    for alpha in alphas:
        path = b2_alpha_result_path(ctx.cfg, alpha, training_seed)
        if not path.exists():
            aggregate_model_parts(
                model="b2", alpha=alpha, split="validation", training_seed=training_seed
            )
        df = pd.read_csv(path)
        rows.append(
            {
                "alpha": alpha,
                "mean_sharpe": float(df["Sharpe"].mean()),
                "mean_maxdd": float(df["Max DD"].mean()),
                "mean_pnl": float(df["Final PnL"].mean()),
            }
        )

    summary = pd.DataFrame(rows)
    best_alpha = choose_b2_alpha(summary)
    ctx.cfg.result_path(seeded_result_name("b2_best_alpha.txt", training_seed)).write_text(
        str(best_alpha), encoding="utf-8"
    )
    summary.set_index("alpha").to_csv(ctx.cfg.result_path(
        seeded_result_name("b2_alpha_sweep_summary.csv", training_seed)
    ))
    write_metadata(
        "select_b2",
        {
            "best_alpha": best_alpha,
            "alpha_summary": summary.to_dict(orient="records"),
            "selection_split": "validation",
            "training_seed": training_seed,
            "next_required_step": "b2_promote: copy the selected alpha's final B2, trained on all six development "
                                  "days in parallel with the candidates, to the final B2 names (D-22)",
        },
    )
    print("=== B2 alpha summary ===")
    print(summary.set_index("alpha").to_string(float_format="{:.6f}".format))
    print(f"Selected B2 alpha={best_alpha}")
    return best_alpha


def write_comparison(training_seed: int = 42) -> None:
    ctx = load_context()
    paths = {
        name: ctx.cfg.result_path(seeded_result_name(f"{name.lower()}_test_results.csv", training_seed))
        for name in (*COMPARISON_MODELS, *CONTROL_MODELS)
    }
    episode_paths = {
        name: ctx.cfg.result_path(seeded_result_name(f"{name.lower()}_test_episodes.csv", training_seed))
        for name in paths
    }
    missing = [
        f"{name}:{path}"
        for name, path in (*paths.items(), *episode_paths.items())
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing comparison inputs. " + ", ".join(missing))

    dfs = {name: pd.read_csv(path).sort_values("Day").reset_index(drop=True) for name, path in paths.items()}
    base_dates = dfs["B0"]["Day"].astype(str).tolist()
    for name, df in dfs.items():
        dates = df["Day"].astype(str).tolist()
        if dates != base_dates:
            raise ValueError(f"{name} dates do not align with B0.")

    rows = []
    for metric, higher_better in DAYS_BEST_METRICS.items():
        values = pd.DataFrame({name: dfs[name][metric] for name in COMPARISON_MODELS})
        winners = values.idxmax(axis=1) if higher_better else values.idxmin(axis=1)
        counts = winners.value_counts().reindex(COMPARISON_MODELS, fill_value=0)
        row = {"Metric": metric}
        row.update(counts.to_dict())
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison_path = ctx.cfg.result_path(seeded_result_name("comparison_table.csv", training_seed))
    comparison.to_csv(comparison_path, index=False)

    summary_rows = []
    for name, df in dfs.items():
        row = {"Model": name}
        for metric in SUMMARY_METRICS:
            if metric in df:
                row[f"{metric} Mean"] = float(df[metric].mean())
                row[f"{metric} Median"] = float(df[metric].median())
        episodes = pd.read_csv(episode_paths[name])
        if sorted(set(episodes["Day"].astype(str))) != sorted(base_dates):
            raise ValueError(f"{name} episodes do not cover the comparison days.")
        # K6: CVaR_0.2 of daily MaxDD over seed-averaged days and over full-day episodes.
        row["Max DD CVaR0.2 (days)"] = empirical_ru_cvar(df["Max DD"].to_numpy(float), rho=B3_TAIL_MASS).value
        row["Max DD CVaR0.2 (episodes)"] = empirical_ru_cvar(
            episodes["Max DD"].to_numpy(float), rho=B3_TAIL_MASS
        ).value
        row["Episodes"] = len(episodes)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary_path = ctx.cfg.result_path(seeded_result_name(
        "overall_summary_table.csv", training_seed
    ))
    summary.to_csv(summary_path, index=False)

    write_metadata(
        "final_comparison",
        {
            "comparison_table": comparison_path,
            "comparison_models": COMPARISON_MODELS,
            "overall_summary": summary_path,
            "summary_only_models": CONTROL_MODELS,
            "results_paths": paths,  # the day-level inputs, hashed by write_metadata like the episodes
            "episode_paths": episode_paths,
            "dates": base_dates,
            "training_seed": training_seed,
            "result_type": "formula_as_parameter_ppo_comparison",
        },
    )
    print(f"Saved comparison table -> {comparison_path}")
    print(f"Saved overall summary -> {summary_path}")


def aggregate_ablation() -> None:
    ctx = load_context()
    rows = []
    for path in sorted((ctx.cfg.results_dir / "ablation").glob("*_validation_results.csv")):
        feature_set = path.name.replace("_validation_results.csv", "")
        df = pd.read_csv(path)
        row = {"feature_set": feature_set}
        for metric in SUMMARY_METRICS:
            if metric in df:
                row[f"mean_{metric}"] = float(df[metric].mean())
        rows.append(row)
    if not rows:
        raise FileNotFoundError("No ablation validation files found.")
    out = pd.DataFrame(rows)
    out.to_csv(ctx.cfg.result_path("formula_feature_ablation_results.csv"), index=False)
    print(f"Saved ablation summary -> {ctx.cfg.result_path('formula_feature_ablation_results.csv')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate formula-AS Snellius outputs.")
    parser.add_argument("--model", choices=("b1", "b1plus", "b2", "b3"), default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--split", choices=("validation", "retrospective_holdout"),
        default="retrospective_holdout",
    )
    parser.add_argument("--select-b2", action="store_true")
    parser.add_argument("--alphas", default=None)
    parser.add_argument("--comparison", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument(
        "--training-seed", type=int, default=int(os.environ.get("TRAINING_SEED", 42))
    )
    args = parser.parse_args()

    if args.model:
        aggregate_model_parts(
            model=args.model,
            alpha=args.alpha,
            allow_partial=args.allow_partial,
            split=args.split,
            training_seed=args.training_seed,
        )
    if args.select_b2:
        select_b2(parse_alpha_list(args.alphas), training_seed=args.training_seed)
    if args.comparison:
        write_comparison(training_seed=args.training_seed)
    if args.ablation:
        aggregate_ablation()
    if not any([args.model, args.select_b2, args.comparison, args.ablation]):
        raise SystemExit("No action selected. Use --model, --select-b2, --comparison, or --ablation.")


if __name__ == "__main__":
    main()

