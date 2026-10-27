"""Profile fixed A--S calibration candidates on January 1--4 only.

This utility deliberately does not call ``load_context``: that path may create
the canonical January 1--6 production calibration artifacts.  All files from
this module are confined to ``results/profiling/calibration``.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__:
    from .common import (
        build_config,
        get_git_commit,
        load_or_prepare_manifest,
        require_snellius_paths,
    )
else:
    from common import (  # type: ignore[no-redef]
        build_config,
        get_git_commit,
        load_or_prepare_manifest,
        require_snellius_paths,
    )

from cvar_mm.gym.calibration import (
    FirstPassageCalibrationConfig,
    IntensityCurve,
    collect_censored_first_passage,
    contiguous_supported_depth_prefix,
    empirical_monotonicity_fraction,
    fit_exponential_intensity,
    sum_intensity_curves,
    valid_fit_depth_mask,
)
from cvar_mm.gym.data_loader import load_single_day_calibration_arrays


PROFILE_DATES = ("2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04")
PROFILE_DEPTH_TICKS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0)
PROFILE_CENSOR_SECONDS = 15.0
PROFILE_MIN_HITS = 10
PROFILE_CANDIDATES = (
    ("midpoint_first_passage__non_overlapping", "midpoint_first_passage", "non_overlapping"),
    ("midpoint_first_passage__fixed_stride_15s", "midpoint_first_passage", "fixed_stride"),
    ("best_price_reach__non_overlapping", "best_price_reach", "non_overlapping"),
    ("best_price_reach__fixed_stride_15s", "best_price_reach", "fixed_stride"),
)
PROFILE_FILENAMES = {
    "summary": "calibration_profile_summary.csv",
    "depths": "calibration_profile_depths.csv",
    "per_day": "calibration_profile_per_day.csv",
    "json": "calibration_profile.json",
}
CANONICAL_CALIBRATION_FILENAMES = {
    "formula_train_params.json",
    "formula_train_params.csv",
}


def select_profiling_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return exactly Jan 1--4 candidate-training rows, or fail closed."""

    required = {"date", "split", "usage", "input_path", "input_sha256"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Profiling manifest missing columns: {sorted(missing)}")

    frame = manifest.copy()
    frame["date"] = frame["date"].astype(str)
    selected = frame[frame["date"].isin(PROFILE_DATES)].sort_values("date").reset_index(drop=True)
    if tuple(selected["date"]) != PROFILE_DATES:
        raise ValueError("Calibration profiling requires exactly one row for each of Jan 1--4, 2025")
    if not (selected["usage"] == "candidate_train").all() or not (selected["split"] == "train").all():
        raise ValueError("Calibration profiling dates must all be labelled candidate_train/train")

    declared_candidates = tuple(
        sorted(frame.loc[frame["usage"] == "candidate_train", "date"].tolist())
    )
    if declared_candidates != PROFILE_DATES:
        raise ValueError("The manifest candidate-training set must be exactly Jan 1--4, 2025")
    return selected


def profiling_output_paths(results_dir: str | Path) -> dict[str, Path]:
    """Return fixed profile paths and verify they cannot alias canonical files."""

    results = Path(results_dir).resolve()
    output_dir = results / "profiling" / "calibration"
    paths = {name: output_dir / filename for name, filename in PROFILE_FILENAMES.items()}
    canonical = {(results / name).resolve() for name in CANONICAL_CALIBRATION_FILENAMES}
    if any(path.resolve() in canonical for path in paths.values()):
        raise RuntimeError("Profiling output may not overwrite canonical calibration artifacts")
    return paths


def _candidate_config(
    *, execution_proxy: str, exposure_start_rule: str, tick_size: float
) -> FirstPassageCalibrationConfig:
    return FirstPassageCalibrationConfig(
        tick_size=tick_size,
        depth_ticks=PROFILE_DEPTH_TICKS,
        censor_horizon_seconds=PROFILE_CENSOR_SECONDS,
        execution_proxy=execution_proxy,
        exposure_start_rule=exposure_start_rule,
        exposure_stride_seconds=PROFILE_CENSOR_SECONDS,
        min_hits_per_depth=PROFILE_MIN_HITS,
    )


def _fit_diagnostics(curve: IntensityCurve, minimum: int) -> tuple[dict[str, Any], Any]:
    valid = valid_fit_depth_mask(curve, minimum)
    prefix = contiguous_supported_depth_prefix(curve, min_hits_per_depth=minimum)
    total_exposures = int(np.sum(curve.exposures))
    total_censored = int(np.sum(curve.censored))
    base: dict[str, Any] = {
        "side": curve.side,
        "fit_valid": False,
        "fit_error": None,
        "A": None,
        "kappa": None,
        "weighted_log_r2": None,
        "weighted_log_rmse": None,
        "valid_depth_count": int(valid.sum()),
        "empirical_monotonicity_fraction": empirical_monotonicity_fraction(
            curve, min_hits_per_depth=minimum
        ),
        "supported_prefix_count": int(prefix.size),
        "supported_prefix_max_depth_ticks": None if prefix.size == 0 else float(prefix[-1]),
        "supported_prefix_depth_ticks": prefix.tolist(),
        "total_hits": int(np.sum(curve.hits)),
        "total_exposure_seconds": float(np.sum(curve.total_exposure_seconds)),
        "total_exposures": total_exposures,
        "total_censored": total_censored,
        "overall_censoring_fraction": (
            float(total_censored / total_exposures) if total_exposures else None
        ),
    }
    try:
        fit = fit_exponential_intensity(curve, min_hits_per_depth=minimum)
    except ValueError as exc:
        base["fit_error"] = str(exc)
        return base, None

    base.update(
        {
            "fit_valid": bool(
                np.isfinite(fit.A) and fit.A > 0 and np.isfinite(fit.kappa) and fit.kappa > 0
            ),
            "A": fit.A,
            "kappa": fit.kappa,
            "weighted_log_r2": fit.r_squared_log,
            "weighted_log_rmse": fit.rmse_log,
            "valid_depth_count": fit.valid_depths,
        }
    )
    return base, fit


def _depth_rows(
    candidate_id: str,
    curve: IntensityCurve,
    minimum: int,
    *,
    execution_proxy: str,
    exposure_start_rule: str,
) -> list[dict[str, Any]]:
    valid = valid_fit_depth_mask(curve, minimum)
    return [
        {
            "candidate_id": candidate_id,
            "execution_proxy": execution_proxy,
            "exposure_start_rule": exposure_start_rule,
            "side": curve.side,
            "depth_ticks": float(curve.depth_ticks[index]),
            "depth_price": float(curve.depth_price[index]),
            "hits": int(curve.hits[index]),
            "total_exposure_seconds": float(curve.total_exposure_seconds[index]),
            "exposures": int(curve.exposures[index]),
            "censored_count": int(curve.censored[index]),
            "censoring_fraction": float(curve.censoring_fraction[index]),
            "empirical_intensity_per_second": float(curve.intensity_per_second[index]),
            "valid_fit_depth": bool(valid[index]),
        }
        for index in range(len(curve.depth_ticks))
    ]


def _add_bid_ask_asymmetry(rows: list[dict[str, Any]]) -> None:
    by_side = {row["side"]: row for row in rows}
    bid, ask = by_side["bid"], by_side["ask"]
    A_ratio = None
    kappa_ratio = None
    if bid["fit_valid"] and ask["fit_valid"]:
        A_ratio = float(bid["A"] / ask["A"])
        kappa_ratio = float(bid["kappa"] / ask["kappa"])
    for row in rows:
        row["bid_ask_A_ratio"] = A_ratio
        row["bid_ask_kappa_ratio"] = kappa_ratio


def build_profile_tables(
    manifest: pd.DataFrame,
    *,
    tick_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Run the four fixed candidate fits and return serialisable diagnostics."""

    selected = select_profiling_manifest(manifest)
    days: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        mid, dt, _, bid, ask = load_single_day_calibration_arrays(row.input_path)
        days.append({"date": str(row.date), "mid": mid, "dt": dt, "bid": bid, "ask": ask})
        print(f"Loaded profiling day {row.date}: {len(mid):,} snapshots")

    summary_rows: list[dict[str, Any]] = []
    depth_rows: list[dict[str, Any]] = []
    per_day_rows: list[dict[str, Any]] = []
    candidate_payloads: list[dict[str, Any]] = []

    for candidate_id, proxy, start_rule in PROFILE_CANDIDATES:
        config = _candidate_config(
            execution_proxy=proxy,
            exposure_start_rule=start_rule,
            tick_size=tick_size,
        )
        daily_by_side: dict[str, list[IntensityCurve]] = {"pooled": [], "bid": [], "ask": []}
        candidate_daily: list[dict[str, Any]] = []
        for day in days:
            bid_curve, ask_curve = collect_censored_first_passage(
                day["mid"],
                day["dt"],
                config=config,
                best_bid=day["bid"],
                best_ask=day["ask"],
            )
            pooled_curve = sum_intensity_curves([bid_curve, ask_curve], "pooled")
            day_curves = {"pooled": pooled_curve, "bid": bid_curve, "ask": ask_curve}
            daily_diagnostics: list[dict[str, Any]] = []
            for side, curve in day_curves.items():
                daily_by_side[side].append(curve)
                diagnostic, _ = _fit_diagnostics(curve, config.min_hits_per_depth)
                row = {
                    "candidate_id": candidate_id,
                    "date": day["date"],
                    "execution_proxy": proxy,
                    "exposure_start_rule": start_rule,
                    **diagnostic,
                }
                per_day_rows.append(row)
                daily_diagnostics.append(row.copy())
            _add_bid_ask_asymmetry(daily_diagnostics)
            for target, enriched in zip(per_day_rows[-3:], daily_diagnostics):
                target.update(enriched)
            candidate_daily.extend(daily_diagnostics)

        pooled_curves = {
            side: sum_intensity_curves(curves, side) for side, curves in daily_by_side.items()
        }
        candidate_summary: list[dict[str, Any]] = []
        candidate_depths: list[dict[str, Any]] = []
        for side in ("pooled", "bid", "ask"):
            curve = pooled_curves[side]
            diagnostic, _ = _fit_diagnostics(curve, config.min_hits_per_depth)
            row = {
                "candidate_id": candidate_id,
                "execution_proxy": proxy,
                "exposure_start_rule": start_rule,
                "dates": ",".join(PROFILE_DATES),
                **diagnostic,
            }
            summary_rows.append(row)
            candidate_summary.append(row.copy())
            rows = _depth_rows(
                candidate_id,
                curve,
                config.min_hits_per_depth,
                execution_proxy=proxy,
                exposure_start_rule=start_rule,
            )
            depth_rows.extend(rows)
            candidate_depths.extend(rows)
        _add_bid_ask_asymmetry(candidate_summary)
        for target, enriched in zip(summary_rows[-3:], candidate_summary):
            target.update(enriched)

        candidate_payloads.append(
            {
                "candidate_id": candidate_id,
                "config": asdict(config),
                "summary": candidate_summary,
                "depths": candidate_depths,
                "per_day": candidate_daily,
            }
        )
        print(f"Profiled {candidate_id}")

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(depth_rows),
        pd.DataFrame(per_day_rows),
        candidate_payloads,
    )


def _json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_compatible(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    require_snellius_paths()
    cfg = build_config()
    cfg.ensure_artifact_dirs()
    manifest = load_or_prepare_manifest(cfg)
    selected = select_profiling_manifest(manifest)
    paths = profiling_output_paths(cfg.results_dir)
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)

    summary, depths, per_day, candidates = build_profile_tables(
        selected, tick_size=cfg.tick_size
    )
    summary.to_csv(paths["summary"], index=False)
    depths.to_csv(paths["depths"], index=False)
    per_day.to_csv(paths["per_day"], index=False)

    payload = {
        "provenance": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": get_git_commit(),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "dates_used": list(PROFILE_DATES),
            "inputs": selected[["date", "input_path", "input_sha256"]].to_dict(orient="records"),
            "output_directory": paths["json"].parent,
        },
        "profiling_constraints": {
            "selection_data": "January 1-4 candidate-training days only",
            "excluded_from_selection": "January 5-29",
            "downstream_performance_used": False,
            "automatic_winner_selected": False,
        },
        "candidates": candidates,
    }
    paths["json"].write_text(
        json.dumps(_json_compatible(payload), indent=2, allow_nan=False), encoding="utf-8"
    )
    for path in paths.values():
        print(f"Saved calibration profile -> {path}")


if __name__ == "__main__":
    main()
