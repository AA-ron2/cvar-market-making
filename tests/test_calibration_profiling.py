from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from cvar_mm.gym.experiment_config import ReplayExperimentConfig
from scripts.pipeline import common as snellius_common
from scripts.pipeline import profile_formula_calibration as profiler


def _canonical_manifest() -> pd.DataFrame:
    rows = []
    for index, timestamp in enumerate(pd.date_range("2025-01-01", "2025-01-29")):
        date = timestamp.strftime("%Y-%m-%d")
        if index < 4:
            split, usage = "train", "candidate_train"
        elif index < 6:
            split, usage = "train", "validation"
        else:
            split, usage = "test", "retrospective_holdout"
        rows.append(
            {
                "date": date,
                "split": split,
                "usage": usage,
                "input_path": f"synthetic_{date}.csv",
                "input_sha256": f"hash-{date}",
            }
        )
    return pd.DataFrame(rows)


def test_profiler_loads_only_january_1_to_4(monkeypatch: pytest.MonkeyPatch):
    manifest = _canonical_manifest()
    loaded: list[str] = []

    def fake_load(path: str):
        loaded.append(path)
        mid = np.arange(100.0, 141.0)
        dt = np.ones_like(mid)
        dt[0] = 0.0
        return mid, dt, None, mid - 0.5, mid + 0.5

    monkeypatch.setattr(profiler, "load_single_day_calibration_arrays", fake_load)
    summary, depths, per_day, _ = profiler.build_profile_tables(manifest, tick_size=1.0)

    assert loaded == [f"synthetic_{date}.csv" for date in profiler.PROFILE_DATES]
    assert not any("2025-01-05" in path or "2025-01-06" in path for path in loaded)
    assert not any(f"2025-01-{day:02d}" in path for path in loaded for day in range(7, 30))
    assert len(summary) == 4 * 3
    assert len(depths) == 4 * 3 * len(profiler.PROFILE_DEPTH_TICKS)
    assert len(per_day) == 4 * 4 * 3
    assert {
        "depth_ticks",
        "depth_price",
        "hits",
        "total_exposure_seconds",
        "exposures",
        "censored_count",
        "censoring_fraction",
        "empirical_intensity_per_second",
        "valid_fit_depth",
    } <= set(depths.columns)
    assert (depths["exposures"] == depths["hits"] + depths["censored_count"]).all()
    assert {
        "A",
        "kappa",
        "valid_depth_count",
        "weighted_log_r2",
        "weighted_log_rmse",
        "empirical_monotonicity_fraction",
        "supported_prefix_count",
    } <= set(summary.columns)
    assert set(per_day["date"]) == set(profiler.PROFILE_DATES)


def test_profiler_rejects_noncanonical_candidate_training_membership():
    manifest = _canonical_manifest()
    manifest.loc[4, "usage"] = "candidate_train"
    with pytest.raises(ValueError, match="candidate-training set must be exactly"):
        profiler.select_profiling_manifest(manifest)


def test_profile_outputs_are_separate_from_canonical_calibration():
    results_dir = Path.cwd() / ".synthetic_results_path_check"
    paths = profiler.profiling_output_paths(results_dir)
    canonical = {
        (results_dir / "formula_train_params.json").resolve(),
        (results_dir / "formula_train_params.csv").resolve(),
    }

    assert set(path.resolve() for path in paths.values()).isdisjoint(canonical)
    assert all(path.parent == (results_dir / "profiling" / "calibration").resolve() for path in paths.values())


def test_primary_b0_and_rl_gamma_bounds_are_aligned(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORMULA_GAMMA_MIN", "0.01")
    monkeypatch.setenv("FORMULA_GAMMA_MAX", "0.9")
    cfg = snellius_common.build_config()
    kwargs = snellius_common.formula_kwargs_from_config({"sigma_train": 0.2}, cfg)

    assert ReplayExperimentConfig().as_gamma_range == (0.01, 0.9)
    assert cfg.as_gamma_range == (kwargs["gamma_min"], kwargs["gamma_max"])
    assert snellius_common.primary_b0_gamma_range(cfg, kwargs) == (0.01, 0.9)

    mismatched = replace(cfg, as_gamma_range=(0.001, 1.0))
    with pytest.raises(RuntimeError, match="must equal"):
        snellius_common.primary_b0_gamma_range(mismatched, kwargs)

    assert snellius_common.require_gamma_within_bounds(0.01, (0.01, 0.9)) == 0.01
    assert snellius_common.require_gamma_within_bounds(0.9, (0.01, 0.9)) == 0.9
    with pytest.raises(RuntimeError, match="outside configured bounds"):
        snellius_common.require_gamma_within_bounds(0.9817, (0.01, 0.9))
