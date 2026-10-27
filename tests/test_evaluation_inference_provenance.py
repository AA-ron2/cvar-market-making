from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvar_mm.gym.helpers.fast_rollout import fast_simulate
from cvar_mm.gym.experiment_config import ReplayExperimentConfig
from cvar_mm.gym.statistical_inference import (
    circular_block_bootstrap_mean_ci,
    holm_adjust,
    paired_day_differences,
    pratt_wilcoxon_pvalue,
)
from scripts.pipeline import common as snellius_common
from scripts.pipeline.common import (
    load_or_prepare_manifest,
    prepare_manifest,
    require_complete_day_budget,
)


def test_complete_day_budget_depends_only_on_budget_and_day_lengths():
    daily_midprices = [np.arange(3.0), np.arange(4.0)]
    assert require_complete_day_budget(5, daily_midprices) == 5
    assert require_complete_day_budget(6, daily_midprices) == 5
    with pytest.raises(ValueError, match="below the 5 transitions"):
        require_complete_day_budget(4, daily_midprices)


def test_manifest_enforces_canonical_dates_and_separates_later_holdout(monkeypatch):
    cfg = ReplayExperimentConfig(
        repo_root=Path.cwd(),
        datasets_subdir="datasets",
        results_subdir="results",
    )
    files = [
        cfg.data_path(date.strftime("%Y-%m-%d"))
        for date in pd.date_range("2025-01-01", "2025-01-31")
    ]

    class InMemoryManifestPath:
        parent = None

        def mkdir(self, **kwargs):
            pass

        def exists(self):
            return True

    manifest_path = InMemoryManifestPath()
    manifest_path.parent = manifest_path
    monkeypatch.setattr(Path, "glob", lambda self, pattern: files.copy())
    monkeypatch.setattr(snellius_common, "file_sha256", lambda path: "synthetic-hash")
    monkeypatch.setattr(snellius_common, "manifest_path", lambda config: manifest_path)
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda self, path, index: None)

    manifest = prepare_manifest(cfg)
    assert manifest["date"].iloc[:6].tolist() == [
        date.strftime("%Y-%m-%d") for date in pd.date_range("2025-01-01", "2025-01-06")
    ]
    assert manifest["date"].iloc[6:29].tolist() == [
        date.strftime("%Y-%m-%d") for date in pd.date_range("2025-01-07", "2025-01-29")
    ]
    assert manifest["usage"].iloc[:4].tolist() == ["candidate_train"] * 4
    assert manifest["usage"].iloc[4:6].tolist() == ["validation"] * 2
    assert manifest["usage"].iloc[6:29].tolist() == ["retrospective_holdout"] * 23
    assert manifest["usage"].iloc[29:].tolist() == ["later_untouched_holdout"] * 2
    assert manifest["split"].iloc[29:].tolist() == ["holdout"] * 2
    monkeypatch.setattr(pd, "read_csv", lambda path: manifest.copy())
    assert load_or_prepare_manifest(cfg)["date"].tolist() == manifest["date"].tolist()

    # A previously written manifest must not bypass the fixed split check.
    wrong_date = manifest.copy()
    wrong_date.loc[28, "date"] = "2025-01-30"
    monkeypatch.setattr(pd, "read_csv", lambda path: wrong_date.copy())
    with pytest.raises(ValueError, match="cached corrected split requires exactly"):
        load_or_prepare_manifest(cfg)

    wrong_input = manifest.copy()
    wrong_input.loc[28, "input_path"] = str(cfg.data_path("2025-01-30"))
    monkeypatch.setattr(pd, "read_csv", lambda path: wrong_input.copy())
    with pytest.raises(ValueError, match="cached corrected split requires exactly"):
        load_or_prepare_manifest(cfg)

    bad_cached = manifest.copy()
    bad_cached.loc[29, "usage"] = "retrospective_holdout"
    monkeypatch.setattr(pd, "read_csv", lambda path: bad_cached.copy())
    with pytest.raises(ValueError, match="invalid development, retrospective, or later-holdout"):
        load_or_prepare_manifest(cfg)

    # Missing January 29 may not promote a later day into the retrospective split.
    files.remove(cfg.data_path("2025-01-29"))
    with pytest.raises(ValueError, match="corrected split requires exactly"):
        prepare_manifest(cfg)


def test_fast_evaluation_labels_fixed_clock_primary_and_snapshot_supplementary():
    stats = fast_simulate(
        midprices=np.array([100.0, 100.1, 100.0, 100.2, 100.1]),
        dt_array=np.array([0.0, 30.0, 30.0, 30.0, 30.0]),
        gamma=0.1,
        sigma=0.01,
        kappa=1.0,
        A=0.1,
        terminal_time=120.0,
        tick_size=0.01,
        seed=4,
        fixed_clock_seconds=60.0,
    )
    assert stats["fixed_clock_return_count"] == 2
    assert stats["n_steps"] == 4
    assert "snapshot_sharpe" in stats and "snapshot_sortino" in stats


def test_paired_differences_reject_fill_paths_as_independent_day_rows():
    duplicate_day = pd.DataFrame({"Day": ["d1", "d1"], "Sharpe": [1.0, 2.0]})
    one_per_day = pd.DataFrame({"Day": ["d1"], "Sharpe": [2.0]})
    with pytest.raises(ValueError, match="multiple rows per market day"):
        paired_day_differences(duplicate_day, one_per_day, "Sharpe")


def test_day_paired_inference_helpers_are_deterministic_and_holm_monotone():
    reference = pd.DataFrame({"Day": ["d1", "d2", "d3"], "Sharpe": [1.0, 2.0, 3.0]})
    challenger = pd.DataFrame({"Day": ["d3", "d1", "d2"], "Sharpe": [4.0, 2.0, 3.0]})
    days, differences = paired_day_differences(reference, challenger, "Sharpe")
    assert days == ["d1", "d2", "d3"]
    assert differences == pytest.approx([1.0, 1.0, 1.0])
    assert circular_block_bootstrap_mean_ci(
        differences, block_length=2, repetitions=100, seed=7
    ) == pytest.approx((1.0, 1.0))
    assert 0 <= pratt_wilcoxon_pvalue(differences) <= 1
    adjusted = holm_adjust(np.array([0.01, 0.04, 0.03]))
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
