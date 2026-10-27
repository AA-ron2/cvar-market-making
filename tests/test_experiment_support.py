from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gymnasium")

from cvar_mm.gym.calibration import tune_gamma
from cvar_mm.gym.data_loader import load_single_day
from cvar_mm.gym.experiment_config import BMExperimentConfig, ReplayExperimentConfig, find_repo_root
from cvar_mm.gym.notebook_support import run_qmax_sensitivity, summarise_agent_frames
from cvar_mm.stochastic_processes import MarketReplayMidpriceModel

requires_data = pytest.mark.skipif(
    not os.environ.get("DATA_DIR"), reason="needs the Tardis DOGEUSDT CSVs; set DATA_DIR"
)


def test_find_repo_root_matches_workspace() -> None:
    repo_root = find_repo_root()
    assert (repo_root / "cvar_mm").exists()
    assert (repo_root / "tests").exists()


def test_configs_resolve_repo_relative_paths() -> None:
    bm_cfg = BMExperimentConfig()
    replay_cfg = ReplayExperimentConfig()

    assert bm_cfg.models_dir == bm_cfg.repo_root / "models"
    assert bm_cfg.results_dir == bm_cfg.repo_root / "results"
    assert replay_cfg.models_dir == replay_cfg.repo_root / "models"
    assert replay_cfg.results_dir == replay_cfg.repo_root / "results"


@requires_data
def test_replay_config_finds_the_market_data() -> None:
    replay_cfg = ReplayExperimentConfig()

    assert replay_cfg.data_path().exists()
    assert len(replay_cfg.available_data_files()) >= 1


def test_replay_config_uses_data_dir_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    replay_cfg = ReplayExperimentConfig()

    assert replay_cfg.datasets_dir == tmp_path.resolve()
    assert replay_cfg.data_path() == tmp_path.resolve() / "binance_book_snapshot_25_2025-01-01_DOGEUSDT.csv"


def test_tune_gamma_returns_positive_scalar() -> None:
    pytest.importorskip("optuna")

    midprices = np.linspace(100.0, 101.0, 11)
    dt = np.full(11, 0.1)
    dt[0] = 0.0
    best_gamma, study = tune_gamma(
        midprices=midprices,
        dt_array=dt,
        sigma=2.0,
        kappa=1.5,
        A=140.0,
        tick_size=0.01,
        Q_MAX=10,
        gamma_range=(0.01, 0.2),
        n_trials=2,
        num_trajectories=2,
        seed=0,
        verbose=False,
    )
    assert 0.01 <= best_gamma <= 0.2
    assert all(0.01 <= trial.params["gamma"] <= 0.2 for trial in study.trials)
    assert study.best_trial is not None


@requires_data
def test_qmax_sensitivity_exposes_inventory_pressure_columns() -> None:
    cfg = ReplayExperimentConfig()
    S, dt, _ = load_single_day(str(cfg.data_path()))
    sigma = MarketReplayMidpriceModel(S, dt).volatility
    table = run_qmax_sensitivity(
        S,
        dt,
        gamma=0.1,
        sigma=sigma,
        kappa=cfg.kappa,
        A=cfg.A,
        terminal_time=float(dt.sum()),
        tick_size=cfg.tick_size,
        qmax_candidates=(10, 20),
        num_trajectories=2,
        seed=0,
    )
    assert list(table["Q_MAX"]) == [10, 20]
    assert "Mean |q| Mean" in table.columns
    assert "Near Cap Fraction Mean" in table.columns


def test_summarise_agent_frames_reports_sample_count() -> None:
    frame = summarise_agent_frames({
        "Agent": pd.DataFrame({
            "Sharpe": [0.1, 0.2],
            "Sortino": [0.3, 0.4],
            "Max DD": [1.0, 2.0],
            "P&L-to-MAP": [3.0, 4.0],
            "Final PnL": [5.0, 6.0],
            "Mean |q|": [0.5, 0.6],
            "Near Cap Fraction": [0.0, 0.1],
        }),
    })
    assert frame.loc["Agent", "Samples"] == 2.0
    assert frame.loc["Agent", "Sharpe Mean"] == pytest.approx(0.15)
