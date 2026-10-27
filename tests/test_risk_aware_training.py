"""Script-level tests of the B1-B3 training jobs on synthetic replay days."""

from __future__ import annotations

import importlib
import json
import math
import re
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvar_mm.gym.experiment_config import ReplayExperimentConfig
from scripts.pipeline import common as snellius_common

TICK = 1e-5
JAN_1_6 = {"sigma": 2.7e-5, "A": 0.11, "kappa": 12_000.0}
JAN_1_4 = {"sigma": 3.1e-5, "A": 0.13, "kappa": 11_000.0}
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def synthetic_day(date: str, n_snapshots: int = 12):
    rng = np.random.default_rng(int(date.replace("-", "")))
    midprices = 0.35 + TICK * np.cumsum(np.r_[0, rng.integers(-1, 2, n_snapshots - 1)])
    dt = np.ones(n_snapshots)
    dt[0] = 0.0
    dt[5] = 0.0  # duplicate exchange timestamp: the mid may move, nothing fills
    features = {
        "spread_ticks": np.ones(n_snapshots, dtype=np.float32),
        "lob_imbalance": np.zeros(n_snapshots, dtype=np.float32),
    }
    return midprices, dt, None, features


def manifest(dates, usages) -> pd.DataFrame:
    return pd.DataFrame({
        "date": list(dates),
        "split": ["test" if usage == "retrospective_holdout" else "train" for usage in usages],
        "usage": list(usages),
        "input_path": [f"synthetic_{date}.csv" for date in dates],
    })


class Pipeline:
    """A fake prepared pipeline under tmp_path; scripts run their real main()."""

    def __init__(self, root: Path, monkeypatch, frame: pd.DataFrame):
        cfg = replace(
            ReplayExperimentConfig(repo_root=root, datasets_subdir=root),
            A=JAN_1_6["A"],
            kappa=JAN_1_6["kappa"],
        )
        cfg.ensure_artifact_dirs()
        train_params = {
            "sigma_train": JAN_1_6["sigma"],
            "A_train": JAN_1_6["A"],
            "kappa_train": JAN_1_6["kappa"],
            "candidate_only": {
                "sigma_train": JAN_1_4["sigma"],
                "A_train": JAN_1_4["A"],
                "kappa_train": JAN_1_4["kappa"],
            },
        }
        self.ctx = snellius_common.FormulaPipelineContext(
            project_dir=root,
            data_dir=root,
            cfg=cfg,
            manifest=frame,
            train_params=train_params,
            formula_kwargs=snellius_common.formula_kwargs_from_config(train_params, cfg),
        )
        self.monkeypatch = monkeypatch
        monkeypatch.setenv("POLICY_STD", "0.3")  # D-21, as configs/final_pipeline.env exports it
        monkeypatch.setenv("FINE_TUNE_LR", "0.003")  # D-23, likewise
        self.loaded: list[str] = []
        self.metadata: list[tuple[str, dict]] = []
        self.context_kwargs: list[dict] = []
        monkeypatch.setattr(snellius_common, "load_single_day_with_features", self._load)
        self.write_prepared()  # as after prep: the B2 jobs hash the calibration and the manifest (D-19)

    def _load(self, path, *, tick_size):
        self.loaded.append(DATE_RE.search(Path(path).name).group(0))
        return synthetic_day(self.loaded[-1])

    def run(self, script: str, *argv: str) -> dict:
        module = importlib.import_module(f"scripts.pipeline.{script}")

        def load_context(**kwargs):
            self.context_kwargs.append(kwargs)
            return self.ctx

        self.monkeypatch.setattr(module, "load_context", load_context)
        self.monkeypatch.setattr(
            module,
            "write_metadata",
            lambda job, payload: self.metadata.append((job, payload)),
            raising=False,
        )
        self.monkeypatch.setattr(sys, "argv", [f"{script}.py", *argv])
        module.main()
        return self.metadata[-1][1]

    def train_b1(self, model_stem: str = "ppo_b1_doge", vecnorm_stem: str = "vecnorm_b1"):
        """Save a trained B1 stand-in: non-default weights, obs_rms and ret_rms, and sigma fixed at POLICY_STD by
        the helper B1 trains with (D-21)."""

        from cvar_mm.gym.notebook_support import build_formula_multi_day_replay_env, make_vecnorm
        from cvar_mm.gym.sb3_wrapper import StableBaselinesTradingEnvironment
        from cvar_mm.rewards import PnLReward

        days = [synthetic_day(str(date)) for date in self.ctx.train_manifest["date"]]
        env = build_formula_multi_day_replay_env(
            [day[0] for day in days],
            [day[1] for day in days],
            self.ctx.cfg,
            daily_market_features=[day[3] for day in days],
            reward_fn=PnLReward(),
            mode="sequential",
            **self.ctx.formula_kwargs,
        )
        vecnorm = make_vecnorm(
            StableBaselinesTradingEnvironment(env), self.ctx.cfg, training=True, norm_reward=True
        )
        model = snellius_common.fixed_std_ppo(
            vecnorm, {"n_steps": 64, "batch_size": 32, "n_epochs": 2}, seed=7, device="cpu"
        )
        model.learn(total_timesteps=128)
        assert vecnorm.ret_rms.count > 1
        model.save(str(self.ctx.cfg.model_path(f"{model_stem}.zip")))
        vecnorm.save(str(self.ctx.cfg.vecnorm_path(vecnorm_stem)))
        return model, vecnorm

    def write_prepared(self) -> None:
        """The files prepare_formula_pipeline.py would have written."""

        self.ctx.manifest.to_csv(snellius_common.manifest_path(self.ctx.cfg), index=False)
        snellius_common.train_params_json_path(self.ctx.cfg).write_text(
            json.dumps(self.ctx.train_params), encoding="utf-8"
        )

    def write_threshold(self, d: float = 0.05) -> None:
        """A d file pair with the fields load_b3_threshold checks and summary.md reports (D-17)."""

        results = self.ctx.cfg
        results.result_path("cvar_threshold_d.txt").write_text(f"{d}\n{d / 0.8}\n0.0\n", encoding="utf-8")
        results.result_path("cvar_threshold_b3.json").write_text(
            json.dumps({
                "estimator": "stochastic (D-17)",
                "d": d,
                "rho": 0.2,
                "threshold_multiplier": 0.8,
                "cvar_stochastic": d / 0.8,
                "nu_stochastic": 0.0,
                "cvar_deterministic": 0.9 * d / 0.8,
                "nu_deterministic": 0.0,
                "b1_log_std": [0.0, 0.0],
                "b1_std": [1.0, 1.0],
            }),
            encoding="utf-8",
        )


@pytest.fixture
def two_days(tmp_path, monkeypatch):
    pytest.importorskip("gymnasium")
    pytest.importorskip("stable_baselines3")
    for name in ("TOTAL_TIMESTEPS", "ABLATION_TIMESTEPS", "B2_UPDATES", "B3_UPDATES", "TRAINING_SEED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TRAINING_FILL_PATHS", "2")
    monkeypatch.setenv("B3_LAMBDA_LR", "0.5")
    monkeypatch.setenv("B1_PASSES", "1")  # F-16
    frame = manifest(["2025-01-01", "2025-01-02"], ["candidate_train", "candidate_train"])
    return Pipeline(tmp_path, monkeypatch, frame)


@pytest.fixture
def ppo_iterations(monkeypatch):
    from stable_baselines3 import PPO

    calls: list[int] = []
    original = PPO.train

    def counting_train(self):
        calls.append(self.num_timesteps)
        return original(self)

    monkeypatch.setattr(PPO, "train", counting_train)
    return calls


def test_risk_aware_updates_require_equal_explicit_counts(monkeypatch):
    for name in ("B2_UPDATES", "B3_UPDATES"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="B2_UPDATES must be set"):
        snellius_common.risk_aware_updates("b2")
    monkeypatch.setenv("B2_UPDATES", "3")
    with pytest.raises(RuntimeError, match="B3_UPDATES must be set"):
        snellius_common.risk_aware_updates("b2")
    monkeypatch.setenv("B3_UPDATES", "4")
    with pytest.raises(RuntimeError, match="same k"):
        snellius_common.risk_aware_updates("b3")
    assert snellius_common.risk_aware_updates("b3", allow_unequal=True) == 4
    monkeypatch.setenv("B3_UPDATES", "0")
    with pytest.raises(ValueError, match="positive"):
        snellius_common.risk_aware_updates("b3", allow_unequal=True)
    monkeypatch.setenv("B3_UPDATES", "3")
    assert snellius_common.risk_aware_updates("b2") == 3


def test_b2_and_b3_run_exactly_k_ppo_iterations(two_days, ppo_iterations, monkeypatch):
    monkeypatch.setenv("B2_UPDATES", "3")
    monkeypatch.setenv("B3_UPDATES", "3")
    monkeypatch.setenv("TOTAL_TIMESTEPS", "999999")  # B1 refuses it (F-16); B2 and B3 never read it
    two_days.train_b1()
    two_days.train_b1("ppo_b1_cand_doge", "vecnorm_b1_cand")
    two_days.write_threshold()

    for script, argv in (
        ("train_formula_b2", ("--phase", "candidate", "--alpha", "1.0")),
        ("train_formula_b3", ()),
    ):
        ppo_iterations.clear()
        payload = two_days.run(script, *argv)
        batch = payload["ppo_kwargs"]["n_steps"]
        assert batch == sum(len(synthetic_day(date)[0]) - 1 for date in ("2025-01-01", "2025-01-02"))
        assert len(ppo_iterations) == 3
        assert payload["risk_aware_updates"] == 3
        assert payload["training_fill_paths"] == 2
        assert payload["total_timesteps"] == 3 * batch * 2
        assert payload["ppo_kwargs"]["batch_size"] == math.ceil(batch * 2 / 32)  # D-15: 32 minibatches

    # B3 feasibility (K6): rollouts after the warm-start rollout, one cost per day x path x update.
    results = two_days.ctx.cfg
    feasibility = json.loads(results.result_path("b3_training_feasibility.json").read_text(encoding="utf-8"))
    assert feasibility["rollouts"] == [2, 3] and feasibility["paths_pooled"] == 2 * 2 * 2
    costs = pd.read_csv(results.result_path("b3_training_costs.csv"))
    assert len(costs) == 3 * 2 * 2 and sorted(costs["update"].unique()) == [1, 2, 3]
    history = pd.read_csv(results.result_path("b3_cvar_lagrangian_history.csv"))
    assert history["rollout"].tolist() == [1, 2, 3]  # rollout j is update j in every B3 output
    from cvar_mm.gym.drawdown_objectives import UPDATE_DIAGNOSTIC_COLUMNS

    assert set(UPDATE_DIAGNOSTIC_COLUMNS) <= set(history.columns)  # D-15 and D-21 diagnostics, one row per update


def test_unequal_updates_need_the_explicit_override_flag(two_days, ppo_iterations, monkeypatch):
    monkeypatch.setenv("B2_UPDATES", "2")
    monkeypatch.setenv("B3_UPDATES", "4")
    two_days.train_b1("ppo_b1_cand_doge", "vecnorm_b1_cand")
    ppo_iterations.clear()
    with pytest.raises(RuntimeError, match="same k"):
        two_days.run("train_formula_b2", "--phase", "candidate", "--alpha", "1.0")
    assert ppo_iterations == []

    payload = two_days.run(
        "train_formula_b2", "--phase", "candidate", "--alpha", "1.0", "--allow-unequal-updates"
    )
    assert len(ppo_iterations) == 2
    assert payload["unequal_updates_override"] is True
    assert payload["declared_updates"] == {"B2_UPDATES": "2", "B3_UPDATES": "4"}


def test_risk_aware_ppo_kwargs_split_every_batch_into_32_minibatches(tmp_path, monkeypatch):
    """D-15: minibatches of ceil(Y / 32), so 32 per epoch and at most 320 Adam steps per iteration."""

    monkeypatch.setenv("FINE_TUNE_LR", "0.003")  # D-23, as configs/final_pipeline.env exports it
    cfg = ReplayExperimentConfig(repo_root=tmp_path)
    # Jan 1-6 at N = 16 and N = 8, an odd step count at N = 16, the evidence stand-in, test (c)'s batch.
    for n_steps, num_envs in ((4_524_754, 16), (4_524_754, 8), (2_999_999, 16), (17_994, 8), (22, 16)):
        kwargs = snellius_common.risk_aware_ppo_kwargs(cfg, n_steps, num_envs)
        transitions = n_steps * num_envs
        assert math.ceil(transitions / kwargs["batch_size"]) == 32
        assert 31 * kwargs["batch_size"] < transitions <= 32 * kwargs["batch_size"]
        assert kwargs["n_steps"] == n_steps
        assert (kwargs["n_epochs"], kwargs["target_kl"], kwargs["normalize_advantage"]) == (10, 0.02, False)
        assert kwargs["ent_coef"] == cfg.ppo_ent_coef
        assert kwargs["learning_rate"] == 0.003  # FINE_TUNE_LR, not B1's 3e-4 (D-23)
        for key in ("gamma", "gae_lambda", "clip_range"):
            assert kwargs[key] == cfg.ppo_kwargs()[key]
    with pytest.raises(ValueError, match="at least 32 transitions"):
        snellius_common.risk_aware_ppo_kwargs(cfg, 15, 2)


# What risk_aware_ppo_kwargs returned at 8a42ad8, before D-23 (printed there for these inputs): B1's learning rate.
PPO_KWARGS_AT_8A42AD8 = {
    (4_524_754, 16): {  # Jan 1-6 at N = 16
        "n_steps": 4_524_754, "batch_size": 2_262_377, "n_epochs": 10, "learning_rate": 0.0003, "gamma": 0.999,
        "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01, "target_kl": 0.02, "normalize_advantage": False,
    },
    (22, 16): {  # two synthetic days at N = 16
        "n_steps": 22, "batch_size": 11, "n_epochs": 10, "learning_rate": 0.0003, "gamma": 0.999,
        "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01, "target_kl": 0.02, "normalize_advantage": False,
    },
}


def test_risk_aware_ppo_kwargs_change_only_the_learning_rate_to_fine_tune_lr(tmp_path, monkeypatch):
    """D-23 test (a): risk_aware_ppo_kwargs returns learning_rate = FINE_TUNE_LR = 0.003, and every other key, in
    the same order, as at 8a42ad8; B1's own settings (cfg.ppo_kwargs(), 3e-4) are unchanged."""

    monkeypatch.setenv("FINE_TUNE_LR", "0.003")
    cfg = ReplayExperimentConfig(repo_root=tmp_path)
    for (n_steps, num_envs), before in PPO_KWARGS_AT_8A42AD8.items():
        kwargs = snellius_common.risk_aware_ppo_kwargs(cfg, n_steps, num_envs)
        assert list(kwargs) == list(before)  # the recorded ppo_kwargs differ in learning_rate's value only
        assert kwargs == {**before, "learning_rate": 0.003}
    assert cfg.ppo_learning_rate == cfg.ppo_kwargs()["learning_rate"] == 3e-4  # B1, B1_cand and NB6


def test_fine_tune_lr_fails_closed_naming_d23(tmp_path, monkeypatch):
    """D-23: FINE_TUNE_LR is read like POLICY_STD, a positive number from the environment, or an error naming
    D-23; risk_aware_ppo_kwargs raises it too, so no fine-tune can fall back to B1's 3e-4."""

    cfg = ReplayExperimentConfig(repo_root=tmp_path)
    for raw in ("", "0", "-0.003", "nan", "inf", "0.003x"):
        monkeypatch.setenv("FINE_TUNE_LR", raw)
        with pytest.raises(ValueError, match="FINE_TUNE_LR must be a positive number.*D-23"):
            snellius_common.fine_tune_lr()
        with pytest.raises(ValueError, match="FINE_TUNE_LR must be a positive number.*D-23"):
            snellius_common.risk_aware_ppo_kwargs(cfg, 22, 16)
    monkeypatch.delenv("FINE_TUNE_LR")
    with pytest.raises(ValueError, match="D-23"):
        snellius_common.risk_aware_ppo_kwargs(cfg, 22, 16)
    monkeypatch.setenv("FINE_TUNE_LR", "0.003")
    assert snellius_common.fine_tune_lr() == 0.003


def test_final_b2_and_b3_train_with_identical_d15_ppo_settings(two_days, monkeypatch):
    """D-15 test (c): one helper gives final B2 and B3 the same PPO, as recorded and as built."""

    from stable_baselines3 import PPO

    monkeypatch.setenv("TRAINING_FILL_PATHS", "16")
    monkeypatch.setenv("B2_UPDATES", "2")
    monkeypatch.setenv("B3_UPDATES", "2")
    two_days.train_b1()
    two_days.write_threshold()
    two_days.ctx.cfg.result_path("b2_best_alpha.txt").write_text("1.0", encoding="utf-8")

    built: list[dict] = []
    original_learn = PPO.learn

    def recording_learn(self, *args, **kwargs):
        built.append({
            "n_steps": self.n_steps, "num_envs": self.get_env().num_envs, "batch_size": self.batch_size,
            "n_epochs": self.n_epochs, "target_kl": self.target_kl,
            "normalize_advantage": self.normalize_advantage, "ent_coef": self.ent_coef,
            "learning_rate": self.learning_rate, "gamma": self.gamma, "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range(1.0),
        })
        return original_learn(self, *args, **kwargs)

    monkeypatch.setattr(PPO, "learn", recording_learn)
    b2 = two_days.run("train_formula_b2", "--phase", "final")
    b3 = two_days.run("train_formula_b3")

    assert b2["ppo_kwargs"] == b3["ppo_kwargs"]
    assert len(built) == 2 and built[0] == built[1]
    for payload in (b2, b3):  # item 8: the Jan 1-6 calibration they trained with
        assert (payload["experiment_config"].A, payload["experiment_config"].kappa) == (JAN_1_6["A"], JAN_1_6["kappa"])
    live = built[0]
    assert live["n_steps"] * live["num_envs"] == 22 * 16 == live["batch_size"] * 32
    assert (live["n_epochs"], live["target_kl"], live["normalize_advantage"]) == (10, 0.02, False)
    assert live["ent_coef"] == two_days.ctx.cfg.ppo_ent_coef
    assert live["learning_rate"] == 0.003 != two_days.ctx.cfg.ppo_learning_rate  # FINE_TUNE_LR, not B1's (D-23)
    for key in ("n_steps", "batch_size", "n_epochs", "target_kl", "normalize_advantage", "ent_coef",
                "learning_rate", "gamma", "gae_lambda", "clip_range"):
        assert live[key] == b2["ppo_kwargs"][key], key


def test_final_model_stems_name_the_four_models_evaluated_on_jan_7_to_29():
    assert {model: snellius_common.final_model_stems(model, 42) for model in ("b1", "b1plus", "b2", "b3")} == {
        "b1": ("ppo_b1_doge", "vecnorm_b1"),
        "b1plus": ("ppo_b1plus_doge", "vecnorm_b1plus"),  # D-18
        "b2": ("ppo_b2_doge", "vecnorm_b2"),
        "b3": ("ppo_b3_doge", "vecnorm_b3"),
    }
    assert snellius_common.final_model_stems("b1plus", 43) == ("ppo_b1plus_doge_seed43", "vecnorm_b1plus_seed43")
    with pytest.raises(ValueError, match="No final model"):
        snellius_common.final_model_stems("b0", 42)


def test_b1plus_trains_on_exactly_the_standardized_pnl_advantages(two_days, monkeypatch):
    """D-18: B1+ is the B2 path at alpha = 0, so its score term is identically 0 and every update trains
    on (A - mu_A) / sigma_A of the P&L advantages, bit for bit."""

    from stable_baselines3 import PPO

    from cvar_mm.gym import drawdown_objectives

    monkeypatch.setenv("B2_UPDATES", "2")
    monkeypatch.setenv("B3_UPDATES", "2")
    two_days.train_b1()
    calls: list[tuple] = []
    trained_on: list[np.ndarray] = []
    helper, train = drawdown_objectives.standardized_risk_advantages, PPO.train

    def spying_helper(advantages, segments, coefficients, *, weight, reward_scale):
        calls.append((weight, np.asarray(advantages, dtype=np.float64).copy(), np.asarray(coefficients, dtype=float)))
        return helper(advantages, segments, coefficients, weight=weight, reward_scale=reward_scale)

    def capturing_train(self):
        trained_on.append(self.rollout_buffer.advantages.copy())
        train(self)

    # B2's callback reaches the helper through the module (B3 imports the name itself).
    monkeypatch.setattr(drawdown_objectives, "standardized_risk_advantages", spying_helper)
    monkeypatch.setattr(PPO, "train", capturing_train)
    payload = two_days.run("train_formula_b2", "--phase", "final", "--control", "b1plus")

    assert (payload["model"], payload["alpha"]) == ("b1plus", 0.0)
    assert [call[0] for call in calls] == [0.0, 0.0]  # one batch per update, k = 2, weight w = alpha = 0
    assert len(trained_on) == 2
    for (_, gae, _), used in zip(calls, trained_on):
        np.testing.assert_array_equal(used, ((gae - gae.mean()) / gae.std()).astype(np.float32))
    assert any(np.ptp(costs) > 0 for _, _, costs in calls)  # drawdowns differ: a weight > 0 would have shown


def test_b1plus_trains_exactly_like_the_final_b2_except_alpha(two_days, monkeypatch):
    """D-18 test: B1+'s PPO kwargs, N, k and seed equal final B2's (as recorded and as built). It needs no
    B2 selection (D-19: it waits for B1 only), writes its own stems and leaves final B2's files alone."""

    from stable_baselines3 import PPO

    monkeypatch.setenv("TRAINING_FILL_PATHS", "16")
    monkeypatch.setenv("B2_UPDATES", "2")
    monkeypatch.setenv("B3_UPDATES", "2")
    two_days.train_b1()
    cfg = two_days.ctx.cfg
    selection = cfg.result_path("b2_best_alpha.txt")
    selection.write_text("1.0", encoding="utf-8")
    built: list[dict] = []
    original_learn = PPO.learn

    def recording_learn(self, *args, **kwargs):
        built.append({
            "n_steps": self.n_steps, "num_envs": self.get_env().num_envs, "batch_size": self.batch_size,
            "n_epochs": self.n_epochs, "target_kl": self.target_kl,
            "normalize_advantage": self.normalize_advantage, "ent_coef": self.ent_coef,
            "learning_rate": self.learning_rate, "gamma": self.gamma, "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range(1.0), "seed": self.seed,
        })
        return original_learn(self, *args, **kwargs)

    monkeypatch.setattr(PPO, "learn", recording_learn)
    b2 = two_days.run("train_formula_b2", "--phase", "final")
    b2_files = [cfg.model_path("ppo_b2_doge.zip"), cfg.vecnorm_path("vecnorm_b2")]
    b2_hashes = [snellius_common.file_sha256(path) for path in b2_files]
    selection.unlink()
    for source, target in zip(b2_files, (cfg.model_path("ppo_b1plus_doge.zip"), cfg.vecnorm_path("vecnorm_b1plus"))):
        target.write_bytes(source.read_bytes())  # earlier B1+ files: the legacy archive would copy them
    b1plus = two_days.run("train_formula_b2", "--phase", "final", "--control", "b1plus")

    assert two_days.metadata[-1][0] == "train_formula_b1plus"
    assert (b1plus["model"], b1plus["alpha"], b2["alpha"]) == ("b1plus", 0.0, 1.0)
    assert b1plus["ppo_kwargs"] == b2["ppo_kwargs"]
    assert len(built) == 2 and built[0] == built[1]
    for key in ("phase", "training_seed", "training_fill_paths", "risk_aware_updates", "total_timesteps",
                "train_dates", "minimum_complete_day_transitions", "calibration_scope", "formula_kwargs",
                "warm_start_model", "warm_start_vecnorm", "reward_scale_s", "frozen_log_std", "adam_state"):
        assert b1plus[key] == b2[key], key
    assert (b1plus["training_fill_paths"], b1plus["risk_aware_updates"], b1plus["training_seed"]) == (16, 2, 42)
    assert Path(b2["b2_selection"]).name == "b2_best_alpha.txt" and "b2_selection" not in b1plus
    assert (Path(b1plus["model_path"]).name, Path(b1plus["vecnorm_path"]).name) == (
        "ppo_b1plus_doge.zip", "vecnorm_b1plus.pkl")
    assert all(Path(b1plus[key]).exists() for key in ("model_path", "vecnorm_path"))
    assert [snellius_common.file_sha256(path) for path in b2_files] == b2_hashes  # final B2 untouched
    assert not list(cfg.model_path("x").parent.glob("*b1plus*direct_depth*"))
    # D-22: the pipeline's final B2 is one of the parallel finals, which needs no selection either.
    parallel = two_days.run("train_formula_b2", "--phase", "final", "--alpha", "1.0")
    assert len(built) == 3 and built[2] == built[1]
    for key in ("phase", "training_seed", "training_fill_paths", "risk_aware_updates", "total_timesteps",
                "train_dates", "minimum_complete_day_transitions", "calibration_scope", "formula_kwargs",
                "warm_start_model", "warm_start_vecnorm", "reward_scale_s", "frozen_log_std", "adam_state", "ppo_kwargs"):
        assert parallel[key] == b1plus[key], key
    assert "b2_selection" not in parallel and parallel["alpha"] == 1.0
    assert (Path(parallel["model_path"]).name, Path(parallel["vecnorm_path"]).name) == (
        "ppo_b2_final_alpha1.0_doge.zip", "vecnorm_b2_final_alpha1.0.pkl")
    with pytest.raises(FileNotFoundError):  # the sequential final B2 does need the selection
        two_days.run("train_formula_b2", "--phase", "final")


def test_b1plus_equals_b3_while_lambda_stays_zero(two_days, monkeypatch):
    """D-15a/D-18: at w = 0 the B2 path (B1+) and B3 are one algorithm. With equal seeds and lambda held at 0
    (d far above every drawdown), B1+ after k updates equals B3 after k updates bit for bit; with a binding
    d, B3's second update (lambda > 0) differs, so the equality is not vacuous."""

    import torch
    from stable_baselines3 import PPO

    from scripts.pipeline import train_formula_b3

    # 8 paths per day: with 2 x 2 paths the (1 - rho)-quantile nu is the batch maximum, so (Z - nu)+ = 0 on
    # every path and even lambda = 500 leaves B3 equal to B1+; 16 paths put some above nu.
    monkeypatch.setenv("TRAINING_FILL_PATHS", "8")
    monkeypatch.setenv("B2_UPDATES", "2")
    monkeypatch.setenv("B3_UPDATES", "2")
    b1, _ = two_days.train_b1()
    two_days.run("train_formula_b2", "--phase", "final", "--control", "b1plus")
    b1plus = PPO.load(str(two_days.ctx.cfg.model_path("ppo_b1plus_doge.zip")), device="cpu").policy.state_dict()
    data = snellius_common.load_train_arrays(two_days.ctx)

    def b3(threshold_d: float):
        return train_formula_b3.run_b3(
            two_days.ctx, *data, threshold_d=threshold_d, lambda_lr=0.5, updates=2, fill_paths=8, seed=42,
            tensorboard_log=None,
        )

    slack = b3(1e6)
    assert slack.callback.lambda_used_history == [0.0, 0.0]
    assert_same_tensors(slack.model.policy.state_dict(), b1plus)
    before = b1.policy.state_dict()
    assert any(not torch.equal(b1plus[key], before[key]) for key in before if key.startswith("action_net."))

    binding = b3(1e-9)
    assert binding.callback.lambda_used_history[0] == 0.0 < binding.callback.lambda_used_history[1]
    after = binding.model.policy.state_dict()
    assert any(not torch.equal(after[key], b1plus[key]) for key in b1plus)


def store_metadata(directory: Path, job: str, payload: dict, index: int, commit: str | None = None) -> Path:
    """A job's metadata as write_metadata stores it: the payload, the commit and status at the end of the job,
    and the SHA-256 of every existing file the payload names."""

    record = {
        "job_name": job, "slurm_job_id": "local", "timestamp_utc": f"20261008T12{index:04d}Z",
        "git_commit": commit, "git_status_porcelain": "", **payload,
        "artifact_sha256": snellius_common._existing_file_hashes(payload),
    }
    path = directory / f"{job}_local_{index}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snellius_common._json_safe(record)), encoding="utf-8")
    return path


def test_the_record_follows_final_b2_and_b1plus_back_to_what_they_loaded(two_days, monkeypatch):
    """D-19, D-22: record_profile_stage.py follows the promoted final B2 to the selected alpha's parallel final B2
    and B1+ to its own run, through their real training metadata (the payload as write_metadata stores it), and
    checks the provenance_at_start they took when they loaded B1, the calibration and the manifest. A B1 rerun
    after them is refused, even when the record is rerun."""

    import subprocess

    from scripts.pipeline import record_profile_stage as recorder

    def git(*args: str) -> str:
        identity = ("-c", "user.name=test", "-c", "user.email=test@example.com")
        return subprocess.run(["git", "-C", str(root), *identity, *args], capture_output=True, text=True, check=True).stdout

    monkeypatch.setenv("B2_UPDATES", "1")
    monkeypatch.setenv("B3_UPDATES", "1")
    monkeypatch.setattr(recorder, "RECORD_UPDATES", 1)  # production k = 10 (D-19); one update is enough here
    cfg = two_days.ctx.cfg
    root = cfg.repo_root
    monkeypatch.setattr(snellius_common, "PROJECT_DIR", root)  # training_code_state() reads this checkout
    two_days.train_b1()
    cfg.result_path("b2_best_alpha.txt").write_text("1.0", encoding="utf-8")
    cfg.result_path("b2_alpha_sweep_summary.csv").write_text("alpha,mean_sharpe\n1.0,0.5\n", encoding="utf-8")
    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "the profile stage's commit")
    head = git("rev-parse", "HEAD").strip()
    profiling = cfg.results_dir / "profiling"
    (profiling / "b3_dual").mkdir(parents=True)
    (profiling / "profile_stage.submitted").write_text(f"{head} 20261008T000000Z\n", encoding="utf-8")
    (profiling / "b3_dual" / "selection.json").write_text(json.dumps({"git_commit": head}), encoding="utf-8")

    two_days.run("train_formula_b2", "--phase", "final", "--alpha", "0.1")  # D-22: a final B2 per alpha
    two_days.run("train_formula_b2", "--phase", "final", "--alpha", "1.0")
    two_days.run("train_formula_b2", "--phase", "final", "--control", "b1plus")
    # D-20: the promoted B3 is recorded too; its lineage is checked in test_submit_pipeline.py, not here.
    for path in snellius_common.b3_run_paths(cfg, 42).values():
        path.write_bytes(f"a promoted B3 {path.name}".encode())
    monkeypatch.setattr(recorder, "check_promoted_b3", lambda *args: {"checked": "elsewhere"})
    metadata = cfg.results_dir / "job_metadata"
    for index, (job, payload) in enumerate(two_days.metadata[-3:]):
        assert payload["provenance_at_start"]["git_commit"] == head
        assert payload["provenance_at_start"]["uncommitted_training_code"] == ""
        store_metadata(metadata, job, payload, index, head)  # with this checkout's commit and status at the end
    two_days.run("train_formula_b2", "--promote")  # b2_promote: alpha 1.0 is the selected one

    _, _, provenance = recorder.write_profile_stage_record(cfg)
    assert provenance["submitted"] == head
    assert (provenance["b2"]["alpha"], provenance["b1plus"]["alpha"]) == (1.0, 0.0)
    assert provenance["b2"]["metadata"] == "train_formula_b2_alpha1.0_local_1.json"

    b1 = cfg.model_path("ppo_b1_doge.zip")
    b1.write_bytes(b1.read_bytes() + b"rerun")  # B1 rerun after final B2 and B1+
    with pytest.raises(RuntimeError, match="not trained from the current B1"):
        recorder.write_profile_stage_record(cfg)


@pytest.mark.parametrize(
    "argv",
    [
        ("--phase", "candidate", "--control", "b1plus"),
        ("--phase", "final", "--control", "b1plus", "--alpha", "0.1"),
        ("--phase", "final", "--control", "b1plus", "--allow-unequal-updates"),
    ],
)
def test_the_b1plus_control_is_the_final_phase_at_alpha_zero_only(two_days, ppo_iterations, argv):
    with pytest.raises(ValueError, match="D-18"):
        two_days.run("train_formula_b2", *argv)
    assert two_days.context_kwargs == [] and ppo_iterations == []  # refused before anything loads


@pytest.mark.parametrize(
    "argv",
    [
        ("--phase", "candidate", "--alpha", "0"),
        ("--phase", "candidate", "--alpha", "-1"),
        ("--phase", "candidate", "--alpha", "nan"),
        ("--phase", "candidate", "--alpha", "inf"),
        ("--phase", "final", "--alpha", "0"),
        ("--phase", "final"),  # the selection file below holds 0
    ],
)
def test_b2_refuses_a_non_positive_alpha(two_days, ppo_iterations, argv):
    """F-05: B2 needs alpha > 0. Today's final B2 with alpha = 0 would be B1+ under B2's name (D-18)."""

    two_days.ctx.cfg.result_path("b2_best_alpha.txt").write_text("0", encoding="utf-8")
    with pytest.raises(ValueError, match="F-05"):
        two_days.run("train_formula_b2", *argv)
    assert ppo_iterations == []


def test_b2_and_b3_train_the_policy_mean_and_critic_with_log_std_frozen(two_days, monkeypatch):
    """D-15b/D-15c test (e): one B2 and one B3 iteration move the mean network and critic, never log sigma."""

    import torch
    from stable_baselines3 import PPO

    from scripts.pipeline import train_formula_b3

    monkeypatch.setenv("B2_UPDATES", "1")
    monkeypatch.setenv("B3_UPDATES", "1")
    b1, _ = two_days.train_b1()
    two_days.ctx.cfg.result_path("b2_best_alpha.txt").write_text("1.0", encoding="utf-8")
    trained: list = []
    original_learn = PPO.learn

    def keeping_learn(self, *args, **kwargs):
        at_learn = {
            "policy": {key: value.detach().clone() for key, value in self.policy.state_dict().items()},
            "optimizer_states": len(self.policy.optimizer.state),
            "frozen": {name for name, parameter in self.policy.named_parameters() if not parameter.requires_grad},
            "adam_steps": 0,
        }

        def count_adam_step(*_):
            at_learn["adam_steps"] += 1

        self.policy.optimizer.register_step_post_hook(count_adam_step)
        trained.append((self, at_learn))
        return original_learn(self, *args, **kwargs)

    monkeypatch.setattr(PPO, "learn", keeping_learn)
    two_days.run("train_formula_b2", "--phase", "final")
    train_formula_b3.run_b3(
        two_days.ctx, *snellius_common.load_train_arrays(two_days.ctx), threshold_d=0.05, lambda_lr=0.5,
        updates=1, fill_paths=2, seed=42, tensorboard_log=None,
    )

    before = b1.policy.state_dict()
    assert len(trained) == 2  # B2, then B3
    for model, at_learn in trained:
        # B2 and B3 start from the same state: B1's weights exactly, fresh Adam moments, only log sigma frozen.
        assert_same_tensors(at_learn["policy"], before)
        assert at_learn["optimizer_states"] == 0
        assert at_learn["frozen"] == {"log_std"}
        after = model.policy.state_dict()
        assert torch.equal(after["log_std"], before["log_std"])
        assert model.policy.log_std not in model.policy.optimizer.state  # never got Adam state
        for prefix in ("mlp_extractor.policy_net.", "action_net.", "value_net."):
            assert any(not torch.equal(after[key], before[key]) for key in before if key.startswith(prefix)), prefix
        # Fresh moments (D-15c): Adam's step counter equals the number of steps this iteration took.
        steps = at_learn["adam_steps"]
        assert steps > 0
        assert int(model.policy.optimizer.state[model.policy.action_net.weight]["step"]) == steps


def test_ppo_trains_on_the_d15a_advantages(two_days, monkeypatch):
    """D-15a test (b'): the array PPO optimizes equals the formula, computed here by hand from SB3's GAE."""

    import pickle

    from stable_baselines3 import PPO
    from stable_baselines3.common.buffers import RolloutBuffer

    from cvar_mm.gym import drawdown_objectives

    alpha = 2.0
    monkeypatch.setenv("B2_UPDATES", "1")
    monkeypatch.setenv("B3_UPDATES", "1")
    two_days.train_b1()
    two_days.ctx.cfg.result_path("b2_best_alpha.txt").write_text(str(alpha), encoding="utf-8")
    seen: dict = {}
    gae = RolloutBuffer.compute_returns_and_advantage
    train = PPO.train
    rollout_end = drawdown_objectives.FullDayMaxDrawdownPPOCallback._on_rollout_end

    def copying_gae(self, *args, **kwargs):
        gae(self, *args, **kwargs)
        seen["gae"], seen["returns"] = self.advantages.copy(), self.returns.copy()

    def recording_rollout_end(self):
        seen["segments"] = list(self._segments)
        rollout_end(self)

    def capturing_train(self):
        seen["trained_on"] = self.rollout_buffer.advantages.copy()
        seen["critic_targets"] = self.rollout_buffer.returns.copy()
        train(self)

    monkeypatch.setattr(RolloutBuffer, "compute_returns_and_advantage", copying_gae)
    monkeypatch.setattr(drawdown_objectives.FullDayMaxDrawdownPPOCallback, "_on_rollout_end", recording_rollout_end)
    monkeypatch.setattr(PPO, "train", capturing_train)
    two_days.run("train_formula_b2", "--phase", "final")

    with open(two_days.ctx.cfg.vecnorm_path("vecnorm_b1"), "rb") as handle:
        b1_normalizer = pickle.load(handle)
    s = float(np.sqrt(b1_normalizer.ret_rms.var + b1_normalizer.epsilon))
    advantages = seen["gae"].astype(np.float64)
    sigma_a = advantages.std()
    standardized = (advantages - advantages.mean()) / sigma_a
    expected = standardized.copy()
    for segment in seen["segments"]:
        others = [other.max_drawdown for other in seen["segments"]
                  if other.day_index == segment.day_index and other is not segment]
        loo = segment.max_drawdown - np.mean(others)
        expected[segment.start:segment.stop, segment.env_index] -= alpha * loo / (s * sigma_a)
    assert len(seen["segments"]) == 2 * 2  # two days x two fill paths
    assert np.abs(expected - standardized).max() > 1e-2  # the paths' drawdowns differ: the risk term is there
    np.testing.assert_allclose(seen["trained_on"], expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(seen["critic_targets"], seen["returns"])  # the critic stays P&L-only


def test_callback_guards_refuse_other_units_and_a_moving_log_std(two_days):
    """D-15a/D-15b: the shared callback refuses raw rewards, another s, a trainable or a moved log sigma."""

    from types import SimpleNamespace

    import torch

    from cvar_mm.gym.drawdown_objectives import FullDayMaxDrawdownPPOCallback, vecnormalize_reward_scale
    from cvar_mm.gym.notebook_support import build_formula_multi_day_replay_env, freeze_vecnorm
    from cvar_mm.gym.sb3_wrapper import StableBaselinesTradingEnvironment

    two_days.train_b1()
    days = [synthetic_day(date) for date in ("2025-01-01", "2025-01-02")]
    env = StableBaselinesTradingEnvironment(build_formula_multi_day_replay_env(
        [day[0] for day in days], [day[1] for day in days], two_days.ctx.cfg,
        daily_market_features=[day[3] for day in days], mode="sequential", num_trajectories=2,
        **two_days.ctx.formula_kwargs,
    ))
    normalizer = freeze_vecnorm(two_days.ctx.cfg.vecnorm_path("vecnorm_b1"), env, two_days.ctx.cfg, norm_reward=True)
    s = vecnormalize_reward_scale(normalizer)
    log_std = torch.zeros(2)

    policy = SimpleNamespace(
        log_std=log_std,
        named_parameters=lambda: [("log_std", log_std)],
        optimizer=SimpleNamespace(register_step_post_hook=lambda hook: None),
    )

    def callback(scale=s):
        guarded = FullDayMaxDrawdownPPOCallback(alpha=1.0, expected_day_indices=(0, 1), reward_scale=scale)
        guarded.model = SimpleNamespace(get_env=lambda: normalizer, policy=policy)
        return guarded

    accepted = callback()
    accepted._on_training_start()  # B1's frozen normalizer, its s, log sigma frozen
    with torch.no_grad():
        log_std.add_(1.0)  # in place, as an optimizer step would move it
    with pytest.raises(RuntimeError, match="log sigma changed"):
        accepted._on_rollout_start()
    with pytest.raises(RuntimeError, match="reward scale s differs"):
        callback(s * (1 + 1e-12))._on_training_start()
    log_std.requires_grad_(True)
    with pytest.raises(RuntimeError, match="frozen at B1's value"):
        callback()._on_training_start()
    log_std.requires_grad_(False)
    normalizer.norm_reward = False
    with pytest.raises(RuntimeError, match="norm_reward=True"):
        callback()._on_training_start()


def test_b2_and_b3_record_the_d15_diagnostics_per_update(two_days, monkeypatch):
    """D-15 test (f): metadata and B3's history hold s, sigma_A, the clipped fraction, epochs, B1's log sigma."""

    import pickle

    from cvar_mm.gym.drawdown_objectives import UPDATE_DIAGNOSTICS

    monkeypatch.setenv("B2_UPDATES", "2")
    monkeypatch.setenv("B3_UPDATES", "2")
    b1, _ = two_days.train_b1()
    two_days.write_threshold()
    two_days.ctx.cfg.result_path("b2_best_alpha.txt").write_text("1.0", encoding="utf-8")
    with open(two_days.ctx.cfg.vecnorm_path("vecnorm_b1"), "rb") as handle:
        b1_normalizer = pickle.load(handle)
    s = float(np.sqrt(b1_normalizer.ret_rms.var + b1_normalizer.epsilon))
    b1_log_std = b1.policy.log_std.detach().numpy().copy()
    minibatches = 22  # Y = 22 steps x 2 paths = 44, minibatch ceil(44 / 32) = 2

    for payload in (two_days.run("train_formula_b2", "--phase", "final"), two_days.run("train_formula_b3")):
        assert payload["reward_scale_s"] == s
        np.testing.assert_array_equal(payload["frozen_log_std"], b1_log_std)
        np.testing.assert_allclose(payload["frozen_std"], np.exp(b1_log_std), rtol=1e-6)
        assert payload["adam_state"] == "fresh (D-15c)"
        assert payload["fine_tuning"] == "D-15/D-15a/D-15b/D-15c"
        rows = payload["training_diagnostics"]
        json.dumps(rows, allow_nan=False)  # plain JSON in the job metadata
        assert len(rows) == 2
        for row in rows:
            assert set(row) == set(UPDATE_DIAGNOSTICS)
            assert row["reward_scale_s"] == s
            assert np.isfinite(row["advantage_std"]) and row["advantage_std"] > 0
            assert row["reward_clip_fraction"] == 0.0  # D-15a expects no clipped reward
            assert row["minibatches_per_epoch"] == minibatches
            assert row["epochs_completed"] == row["adam_steps"] // minibatches
            assert row["kl_stopped"] == (row["adam_steps"] < 10 * minibatches)
            assert np.isfinite(row["approx_kl"]) and 0.0 <= row["clip_fraction"] <= 1.0
            assert np.isfinite(row["value_loss"]) and row["value_loss"] >= 0.0  # D-23
            assert np.isfinite(row["explained_variance"]) and row["explained_variance"] <= 1.0  # D-23
            assert row["mean_log_std"] == pytest.approx(float(np.mean(b1_log_std)), rel=1e-6)
            assert row["adam_steps"] > 0 and np.isfinite(row["adam_damping_ratio"]) and row["adam_damping_ratio"] > 0
        assert rows[1]["elapsed_seconds"] >= rows[0]["elapsed_seconds"]
    history = pd.read_csv(two_days.ctx.cfg.result_path("b3_cvar_lagrangian_history.csv"))
    for key in ("advantage_std", "reward_clip_fraction", "epochs_completed", "adam_damping_ratio", "value_loss",
                "explained_variance"):
        # The CSV keeps ~16 significant digits; the metadata keeps the exact values.
        assert history[key].tolist() == pytest.approx([row[key] for row in rows], rel=1e-12), key


@pytest.mark.parametrize(("target_kl", "adam_steps"), [(-1.0, 0), (None, 10 * 22)])
def test_epochs_and_kl_stops_are_counted_from_real_adam_steps(two_days, monkeypatch, target_kl, adam_steps):
    """D-15 diagnostics at fixed outcomes: target_kl -1 stops before the first step, None runs all 10 epochs."""

    from stable_baselines3 import PPO

    monkeypatch.setattr(snellius_common, "RISK_AWARE_TARGET_KL", target_kl)
    monkeypatch.setenv("B2_UPDATES", "1")
    monkeypatch.setenv("B3_UPDATES", "1")
    two_days.train_b1()
    two_days.ctx.cfg.result_path("b2_best_alpha.txt").write_text("1.0", encoding="utf-8")
    models: list = []
    original_learn = PPO.learn

    def keeping_learn(self, *args, **kwargs):
        models.append(self)
        return original_learn(self, *args, **kwargs)

    monkeypatch.setattr(PPO, "learn", keeping_learn)
    (row,) = two_days.run("train_formula_b2", "--phase", "final")["training_diagnostics"]

    assert row["adam_steps"] == adam_steps
    assert (row["epochs_completed"], row["kl_stopped"]) == ((0, True) if adam_steps == 0 else (10, False))
    model = models[-1]
    adam_state = model.policy.optimizer.state.get(model.policy.action_net.weight, {})
    assert int(adam_state.get("step", 0)) == adam_steps  # Adam's own count, from fresh moments (D-15c)
    assert (row["adam_damping_ratio"] is None) == (adam_steps == 0)


def test_a_moving_log_std_stops_b2_before_anything_is_saved(two_days, monkeypatch):
    """D-15b guard on a real run: log sigma moved in place during PPO.train() raises at the update's close."""

    import torch
    from stable_baselines3 import PPO

    monkeypatch.setenv("B2_UPDATES", "1")
    monkeypatch.setenv("B3_UPDATES", "1")
    two_days.train_b1()
    two_days.ctx.cfg.result_path("b2_best_alpha.txt").write_text("1.0", encoding="utf-8")
    original_train = PPO.train

    def drifting_train(self):
        original_train(self)
        with torch.no_grad():
            self.policy.log_std.add_(1e-3)

    monkeypatch.setattr(PPO, "train", drifting_train)
    with pytest.raises(RuntimeError, match="log sigma changed"):
        two_days.run("train_formula_b2", "--phase", "final")
    assert not two_days.ctx.cfg.model_path("ppo_b2_doge.zip").exists()


def test_b3_elapsed_seconds_include_each_update_train_phase(two_days, monkeypatch):
    """RISK_TRAIN_TIME is sized from elapsed_seconds (k = 10 is fixed, D-19), so each entry ends after PPO.train()."""

    from types import SimpleNamespace

    from stable_baselines3 import PPO

    from cvar_mm.gym import drawdown_objectives

    clock = [0.0]  # the shared B2/B3 callback stamps each update after its PPO.train()
    monkeypatch.setattr(drawdown_objectives, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    original_train = PPO.train

    def train_taking_100_seconds(self):
        original_train(self)
        clock[0] += 100.0  # only the PPO optimisation phase advances the clock

    monkeypatch.setattr(PPO, "train", train_taking_100_seconds)
    monkeypatch.setenv("B2_UPDATES", "3")
    monkeypatch.setenv("B3_UPDATES", "3")
    two_days.train_b1()
    two_days.write_threshold()
    two_days.run("train_formula_b3")

    history = pd.read_csv(two_days.ctx.cfg.result_path("b3_cvar_lagrangian_history.csv"))
    assert history["elapsed_seconds"].tolist() == [100.0, 200.0, 300.0]


CANONICAL_DATES = tuple(day.strftime("%Y-%m-%d") for day in pd.date_range("2025-01-01", "2025-01-29"))
CANONICAL_USAGE = ("candidate_train",) * 4 + ("validation",) * 2 + ("retrospective_holdout",) * 23


@pytest.fixture
def canonical(tmp_path, monkeypatch):
    for name in ("TOTAL_TIMESTEPS", "ABLATION_TIMESTEPS", "B2_UPDATES", "B3_UPDATES", "TRAINING_SEED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TRAINING_FILL_PATHS", "2")
    monkeypatch.setenv("B3_LAMBDA_LR", "0.5")
    monkeypatch.setenv("B1_PASSES", "1")  # F-16
    return Pipeline(tmp_path, monkeypatch, manifest(CANONICAL_DATES, CANONICAL_USAGE))


@pytest.fixture
def canonical_sb3(canonical):
    pytest.importorskip("gymnasium")
    pytest.importorskip("stable_baselines3")
    return canonical


def test_candidate_phase_uses_only_jan_1_to_4_data_and_calibration(canonical):
    *_, dates, _, cfg, formula_kwargs = snellius_common.phase_training_inputs(canonical.ctx, "candidate")
    assert dates == canonical.loaded == list(CANONICAL_DATES[:4])
    assert (cfg.A, cfg.kappa, formula_kwargs["sigma"]) == (JAN_1_4["A"], JAN_1_4["kappa"], JAN_1_4["sigma"])

    canonical.loaded.clear()
    *_, dates, _, cfg, formula_kwargs = snellius_common.phase_training_inputs(canonical.ctx, "final")
    assert dates == canonical.loaded == list(CANONICAL_DATES[:6])
    assert (cfg.A, cfg.kappa, formula_kwargs["sigma"]) == (JAN_1_6["A"], JAN_1_6["kappa"], JAN_1_6["sigma"])
    with pytest.raises(ValueError, match="Unknown training phase"):
        snellius_common.phase_training_inputs(canonical.ctx, "validation")


def test_candidate_jobs_refuse_to_prepare_the_split_or_calibrate(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a candidate job read market data to prepare the pipeline")

    monkeypatch.setattr(snellius_common, "PROJECT_DIR", tmp_path)
    monkeypatch.setenv("DATA_DIR", "/data/doge")
    monkeypatch.setattr(snellius_common, "prepare_manifest", forbidden)
    monkeypatch.setattr(snellius_common, "load_single_day_calibration_arrays", forbidden)
    with pytest.raises(FileNotFoundError, match="prepare_formula_pipeline"):
        snellius_common.load_context(require_prepared=True)


@pytest.fixture
def warm_start_spies(monkeypatch):
    """Record which model/normalizer files PPO and VecNormalize load, and B2's state at learn()."""

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    spies = {"loads": [], "learn": []}
    ppo_load, vecnorm_load, ppo_learn = PPO.load, VecNormalize.load, PPO.learn

    def recording_ppo_load(cls, path, *args, **kwargs):
        spies["loads"].append(Path(path).name)
        return ppo_load(path, *args, **kwargs)

    def recording_vecnorm_load(path, venv):
        spies["loads"].append(Path(path).name)
        return vecnorm_load(path, venv)

    def recording_learn(self, *args, **kwargs):
        policy = {key: value.detach().clone() for key, value in self.policy.state_dict().items()}
        frozen = {name for name, parameter in self.policy.named_parameters() if not parameter.requires_grad}
        spies["learn"].append((policy, self.get_env(), len(self.policy.optimizer.state), frozen))
        return ppo_learn(self, *args, **kwargs)

    monkeypatch.setattr(PPO, "load", classmethod(recording_ppo_load))
    monkeypatch.setattr(VecNormalize, "load", staticmethod(recording_vecnorm_load))
    monkeypatch.setattr(PPO, "learn", recording_learn)
    return spies


def assert_same_tensors(left: dict, right: dict) -> None:
    import torch

    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key


def saved_policy_and_normalizer(cfg, model_stem: str, vecnorm_stem: str):
    import pickle

    from stable_baselines3 import PPO

    policy = PPO.load(str(cfg.model_path(f"{model_stem}.zip")), device="cpu").policy.state_dict()
    with open(cfg.vecnorm_path(vecnorm_stem), "rb") as handle:
        return policy, pickle.load(handle)


def test_b1_cand_trains_on_candidate_days_with_candidate_calibration(canonical_sb3, monkeypatch):
    from scripts.pipeline import train_formula_b1

    built = []
    original_builder = train_formula_b1.build_formula_multi_day_replay_env

    def recording_builder(daily_midprices, daily_dt, config, **kwargs):
        built.append((len(daily_midprices), config.A, config.kappa, kwargs["sigma"]))
        return original_builder(daily_midprices, daily_dt, config, **kwargs)

    monkeypatch.setattr(train_formula_b1, "build_formula_multi_day_replay_env", recording_builder)
    payload = canonical_sb3.run("train_formula_b1", "--phase", "candidate")

    assert canonical_sb3.context_kwargs == [{"require_prepared": True}]  # never prepares or calibrates
    assert canonical_sb3.loaded == list(CANONICAL_DATES[:4])
    assert built == [(4, JAN_1_4["A"], JAN_1_4["kappa"], JAN_1_4["sigma"])]
    assert canonical_sb3.metadata[-1][0] == "train_formula_b1_cand"
    assert payload["calibration_scope"] == "candidate_train_only"
    assert canonical_sb3.ctx.cfg.model_path("ppo_b1_cand_doge.zip").exists()
    assert canonical_sb3.ctx.cfg.vecnorm_path("vecnorm_b1_cand").exists()


@pytest.mark.parametrize("passes", [1, 2])
def test_b1_and_b1_cand_train_b1_passes_over_their_own_days(canonical_sb3, monkeypatch, passes):
    """F-16 (I3): B1 and B1_cand each train B1_PASSES x one complete pass over their own phase's days. The
    frozen value is 1; 2 shows that the budget is the product."""

    from stable_baselines3 import PPO

    requested: list[int] = []
    original_learn = PPO.learn

    def recording_learn(self, total_timesteps, *args, **kwargs):
        requested.append(total_timesteps)
        return original_learn(self, total_timesteps, *args, **kwargs)

    monkeypatch.setattr(PPO, "learn", recording_learn)
    monkeypatch.setenv("B1_PASSES", str(passes))
    for phase, dates in (("candidate", CANONICAL_DATES[:4]), ("final", CANONICAL_DATES[:6])):
        payload = canonical_sb3.run("train_formula_b1", "--phase", phase)
        one_pass = sum(len(synthetic_day(date)[0]) - 1 for date in dates)
        assert requested[-1] == payload["total_timesteps"] == passes * one_pass
        assert (payload["b1_passes"], payload["minimum_complete_day_transitions"]) == (passes, one_pass)
    assert len(requested) == 2


@pytest.mark.parametrize("phase", ["candidate", "final"])
def test_b1_refuses_total_timesteps_and_any_other_budget_before_anything_loads(canonical_sb3, monkeypatch, phase):
    """F-16 (I3): neither B1 nor B1_cand trains any budget but B1_PASSES passes. TOTAL_TIMESTEPS, even empty,
    is refused, B1_PASSES must be a positive integer, and the --total-timesteps option is gone."""

    for value in ("5000", ""):
        monkeypatch.setenv("TOTAL_TIMESTEPS", value)
        with pytest.raises(RuntimeError, match="TOTAL_TIMESTEPS is set.*F-16"):
            canonical_sb3.run("train_formula_b1", "--phase", phase)
    monkeypatch.delenv("TOTAL_TIMESTEPS")
    for value in ("0", "1.0", "2x", "-1"):
        monkeypatch.setenv("B1_PASSES", value)
        with pytest.raises(ValueError, match="B1_PASSES must be a positive integer"):
            canonical_sb3.run("train_formula_b1", "--phase", phase)
    monkeypatch.delenv("B1_PASSES")
    with pytest.raises(RuntimeError, match="B1_PASSES must be set"):
        canonical_sb3.run("train_formula_b1", "--phase", phase)
    monkeypatch.setenv("B1_PASSES", "1")
    with pytest.raises(SystemExit):
        canonical_sb3.run("train_formula_b1", "--phase", phase, "--total-timesteps", "5000")
    assert canonical_sb3.context_kwargs == [] and canonical_sb3.loaded == []


@pytest.mark.parametrize(
    ("phase", "control", "source", "days"),
    [
        ("candidate", None, ("ppo_b1_cand_doge", "vecnorm_b1_cand"), 4),
        ("final", None, ("ppo_b1_doge", "vecnorm_b1"), 6),
        ("final", "b1plus", ("ppo_b1_doge", "vecnorm_b1"), 6),  # B1+ starts where the final B2 does (D-18)
    ],
)
def test_b2_warm_starts_from_its_phase_b1_with_the_frozen_normalizer(
    canonical_sb3, warm_start_spies, monkeypatch, phase, control, source, days
):
    from stable_baselines3 import PPO

    monkeypatch.setenv("B2_UPDATES", "1")
    monkeypatch.setenv("B3_UPDATES", "1")
    canonical_sb3.run("train_formula_b1", "--phase", phase)
    assert canonical_sb3.loaded == list(CANONICAL_DATES[:days])  # the B1 job itself: never Jan 7-29 (D-19)
    source_policy, source_normalizer = saved_policy_and_normalizer(canonical_sb3.ctx.cfg, *source)
    fresh = PPO("MlpPolicy", warm_start_spies["learn"][-1][1], seed=42, device="cpu")
    assert any(  # B1 was trained, so equality below cannot hold by seed alone
        not np.array_equal(fresh.policy.state_dict()[key].numpy(), value.numpy())
        for key, value in source_policy.items()
    )

    canonical_sb3.loaded.clear()
    warm_start_spies["loads"].clear()
    canonical_sb3.ctx.cfg.result_path("b2_best_alpha.txt").write_text("0.1", encoding="utf-8")
    argv = ("--phase", phase) + (("--alpha", "0.1") if phase == "candidate" else ())
    payload = canonical_sb3.run("train_formula_b2", *argv, *(("--control", control) if control else ()))

    assert warm_start_spies["loads"] == [f"{source[1]}.pkl", f"{source[0]}.zip"]
    assert canonical_sb3.context_kwargs[-2:] == [{"require_prepared": phase == "candidate"}] * 2  # B1, then B2
    assert canonical_sb3.loaded == list(CANONICAL_DATES[:days])
    policy_at_learn, normalizer, optimizer_states, frozen = warm_start_spies["learn"][-1]
    scale = float(np.sqrt(source_normalizer.ret_rms.var + source_normalizer.epsilon))
    assert payload["reward_scale_s"] == scale != 1.0
    # All of B1's tensors, value head included: the critic stays in B1's r/s units (D-15a).
    assert_same_tensors(policy_at_learn, source_policy)
    assert optimizer_states == 0 and payload["adam_state"] == "fresh (D-15c)"
    assert frozen == {"log_std"}  # D-15b: only the policy mean and the critic train
    np.testing.assert_array_equal(payload["frozen_log_std"], source_policy["log_std"].numpy())
    np.testing.assert_allclose(payload["frozen_std"], np.exp(source_policy["log_std"].numpy()), rtol=1e-6)
    # B1's frozen normalizer, rewards included: B1's s and clip_reward (D-15a).
    assert normalizer.training is False and normalizer.norm_reward is True
    for name in ("obs_rms", "ret_rms"):
        for statistic in ("mean", "var", "count"):
            np.testing.assert_array_equal(
                getattr(getattr(normalizer, name), statistic), getattr(getattr(source_normalizer, name), statistic)
            )
    assert (normalizer.clip_reward, normalizer.epsilon) == (source_normalizer.clip_reward, source_normalizer.epsilon)
    assert Path(payload["warm_start_model"]).name == f"{source[0]}.zip"
    assert payload["normalizer_frozen_during_b2"] is True
    # Item 8: B1 and B2 record the calibrated A and kappa they trained with, the phase's own (Jan 1-4 or Jan 1-6).
    calibration = JAN_1_4 if phase == "candidate" else JAN_1_6
    for _, recorded in canonical_sb3.metadata[-2:]:
        assert (recorded["experiment_config"].A, recorded["experiment_config"].kappa) == (
            calibration["A"], calibration["kappa"])
        assert recorded["formula_kwargs"]["sigma"] == calibration["sigma"]


def test_b1plus_is_evaluated_on_one_retrospective_day_per_task_like_the_others(canonical_sb3, monkeypatch):
    """D-18: evaluate_formula_model.py --model b1plus loads B1+'s own files and its Jan 7-29 day only. B1+
    trained on Jan 1-6, so the validation split and --alpha are refused before anything loads."""

    monkeypatch.setenv("B2_UPDATES", "1")
    monkeypatch.setenv("B3_UPDATES", "1")
    canonical_sb3.run("train_formula_b1", "--phase", "final")
    canonical_sb3.run("train_formula_b2", "--phase", "final", "--control", "b1plus")
    for argv in (("--split", "validation"), ("--alpha", "0.1")):
        canonical_sb3.loaded.clear()
        calls = len(canonical_sb3.context_kwargs)
        with pytest.raises(ValueError, match="D-18"):
            canonical_sb3.run("evaluate_formula_model", "--model", "b1plus", "--day-index", "0", *argv)
        assert canonical_sb3.loaded == [] and len(canonical_sb3.context_kwargs) == calls

    module = importlib.import_module("scripts.pipeline.evaluate_formula_model")
    calibrations = []
    evaluate = module.evaluate_formula_rl_per_day

    def recording(**kwargs):
        calibrations.append((kwargs["config"].A, kwargs["config"].kappa, kwargs["sigma"]))
        return evaluate(**kwargs)

    monkeypatch.setattr(module, "evaluate_formula_rl_per_day", recording)
    canonical_sb3.loaded.clear()
    payload = canonical_sb3.run("evaluate_formula_model", "--model", "b1plus", "--day-index", "0")
    assert canonical_sb3.loaded == ["2025-01-07"]
    assert (Path(payload["model_path"]).name, Path(payload["vecnorm_path"]).name) == (
        "ppo_b1plus_doge.zip", "vecnorm_b1plus.pkl")
    cfg = canonical_sb3.ctx.cfg
    part = snellius_common.eval_part_path(cfg, model="b1plus", date="2025-01-07")
    assert Path(payload["output_path"]) == part and part.parent.name == "b1plus"
    assert pd.read_csv(part)["Model"].tolist() == ["B1PLUS"]
    assert snellius_common.eval_episode_path(cfg, model="b1plus", date="2025-01-07").exists()

    # Every final model goes through the same branch: its own stems, the Jan 1-6 calibration (not Jan 1-4).
    for model in ("b2", "b3"):  # stand-ins: the branch is chosen by the model key
        model_stem, vecnorm_stem = snellius_common.final_model_stems(model, 42)
        cfg.model_path(f"{model_stem}.zip").write_bytes(cfg.model_path("ppo_b1plus_doge.zip").read_bytes())
        cfg.vecnorm_path(vecnorm_stem).write_bytes(cfg.vecnorm_path("vecnorm_b1plus").read_bytes())
    # D-20: B3 is evaluated only as the profile-stage record hashed it, checked before its day loads.
    b3 = snellius_common.b3_run_paths(cfg, 42)
    record = cfg.results_dir / "profiling" / "profile_stage.sha256"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        snellius_common.sha256_lines(cfg.repo_root, [b3["model_path"], b3["vecnorm_path"]]), encoding="utf-8", newline="\n")
    original = b3["model_path"].read_bytes()
    b3["model_path"].write_bytes(original + b"retrained")
    canonical_sb3.loaded.clear()
    with pytest.raises(RuntimeError, match="differ from the profile-stage record"):
        canonical_sb3.run("evaluate_formula_model", "--model", "b3", "--day-index", "1")
    assert canonical_sb3.loaded == []
    b3["model_path"].write_bytes(original)
    for model in ("b1", "b2", "b3"):
        payload = canonical_sb3.run("evaluate_formula_model", "--model", model, "--day-index", "1")
        assert Path(payload["model_path"]).name == f"{snellius_common.final_model_stems(model, 42)[0]}.zip"
    assert set(calibrations) == {(JAN_1_6["A"], JAN_1_6["kappa"], JAN_1_6["sigma"])}
    assert len(calibrations) == 4 * len(snellius_common.parse_seed_list())  # one call per model and fill seed
    # Item 8: the metadata records the calibration the evaluation used, not the config defaults.
    recorded = payload["experiment_config"]
    assert (recorded.A, recorded.kappa, payload["formula_kwargs"]["sigma"]) == (
        JAN_1_6["A"], JAN_1_6["kappa"], JAN_1_6["sigma"])


def test_b2_validation_evaluation_loads_only_its_jan_5_or_6_day(canonical_sb3, monkeypatch):
    """A profile-stage job under D-19: a B2 candidate's validation reads Jan 5 or 6 only, with the
    candidate model and the Jan 1-4 calibration."""

    module = importlib.import_module("scripts.pipeline.evaluate_formula_model")
    monkeypatch.setenv("B2_UPDATES", "1")
    monkeypatch.setenv("B3_UPDATES", "1")
    canonical_sb3.run("train_formula_b1", "--phase", "candidate")
    canonical_sb3.run("train_formula_b2", "--phase", "candidate", "--alpha", "0.1")
    calibrations = []
    evaluate = module.evaluate_formula_rl_per_day

    def recording(**kwargs):
        calibrations.append((kwargs["config"].A, kwargs["config"].kappa, kwargs["sigma"]))
        return evaluate(**kwargs)

    monkeypatch.setattr(module, "evaluate_formula_rl_per_day", recording)
    for index, date in ((0, "2025-01-05"), (1, "2025-01-06")):
        canonical_sb3.loaded.clear()
        payload = canonical_sb3.run(
            "evaluate_formula_model", "--model", "b2", "--split", "validation", "--alpha", "0.1",
            "--day-index", str(index),
        )
        assert canonical_sb3.loaded == [date]
        assert Path(payload["model_path"]).name == "ppo_b2_candidate_alpha0.1_doge.zip"
        recorded = payload["experiment_config"]  # item 8: the candidate's Jan 1-4 calibration
        assert (recorded.A, recorded.kappa, payload["formula_kwargs"]["sigma"]) == (
            JAN_1_4["A"], JAN_1_4["kappa"], JAN_1_4["sigma"])
    assert set(calibrations) == {(JAN_1_4["A"], JAN_1_4["kappa"], JAN_1_4["sigma"])}


def test_prep_parses_only_jan_1_to_6(tmp_path, monkeypatch):
    """D-19 (d): prep is a profile-stage job. Its calibration parses exactly Jan 1-6; the manifest only hashes
    the Jan 7-29 files' bytes (bookkeeping, as approved for the K7 profile stage)."""

    class Stop(Exception):
        pass

    def calibration_arrays(path):
        loaded.append(DATE_RE.search(Path(path).name).group(0))
        midprices, dt, _, _ = synthetic_day(loaded[-1])
        return midprices, dt, None, midprices - TICK, midprices + TICK

    def stop(*args, **kwargs):  # after the loop over the training rows
        raise Stop

    loaded: list[str] = []
    # This checkout's frozen file, beside this test: snellius_common.PROJECT_DIR follows an exported PROJECT_DIR,
    # which can name another checkout (a stale one in ~/.bashrc failed this test on Snellius).
    frozen_env = Path(__file__).resolve().parents[1] / "configs" / "final_pipeline.env"
    frozen = {name: value for name, value in (
        line[len("export "):].split("=", 1) for line in frozen_env.read_text(encoding="utf-8").splitlines()
        if line.startswith("export CALIBRATION_")
    )}
    for name in ("CALIBRATION_PROXY", "CALIBRATION_START_RULE", "CALIBRATION_DEPTH_TICKS"):
        monkeypatch.setenv(name, frozen[name].strip('"'))
    monkeypatch.setattr(snellius_common, "load_single_day_calibration_arrays", calibration_arrays)
    monkeypatch.setattr(snellius_common, "calibrate_from_days", stop)
    with pytest.raises(Stop):
        snellius_common.load_or_create_train_params(
            ReplayExperimentConfig(repo_root=tmp_path), manifest(CANONICAL_DATES, CANONICAL_USAGE), force=True
        )
    assert loaded == list(CANONICAL_DATES[:6])


def test_the_nb6_ablation_trains_on_jan_1_to_4_and_evaluates_on_jan_5_to_6(canonical_sb3, monkeypatch):
    """A profile-stage job under D-19 (the author's choice of 2026-10-07): NB6 never reads Jan 7-29."""

    from cvar_mm.gym import data_loader

    monkeypatch.setattr(data_loader, "load_single_day_with_features", canonical_sb3._load)  # imported at call time
    canonical_sb3.run("run_feature_ablation", "--feature-set", "base_state")
    assert canonical_sb3.loaded == list(CANONICAL_DATES[:6])  # Jan 1-4 to train, then Jan 5-6 to evaluate
    assert canonical_sb3.ctx.cfg.result_path("ablation/base_state_validation_results.csv").exists()


def test_shap_reads_only_the_first_training_day():
    """A profile-stage job under D-19: SHAP explains B1 on Jan 1 and has no route to Jan 7-29."""

    source = (Path(snellius_common.__file__).parent / "run_shap_diagnostics.py").read_text(encoding="utf-8")
    assert "ctx.train_manifest.iloc[0]" in source
    for route in ("test_manifest", "load_test_day", "validation_manifest", "retrospective"):
        assert route not in source, route


def test_shap_loads_jan_1_only_when_shap_is_installed(canonical_sb3, monkeypatch):
    pytest.importorskip("shap")
    from cvar_mm.gym import data_loader

    canonical_sb3.run("train_formula_b1", "--phase", "final")
    canonical_sb3.loaded.clear()
    monkeypatch.setattr(data_loader, "load_single_day_with_features", canonical_sb3._load)
    canonical_sb3.run("run_shap_diagnostics", "--samples", "20", "--background", "5", "--nsamples", "10")
    assert canonical_sb3.loaded == [CANONICAL_DATES[0]]


def test_dual_profiling_loads_only_jan_1_to_6_and_writes_only_profiling_outputs(canonical_sb3):
    from scripts.pipeline import profile_b3_dual

    canonical_sb3.write_prepared()
    canonical_sb3.run("train_formula_b1", "--phase", "final")
    # H - d < 0 throughout: lambda stays 0, so the three runs coincide and D-12a's tie goes to 0.5
    canonical_sb3.write_threshold(d=1e6)
    root = canonical_sb3.ctx.cfg.repo_root
    before = {path for path in root.rglob("*") if path.is_file()}
    canonical_sb3.loaded.clear()

    canonical_sb3.run("profile_b3_dual")

    assert canonical_sb3.context_kwargs[-1] == {"require_prepared": True}  # M2: the profiler never prepares
    assert canonical_sb3.loaded == list(CANONICAL_DATES[:6])  # never Jan 7-29
    directory = canonical_sb3.ctx.cfg.results_dir / "profiling" / "b3_dual"
    created = {path for path in root.rglob("*") if path.is_file()} - before
    cfg = canonical_sb3.ctx.cfg
    # D-20: besides the profiling outputs, each eta's B3 model and normalizer, and the promoted final B3
    runs = {path for eta in profile_b3_dual.ETA_GRID for path in snellius_common.eta_run_paths(cfg, eta).values()}
    final_b3 = snellius_common.b3_run_paths(cfg, 42)
    assert runs | set(final_b3.values()) <= created
    assert all(directory in path.parents for path in created - runs - set(final_b3.values()))
    assert (directory / "selected_lambda_lr.txt").read_text(encoding="utf-8") == "0.5\n"
    # The tie goes to 0.5, whose run is now the final B3, byte for byte.
    selected_run = snellius_common.eta_run_paths(cfg, 0.5)
    assert {key: path.read_bytes() for key, path in final_b3.items()} == {
        key: path.read_bytes() for key, path in selected_run.items()}
    records = {job: payload for job, payload in canonical_sb3.metadata if job.startswith("profile_b3_dual_eta")}
    assert sorted(records) == [snellius_common.eta_task_job_name(eta) for eta in profile_b3_dual.ETA_GRID]
    for eta in profile_b3_dual.ETA_GRID:
        payload = records[snellius_common.eta_task_job_name(eta)]
        assert (payload["eta"], payload["lambda_learning_rate"], payload["risk_aware_updates"]) == (eta, eta, 10)
        assert (payload["training_seed"], payload["training_fill_paths"], payload["threshold_d"]) == (42, 2, 1e6)
        assert {key: payload[key] for key in snellius_common.B3_RUN_FILES} == snellius_common.eta_run_paths(cfg, eta)
        start = payload["provenance_at_start"]  # D-20: load-time hashes and the start commit
        assert {"git_commit", "uncommitted_training_code"} <= set(start)
        assert start["input_sha256"] == {
            name: snellius_common.file_sha256(path) for name, path in snellius_common.b3_inputs(cfg, 42).items()}
        assert (payload["experiment_config"].A, payload["experiment_config"].kappa) == (JAN_1_6["A"], JAN_1_6["kappa"])
    selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    assert (selection["rule"], selection["updates_per_eta"]) == ("D-12a", profile_b3_dual.PROFILE_UPDATES) == ("D-12a", 10)
    assert selection["selected_lambda_lr"] == 0.5
    assert selection["selection_note"] == profile_b3_dual.TIE_BY_CONSTRUCTION_NOTE
    assert all(summary["kept"] for summary in selection["per_eta"].values())
    page = (directory / "summary.md").read_text(encoding="utf-8")  # the eta page of the go/no-go review
    assert "**Selected: η_λ = 0.5.**" in page and "**Tie by construction.**" in page
    from cvar_mm.gym.drawdown_objectives import UPDATE_DIAGNOSTIC_COLUMNS

    frames = {
        eta: pd.read_csv(directory / f"history_eta{eta}.csv", float_precision="round_trip")
        for eta in profile_b3_dual.ETA_GRID
    }
    for frame in frames.values():
        assert len(frame) == profile_b3_dual.PROFILE_UPDATES
        assert (frame["training_fill_paths"] == 2).all()
        assert (frame["lambda"] == 0.0).all()
        assert set(UPDATE_DIAGNOSTIC_COLUMNS) <= set(frame.columns)  # D-15 and D-21 diagnostics in the histories
        memory = frame["peak_rss_so_far_gib"].to_numpy(dtype=float)
        if sys.platform.startswith("linux"):  # Snellius: the process's running maximum
            assert np.isfinite(memory).all() and (memory > 0).all() and (np.diff(memory) >= 0).all()
            assert (frame["peak_rss_source"] == snellius_common.PEAK_RSS_LINUX).all()
        else:  # WORK_ORDER item 3: null on Windows, with exactly this source
            assert np.isnan(memory).all()
            assert (frame["peak_rss_source"] == "unavailable on this platform").all()
    first = frames[profile_b3_dual.ETA_GRID[0]]["ru_cvar"].tolist()
    assert all(frame["ru_cvar"].tolist() == first for frame in frames.values())  # the tie is genuine here
    hashed = [line.split("  ")[1] for line in (directory / "inputs.sha256").read_text().splitlines()]
    assert hashed == [
        "models/ppo_b1_doge.zip",
        "models/vecnorm_b1.pkl",
        "results/cvar_threshold_d.txt",
        "results/cvar_threshold_b3.json",
        "results/formula_train_params.json",
        "results/formula_manifest.csv",
    ]

    canonical_sb3.write_threshold(d=2e6)  # d changed after profiling: selection must refuse
    with pytest.raises(RuntimeError, match="other B1/d/calibration files"):
        canonical_sb3.run("profile_b3_dual", "--select")
    assert canonical_sb3.context_kwargs[-1] == {"require_prepared": True}
    # The refused selection first moved the earlier choice and the B3 it promoted aside, so neither can stand
    # in (D-12a: no fallback; D-20).
    assert not any((directory / name).exists() for name in profile_b3_dual.SELECTION_OUTPUTS)
    assert not any(path.exists() for path in final_b3.values())
    (archive,) = (directory / "superseded").iterdir()
    assert sorted(path.name for path in archive.iterdir()) == sorted(
        [*profile_b3_dual.SELECTION_OUTPUTS, *(path.name for path in final_b3.values())])


def test_the_memory_probe_samples_once_after_each_update_and_leaves_b3_unchanged(two_days, ppo_iterations, monkeypatch):
    from scripts.pipeline import profile_b3_dual, train_formula_b3

    two_days.train_b1()
    data = snellius_common.load_train_arrays(two_days.ctx)
    monkeypatch.setattr(profile_b3_dual, "PROFILE_UPDATES", 3)
    # The fake reading is the number of B3 PPO.train() calls so far, so each sample shows when it was taken.
    b1_calls = len(ppo_iterations)
    monkeypatch.setattr(
        profile_b3_dual, "peak_rss_so_far_gib", lambda: (float(len(ppo_iterations) - b1_calls), "a test source")
    )
    # and the step count at each sample, which a sample taken during the next rollout would overshoot
    sampled_at: list[int] = []
    sample = profile_b3_dual.PeakRssPerUpdate._sample

    def recording_sample(self):
        sampled_at.append(self.num_timesteps)
        sample(self)

    monkeypatch.setattr(profile_b3_dual.PeakRssPerUpdate, "_sample", recording_sample)

    history, profiled = profile_b3_dual.profile_eta(two_days.ctx, data, 2.0, threshold_d=0.05, fill_paths=2, seed=42)

    assert history["peak_rss_so_far_gib"].tolist() == [1.0, 2.0, 3.0]  # after updates 1, 2 and 3 ...
    assert sampled_at == ppo_iterations[b1_calls:]  # ... and before any step of the next rollout
    assert (history["peak_rss_source"] == "a test source").all()
    plain_run = train_formula_b3.run_b3(
        two_days.ctx, *data, threshold_d=0.05, lambda_lr=2.0, updates=3, fill_paths=2, seed=42, tensorboard_log=None,
    )
    plain = plain_run.callback
    assert history["lambda"].tolist() == plain.lambda_history
    assert history["nu"].tolist() == plain.nu_history
    assert history["ru_cvar"].tolist() == plain.cvar_history
    assert history["violation"].tolist() == plain.violation_history
    assert history["elapsed_seconds"].size == len(plain.diagnostics) == 3
    # D-20's premise: a profiling run is the B3 run itself, policy and critic bit for bit, so promoting the
    # selected run equals retraining B3 with its eta.
    assert_same_tensors(profiled.model.policy.state_dict(), plain_run.model.policy.state_dict())


def test_dotted_b2_stems_keep_their_full_zip_name_and_hash(canonical_sb3, monkeypatch):
    monkeypatch.setenv("B2_UPDATES", "1")
    monkeypatch.setenv("B3_UPDATES", "1")
    canonical_sb3.run("train_formula_b1", "--phase", "candidate")
    payload = canonical_sb3.run("train_formula_b2", "--phase", "candidate", "--alpha", "0.1")

    models = canonical_sb3.ctx.cfg.models_dir
    assert Path(payload["model_path"]) == models / "ppo_b2_candidate_alpha0.1_doge.zip"
    assert (models / "ppo_b2_candidate_alpha0.1_doge.zip").is_file()
    assert not (models / "ppo_b2_candidate_alpha0.zip").exists()  # what with_suffix(".zip") produced
    assert not (models / "ppo_b2_candidate_alpha0.1_doge").exists()  # what SB3 saved for a dotted stem
    assert "payload.model_path" in snellius_common._existing_file_hashes(payload)


def test_no_pipeline_script_derives_a_model_path_with_with_suffix():
    """K8 and item 8 (M5): no `with_suffix(".zip")` anywhere, and every `model_path(...)` in a script the pipeline
    submits (SHAP included) or imports is given an explicit "<stem>.zip". SB3 reads an extension-less path
    first, so a bare stem could load a stale file."""

    import ast

    scripts = Path(__file__).resolve().parents[1] / "scripts" / "pipeline"
    offenders = [path.name for path in scripts.glob("*.py") if 'with_suffix(".zip")' in path.read_text(encoding="utf-8")]
    assert offenders == []
    submit_script = (scripts / "submit_final_pipeline.sh").read_text(encoding="utf-8")
    submitted = set(re.findall(r"scripts/pipeline/(\w+\.py)", submit_script))
    assert "run_shap_diagnostics.py" in submitted
    for name in sorted(submitted | {"common.py", "train_formula_b3.py"}):
        for node in ast.walk(ast.parse((scripts / name).read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "model_path":
                (argument,) = node.args
                last = argument.values[-1] if isinstance(argument, ast.JoinedStr) else argument
                if not (isinstance(last, ast.Constant) and str(last.value).endswith(".zip")):
                    offenders.append(f"{name}:{node.lineno}")
    assert offenders == []


def test_shap_explains_the_seeded_final_models_through_their_zip_paths(tmp_path, monkeypatch):
    """Item 8 (M5): SHAP resolves its files through common.final_model_stems (seeded stems, explicit .zip),
    names no pre-correction B2 file, and refuses --alpha."""

    from scripts.pipeline import run_shap_diagnostics as shap_job

    cfg = ReplayExperimentConfig(repo_root=tmp_path)
    for model in ("b1", "b1plus", "b2", "b3"):
        model_stem, vecnorm_stem = snellius_common.final_model_stems(model, 42)
        assert shap_job.resolve_model_paths(cfg, model, 42) == (
            cfg.models_dir / f"{model_stem}.zip", cfg.vecnorm_path(vecnorm_stem), model)
    assert shap_job.resolve_model_paths(cfg, "b2", 43) == (
        cfg.models_dir / "ppo_b2_doge_seed43.zip", cfg.models_dir / "vecnorm_b2_seed43.pkl", "b2_seed43")
    with pytest.raises(ValueError, match="No final model"):
        shap_job.resolve_model_paths(cfg, "b2_alpha0.1", 42)
    monkeypatch.setattr(shap_job, "load_context", lambda **_: pytest.fail("loaded before the argument check"))
    monkeypatch.setattr(sys, "argv", ["run_shap_diagnostics.py", "--model", "b2", "--alpha", "0.1"])
    with pytest.raises(SystemExit):
        shap_job.main()


def test_metadata_records_the_calibration_a_job_used_never_the_config_defaults(tmp_path, monkeypatch):
    """Item 8: a job's payload["experiment_config"] (the configuration it simulated with) replaces the base
    record; without one, A and kappa are None rather than the ReplayExperimentConfig defaults."""

    monkeypatch.setattr(snellius_common, "PROJECT_DIR", tmp_path)  # build_config() writes under this root
    used = replace(ReplayExperimentConfig(repo_root=tmp_path), A=JAN_1_4["A"], kappa=JAN_1_4["kappa"])
    trained = json.loads(snellius_common.write_metadata("job_used", {"experiment_config": used}).read_text(encoding="utf-8"))
    assert (trained["experiment_config"]["A"], trained["experiment_config"]["kappa"]) == (JAN_1_4["A"], JAN_1_4["kappa"])
    other = json.loads(snellius_common.write_metadata("job_other", {}).read_text(encoding="utf-8"))
    assert (other["experiment_config"]["A"], other["experiment_config"]["kappa"]) == (None, None)
    assert other["experiment_config"]["tick_size"] == ReplayExperimentConfig().tick_size
    assert {"A", "kappa"} <= set(asdict(ReplayExperimentConfig()))  # the fields this test nulls exist


# D-17: the threshold d from stochastic B1 rollouts (collector-level tests: test_b3_threshold_d17.py).
D17_JSON_KEYS = {
    "estimator", "d", "rho", "threshold_multiplier",
    "cvar_stochastic", "nu_stochastic", "cvar_deterministic", "nu_deterministic",
    "b1_log_std", "b1_std", "b1_pinned_share", "fill_seeds", "rollouts_per_seed", "paths_per_estimator",
    "policy_sampling_seed", "training_seed", "constraint_scope", "reference_model",
    "reference_model_path", "reference_vecnorm_path", "train_dates", "trajectory_records",
}


def with_fill_rate(pipeline: Pipeline, A: float = 1.0) -> Pipeline:
    """A quote fills with probability of order 0.3 per 1 s step: inventory builds up and adverse mid moves
    give drawdowns, so the full-day MaxDD differs from path to path. At A = 5 nearly every quote fills,
    wealth only rises and every short synthetic day has MaxDD 0; at A = 0.11 most paths never fill."""

    pipeline.ctx = replace(pipeline.ctx, cfg=replace(pipeline.ctx.cfg, A=A))
    return pipeline


def test_b3_first_rollout_draws_exactly_the_stochastic_d_rollouts(tmp_path, monkeypatch):
    """The D-17 premise: B3's rollout 1 (the warm-start B1) and the d calibration take the same actions
    and give the same Z, bit for bit, when their torch and fill seeds agree."""

    pytest.importorskip("gymnasium")
    pytest.importorskip("stable_baselines3")
    import torch
    from stable_baselines3 import PPO

    from cvar_mm.gym import cvar_lagrangian
    from cvar_mm.gym.cvar_lagrangian import collect_b1_full_day_maxdrawdowns, empirical_ru_cvar
    from cvar_mm.gym.sb3_wrapper import StableBaselinesTradingEnvironment
    from scripts.pipeline import train_formula_b3

    for name in ("TOTAL_TIMESTEPS", "B2_UPDATES", "B3_UPDATES", "TRAINING_SEED"):
        monkeypatch.delenv(name, raising=False)
    pipeline = with_fill_rate(Pipeline(tmp_path, monkeypatch, manifest(["2025-01-01"], ["candidate_train"])))
    pipeline.train_b1()
    # D-21: noise of sigma0 = 0.3 around a mean near 0 almost never reaches the bounds, so move B1's mean action
    # near them: the equalities below then cover clipped and unclipped actions alike.
    b1_path = pipeline.ctx.cfg.model_path("ppo_b1_doge.zip")
    b1 = PPO.load(str(b1_path), device="cpu")
    with torch.no_grad():
        b1.policy.action_net.bias.copy_(torch.tensor([0.9, -0.9]))
    b1.save(str(b1_path))
    train_S, train_dt, train_dates, train_features = snellius_common.load_train_arrays(pipeline.ctx)

    original_learn = PPO.learn
    original_step_async = StableBaselinesTradingEnvironment.step_async
    original_get_action = cvar_lagrangian._BatchedPolicyAgent.get_action
    b3_actions: list[np.ndarray] = []
    calibration_actions: list[np.ndarray] = []

    def learn_from_seed_42(self, *args, **kwargs):
        torch.manual_seed(42)  # the calibration's torch seed for fill seed 42; PPO(seed=42) seeds the fills alike
        return original_learn(self, *args, **kwargs)

    def recording_step_async(self, actions):
        b3_actions.append(np.array(actions))  # what PPO executes: its sample, clipped to the action box
        return original_step_async(self, actions)

    def recording_get_action(self, state):
        actions = original_get_action(self, state)
        calibration_actions.append(np.array(actions))
        return actions

    monkeypatch.setattr(PPO, "learn", learn_from_seed_42)
    monkeypatch.setattr(StableBaselinesTradingEnvironment, "step_async", recording_step_async)
    monkeypatch.setattr(cvar_lagrangian._BatchedPolicyAgent, "get_action", recording_get_action)
    # N = 8 paths: (8, 2) = 16 noise draws per step, the torch routine production's N = 16 uses too.
    run = train_formula_b3.run_b3(
        pipeline.ctx, train_S, train_dt, train_dates, train_features,
        threshold_d=1.0, lambda_lr=0.5, updates=1, fill_paths=8, seed=42, tensorboard_log=None,
    )
    rollout_1 = sorted(run.callback.segment_history[0], key=lambda segment: segment.env_index)
    b3_costs = np.array([segment.max_drawdown for segment in rollout_1])

    calibration = dict(
        daily_midprices=train_S,
        daily_dt_arrays=train_dt,
        daily_market_features=train_features,
        model=PPO.load(str(pipeline.ctx.cfg.model_path("ppo_b1_doge.zip")), device="cpu"),
        vecnorm_path=pipeline.ctx.cfg.vecnorm_path("vecnorm_b1"),
        config=pipeline.ctx.cfg,
        formula_kwargs=pipeline.ctx.formula_kwargs,
        fill_seeds=[42],
        rollouts_per_seed=8,
    )
    stochastic, _ = collect_b1_full_day_maxdrawdowns(**calibration, deterministic=False)
    stochastic_actions = np.stack(calibration_actions)
    calibration_actions.clear()
    deterministic, _ = collect_b1_full_day_maxdrawdowns(**calibration, deterministic=True)

    steps = len(train_S[0]) - 1
    assert np.stack(b3_actions).shape == stochastic_actions.shape == (steps, 8, 2)
    np.testing.assert_array_equal(np.stack(b3_actions), stochastic_actions)  # every sampled action of rollout 1
    assert 0 < np.mean(np.abs(stochastic_actions) == 1.0) < 1  # some actions are clipped, as in PPO
    assert np.ptp(b3_costs) > 0  # the paths' drawdowns differ, so the equality of Z is informative too
    np.testing.assert_array_equal(b3_costs, stochastic)
    assert run.callback.cvar_history[0] == empirical_ru_cvar(stochastic).value  # H at warm start
    assert not np.array_equal(np.stack(calibration_actions), stochastic_actions)  # the mean actions differ
    assert not np.array_equal(stochastic, deterministic)


def test_calibrate_takes_d_from_the_stochastic_pass(two_days, monkeypatch):
    from cvar_mm.gym.cvar_lagrangian import empirical_ru_cvar
    from scripts.pipeline import calibrate_cvar_b3

    two_days.train_b1()
    arrays = {False: 10.0 * np.arange(1.0, 11.0), True: np.arange(1.0, 11.0)}
    calls: list[dict] = []

    def spy(**kwargs):
        calls.append(kwargs)
        mode = kwargs["deterministic"]
        observations = np.zeros((2, *kwargs["model"].model.observation_space.shape), dtype=np.float32)
        kwargs["model"].predict(observations, deterministic=mode)  # two decisions per pass (D-21's share)
        return arrays[mode].copy(), [{"deterministic": mode, "max_drawdown": float(z)} for z in arrays[mode]]

    monkeypatch.setattr(calibrate_cvar_b3, "collect_b1_full_day_maxdrawdowns", spy)
    payload = two_days.run("calibrate_cvar_b3", "--rollouts-per-seed", "5", "--fill-seeds", "42 43")

    assert [call["deterministic"] for call in calls] == [False, True]
    assert calls[0].keys() == calls[1].keys()
    assert all(calls[0][key] is calls[1][key] for key in calls[0] if key != "deterministic")  # same rollouts
    assert calls[0]["fill_seeds"] == [42, 43] and calls[0]["rollouts_per_seed"] == 5
    results = two_days.ctx.cfg
    threshold = json.loads(results.result_path("cvar_threshold_b3.json").read_text(encoding="utf-8"))
    lines = results.result_path("cvar_threshold_d.txt").read_text(encoding="utf-8").splitlines()
    stochastic = empirical_ru_cvar(arrays[False])
    assert threshold["d"] == 0.8 * stochastic.value == payload["d"]
    assert (threshold["cvar_stochastic"], threshold["nu_stochastic"]) == (stochastic.value, stochastic.nu)
    assert threshold["cvar_deterministic"] == empirical_ru_cvar(arrays[True]).value
    assert [float(line) for line in lines] == [threshold["d"], stochastic.value, stochastic.nu]


def test_calibrate_writes_every_d17_field_reproducibly(two_days):
    from cvar_mm.gym.cvar_lagrangian import B3_POLICY_SAMPLING_SEED_RULE, B3_THRESHOLD_ESTIMATOR, empirical_ru_cvar
    from scripts.pipeline.train_formula_b3 import load_b3_threshold

    model, _ = with_fill_rate(two_days).train_b1()
    argv = ("--rollouts-per-seed", "3", "--fill-seeds", "42 43")
    payload = two_days.run("calibrate_cvar_b3", *argv)
    results = two_days.ctx.cfg
    text = results.result_path("cvar_threshold_d.txt").read_text(encoding="utf-8")
    raw = results.result_path("cvar_threshold_b3.json").read_text(encoding="utf-8")
    threshold = json.loads(raw)

    assert set(threshold) == D17_JSON_KEYS
    assert threshold["estimator"] == B3_THRESHOLD_ESTIMATOR == "stochastic (D-17)"
    assert threshold["d"] == 0.8 * threshold["cvar_stochastic"] > 0
    assert float(text.splitlines()[0]) == threshold["d"]
    for mode, name in ((False, "stochastic"), (True, "deterministic")):
        costs = np.array([r["max_drawdown"] for r in threshold["trajectory_records"] if r["deterministic"] is mode])
        assert costs.size == 2 * 2 * 3  # days x fill seeds x paths
        estimate = empirical_ru_cvar(costs)
        assert (threshold[f"cvar_{name}"], threshold[f"nu_{name}"]) == (estimate.value, estimate.nu)
    assert threshold["cvar_stochastic"] != threshold["cvar_deterministic"]
    log_std = model.policy.log_std.detach().cpu().numpy()
    assert threshold["b1_log_std"] == log_std.tolist() and threshold["b1_std"] == np.exp(log_std).tolist()
    share = threshold["b1_pinned_share"]  # D-21 (e): deterministic decisions with |mu_i| >= 1, per action
    assert len(share) == 2 and all(isinstance(value, float) and 0.0 <= value <= 1.0 for value in share)
    assert (threshold["fill_seeds"], threshold["rollouts_per_seed"], threshold["paths_per_estimator"]) == ([42, 43], 3, 12)
    assert threshold["train_dates"] == ["2025-01-01", "2025-01-02"]
    assert threshold["policy_sampling_seed"] == B3_POLICY_SAMPLING_SEED_RULE
    assert Path(threshold["reference_model_path"]).name == "ppo_b1_doge.zip"
    assert {key: payload[key] for key in D17_JSON_KEYS - {"trajectory_records"}} == {
        key: threshold[key] for key in D17_JSON_KEYS - {"trajectory_records"}
    }
    assert set(payload["seconds_per_pass"]) == {"stochastic", "deterministic"}
    assert "payload.reference_model_path" in snellius_common._existing_file_hashes(payload)  # B1's SHA-256
    assert load_b3_threshold(two_days.ctx, 42) == threshold["d"]  # B3 and the eta profiler accept it

    two_days.run("calibrate_cvar_b3", *argv)  # the profile stage hashes these files: a rerun is byte-identical
    assert results.result_path("cvar_threshold_d.txt").read_text(encoding="utf-8") == text
    assert results.result_path("cvar_threshold_b3.json").read_text(encoding="utf-8") == raw
    with pytest.raises(ValueError, match="at least 1"):  # an explicit 0 is refused, not replaced by the default
        two_days.run("calibrate_cvar_b3", "--rollouts-per-seed", "0", "--fill-seeds", "42 43")


def test_b3_refuses_a_threshold_that_d17_did_not_produce(two_days):
    """B3 training and the eta profiler fail closed on a d file pair the D-17 calibration did not write."""

    from scripts.pipeline.train_formula_b3 import load_b3_threshold

    two_days.write_threshold(d=0.05)
    assert load_b3_threshold(two_days.ctx, 42) == 0.05
    path = two_days.ctx.cfg.result_path("cvar_threshold_b3.json")
    calibrated = json.loads(path.read_text(encoding="utf-8"))
    for metadata in (
        {"rho": 0.2, "empirical_nu": 0.0},  # the pre-D-17 file: no estimator, no d
        {**calibrated, "estimator": "deterministic"},
        {**calibrated, "threshold_multiplier": 0.9},
        {**calibrated, "d": 0.06},  # the txt and the JSON disagree
    ):
        path.write_text(json.dumps(metadata), encoding="utf-8")
        with pytest.raises(ValueError, match="D-17"):
            load_b3_threshold(two_days.ctx, 42)
    path.write_text(json.dumps({**calibrated, "rho": 0.25}), encoding="utf-8")
    with pytest.raises(ValueError, match="rho"):
        load_b3_threshold(two_days.ctx, 42)


def test_the_profile_stage_hashes_the_d17_threshold_files(canonical_sb3):
    from scripts.pipeline import profile_b3_dual

    with_fill_rate(canonical_sb3).write_prepared()
    canonical_sb3.train_b1()
    canonical_sb3.loaded.clear()
    canonical_sb3.run("calibrate_cvar_b3", "--rollouts-per-seed", "2", "--fill-seeds", "42 43")
    assert canonical_sb3.loaded == list(CANONICAL_DATES[:6])  # d sees Jan 1-6 only

    canonical_sb3.run("profile_b3_dual", "--eta-index", "0")
    directory = canonical_sb3.ctx.cfg.results_dir / "profiling" / "b3_dual"
    hashes = profile_b3_dual.inputs_path(directory, profile_b3_dual.ETA_GRID[0]).read_text(encoding="utf-8")
    hashed = {name: digest for digest, name in (line.split("  ") for line in hashes.splitlines())}
    root = canonical_sb3.ctx.cfg.repo_root
    for name in ("results/cvar_threshold_d.txt", "results/cvar_threshold_b3.json"):
        assert hashed[name] == snellius_common.file_sha256(root / name)
    job, payload = canonical_sb3.metadata[-1]  # D-20: the eta task's own record, with its load-time hashes
    assert job == snellius_common.eta_task_job_name(profile_b3_dual.ETA_GRID[0])
    loaded = payload["provenance_at_start"]["input_sha256"]
    assert (loaded["threshold_d"], loaded["threshold_json"]) == (
        hashed["results/cvar_threshold_d.txt"], hashed["results/cvar_threshold_b3.json"])


class StopBeforeTraining(Exception):
    """Raised by PPO.learn under stop_before_training(): by then every job has built its reward and its setup."""


def stop_before_training(monkeypatch) -> dict[str, list]:
    """From now on PPO.learn records (model, total_timesteps) and stops before any update, and every
    CjMmCriterion is kept as it was built."""

    from stable_baselines3 import PPO

    from cvar_mm.rewards.reward_funcs import CjMmCriterion

    captured: dict[str, list] = {"learn": [], "rewards": []}
    criterion_init = CjMmCriterion.__init__

    def recording_init(self, *args, **kwargs):
        criterion_init(self, *args, **kwargs)
        captured["rewards"].append(self)

    def stopping_learn(self, total_timesteps, *args, **kwargs):
        captured["learn"].append((self, total_timesteps))
        raise StopBeforeTraining

    monkeypatch.setattr(CjMmCriterion, "__init__", recording_init)
    monkeypatch.setattr(PPO, "learn", stopping_learn)
    return captured


def prepare_every_training_job(pipeline: Pipeline, monkeypatch) -> None:
    """What the jobs read before they train: B1 and B1_cand stand-ins, d, the B2 selection, and k = 10 (D-19)."""

    monkeypatch.setenv("B2_UPDATES", "10")
    monkeypatch.setenv("B3_UPDATES", "10")
    pipeline.train_b1()
    pipeline.train_b1("ppo_b1_cand_doge", "vecnorm_b1_cand")
    pipeline.write_threshold()
    pipeline.ctx.cfg.result_path("b2_best_alpha.txt").write_text("1.0", encoding="utf-8")


TRAINING_JOBS = (
    ("train_formula_b1", ("--phase", "final")),  # B1
    ("train_formula_b1", ("--phase", "candidate")),  # B1_cand
    ("train_formula_b2", ("--phase", "candidate", "--alpha", "1.0")),  # a B2 candidate
    ("train_formula_b2", ("--phase", "final")),  # final B2, the sequential path
    ("train_formula_b2", ("--phase", "final", "--alpha", "1.0")),  # final B2 as the pipeline trains it (D-22)
    ("train_formula_b2", ("--phase", "final", "--control", "b1plus")),  # B1+ (D-18)
    ("train_formula_b3", ()),  # B3's own script
    ("profile_b3_dual", ("--eta-index", "0")),  # B3 as the pipeline trains it: an eta task (D-20)
)


def test_every_training_job_builds_its_reward_with_phi_0(canonical_sb3, monkeypatch):
    """Item 9 (a), D-10: B1, B1_cand, the B2 candidates, final B2, B1+ and B3 each train on a CjMmCriterion
    whose effective per_step_inventory_aversion, after the constructor (whose default is 0.01), is 0.0, and
    that object is the reward their training environment computes."""

    prepare_every_training_job(canonical_sb3, monkeypatch)
    captured = stop_before_training(monkeypatch)
    for script, argv in TRAINING_JOBS:
        built = len(captured["rewards"])
        with pytest.raises(StopBeforeTraining):
            canonical_sb3.run(script, *argv)
        rewards = captured["rewards"][built:]
        model, _ = captured["learn"][-1]
        trained_on = model.get_env().venv.env.base_env.reward_function
        assert len(rewards) == 1 and trained_on is rewards[0], (script, argv)
        assert trained_on.per_step_inventory_aversion == 0.0, (script, argv)
    assert len(captured["learn"]) == len(TRAINING_JOBS)


def test_only_the_fine_tunes_train_at_fine_tune_lr(canonical_sb3, monkeypatch):
    """D-23 tests (a) and (b), job by job: a B2 candidate, final B2 (sequential and parallel), B1+, B3 and an eta
    task build their PPO at FINE_TUNE_LR = 0.003, with an Adam at 0.003 after the warm start from a B1 trained at
    3e-4; B1, B1_cand and an NB6 ablation build theirs at 3e-4. Without FINE_TUNE_LR every fine-tune refuses,
    naming D-23, before it trains, while B1, B1_cand and NB6, which never read it, are unchanged."""

    from cvar_mm.gym import data_loader

    prepare_every_training_job(canonical_sb3, monkeypatch)
    monkeypatch.setattr(data_loader, "load_single_day_with_features", canonical_sb3._load)  # NB6 imports it late
    captured = stop_before_training(monkeypatch)
    b1_family = {"train_formula_b1", "run_feature_ablation"}
    jobs = (*TRAINING_JOBS, ("run_feature_ablation", ("--feature-set", "base_state")))  # the last: an NB6 job
    for exported in (True, False):
        if not exported:
            monkeypatch.delenv("FINE_TUNE_LR")
        for script, argv in jobs:
            learned = len(captured["learn"])
            if not exported and script not in b1_family:
                with pytest.raises(ValueError, match="FINE_TUNE_LR must be a positive number.*D-23"):
                    canonical_sb3.run(script, *argv)
                assert len(captured["learn"]) == learned, (script, argv)  # refused before learn()
                continue
            with pytest.raises(StopBeforeTraining):
                canonical_sb3.run(script, *argv)
            model, _ = captured["learn"][-1]
            expected = 3e-4 if script in b1_family else 0.003
            assert model.learning_rate == expected, (script, argv, exported)
            assert [group["lr"] for group in model.policy.optimizer.param_groups] == [expected], (script, argv)


def setup_at_learn(model, total_timesteps: int) -> dict:
    """What a risk-aware job fixes before its first update, read off the model when learn() starts."""

    import torch

    env = model.get_env()
    optimizer = model.policy.optimizer
    return {
        "policy": {key: value.detach().clone() for key, value in model.policy.state_dict().items()},
        "trainable": sorted(name for name, parameter in model.policy.named_parameters() if parameter.requires_grad),
        "adam_moments": len(optimizer.state),
        "adam_groups": [{key: value for key, value in group.items() if key != "params"} for group in optimizer.param_groups],
        "normalizer": {
            name: (getattr(env, name).mean.copy(), getattr(env, name).var.copy(), getattr(env, name).count)
            for name in ("obs_rms", "ret_rms")
        },
        "normalizer_settings": (
            type(env).__name__, env.training, env.norm_obs, env.norm_reward, env.clip_obs, env.clip_reward,
            env.epsilon, env.gamma,
        ),
        "fill_paths": env.num_envs,
        "updates": total_timesteps / (model.n_steps * env.num_envs),
        "seed": model.seed,
        "ppo": {
            **{name: getattr(model, name) for name in (
                "n_steps", "batch_size", "n_epochs", "target_kl", "normalize_advantage", "ent_coef", "vf_coef",
                "max_grad_norm", "learning_rate", "gamma", "gae_lambda", "use_sde", "policy_kwargs",
            )},
            "clip_range": model.clip_range(1.0),
            "clip_range_vf": None if model.clip_range_vf is None else model.clip_range_vf(1.0),
            "spaces": (model.observation_space, model.action_space),
        },
        "rng": (torch.get_rng_state(), np.random.get_state()),
    }


def test_final_b2_b3_and_b1plus_start_from_identical_setups(canonical_sb3, monkeypatch):
    """Item 9 (b), D-11, D-15/D-15a/D-15b/D-15c, D-18, D-19, D-20, D-22: when learn() starts, final B2 (sequential,
    and the parallel final the pipeline promotes), B1+ and B3 (the eta task the pipeline promotes, and B3's own
    script) have B1's weights, fresh Adam moments with equal settings, B1's frozen normalizer, the frozen log
    sigma, the same N, k = 10, seed 42, the same PPO settings and the same global random states."""

    import torch

    prepare_every_training_job(canonical_sb3, monkeypatch)
    source_policy, source_normalizer = saved_policy_and_normalizer(canonical_sb3.ctx.cfg, "ppo_b1_doge", "vecnorm_b1")
    captured = stop_before_training(monkeypatch)
    setups = {}
    for name, script, argv in (
        ("final B2", "train_formula_b2", ("--phase", "final")),
        ("final B2 (parallel, D-22)", "train_formula_b2", ("--phase", "final", "--alpha", "1.0")),
        ("B1+", "train_formula_b2", ("--phase", "final", "--control", "b1plus")),
        ("B3 (eta task, D-20)", "profile_b3_dual", ("--eta-index", "1")),
        ("B3 (train_formula_b3)", "train_formula_b3", ()),
    ):
        with pytest.raises(StopBeforeTraining):
            canonical_sb3.run(script, *argv)
        setups[name] = setup_at_learn(*captured["learn"][-1])

    b2 = setups["final B2"]
    assert_same_tensors(b2["policy"], source_policy)  # B1's tensors, value head and log sigma included
    assert b2["adam_moments"] == 0  # fresh moments (D-15c)
    assert "log_std" not in b2["trainable"] and b2["trainable"]  # D-15b: the mean and the critic train
    assert b2["normalizer_settings"][:4] == ("VecNormalize", False, True, True)  # frozen, rewards on (D-15a)
    for name in ("obs_rms", "ret_rms"):
        statistics = getattr(source_normalizer, name)
        for mine, theirs in zip(b2["normalizer"][name], (statistics.mean, statistics.var, statistics.count)):
            np.testing.assert_array_equal(mine, theirs)
    assert (b2["fill_paths"], b2["updates"], b2["seed"]) == (2, 10, 42)
    for name, setup in setups.items():
        assert_same_tensors(setup["policy"], b2["policy"])
        for key in ("trainable", "adam_moments", "adam_groups", "normalizer_settings", "fill_paths", "updates",
                    "seed", "ppo"):
            assert setup[key] == b2[key], (name, key)
        for statistic in ("obs_rms", "ret_rms"):
            for mine, theirs in zip(setup["normalizer"][statistic], b2["normalizer"][statistic]):
                np.testing.assert_array_equal(mine, theirs, err_msg=f"{name} {statistic}")
        assert torch.equal(setup["rng"][0], b2["rng"][0]), name
        numpy_state, b2_state = setup["rng"][1], b2["rng"][1]
        assert numpy_state[0] == b2_state[0] and np.array_equal(numpy_state[1], b2_state[1]), name
        assert numpy_state[2:] == b2_state[2:], name


# D-21: every PPO policy explores with a fixed sigma0 = POLICY_STD (0.3 in configs/final_pipeline.env).
SIGMA0 = 0.3
LOG_SIGMA0_FLOAT32 = float(np.float32(math.log(SIGMA0)))  # what log_std_init = log(0.3) stores: -1.2039728164672852
SIGMA0_FLOAT32 = float(np.exp(np.float32(math.log(SIGMA0))))  # frozen_std as recorded (a float32 exp): 0.29999998


@pytest.fixture
def learn_spy(monkeypatch):
    """PPO.learn records each model and its policy's tensors when learning starts."""

    from stable_baselines3 import PPO

    started: list[tuple] = []
    original_learn = PPO.learn

    def recording_learn(self, *args, **kwargs):
        started.append((self, {key: value.detach().clone() for key, value in self.policy.state_dict().items()}))
        return original_learn(self, *args, **kwargs)

    monkeypatch.setattr(PPO, "learn", recording_learn)
    return started


@pytest.mark.parametrize(
    ("script", "argv", "saved"),
    [
        ("train_formula_b1", ("--phase", "final"), "ppo_b1_doge.zip"),  # B1
        ("train_formula_b1", ("--phase", "candidate"), "ppo_b1_cand_doge.zip"),  # B1_cand
        ("run_feature_ablation", ("--feature-set", "base_state"), "ppo_ablation_base_state.zip"),  # an NB6 job
    ],
)
def test_b1_b1_cand_and_nb6_train_with_sigma0_fixed(canonical_sb3, learn_spy, monkeypatch, script, argv, saved):
    """D-21 test (a): a short synthetic B1, B1_cand and NB6 run each ends with log sigma == log(0.3) exactly in
    both dimensions (float32, bit for bit) and requires_grad False, while the mean network's weights changed;
    the saved model holds the same log sigma, and the metadata records sigma0 and that ent_coef has no effect."""

    import torch
    from stable_baselines3 import PPO

    from cvar_mm.gym import data_loader

    monkeypatch.setattr(data_loader, "load_single_day_with_features", canonical_sb3._load)  # NB6 imports it late
    payload = canonical_sb3.run(script, *argv)

    ((model, before),) = learn_spy
    sigma0 = torch.full((2,), math.log(SIGMA0))
    assert model.policy.log_std.dtype == torch.float32 and torch.equal(model.policy.log_std.detach(), sigma0)
    assert model.policy.log_std.requires_grad is False
    after = model.policy.state_dict()
    assert torch.equal(after["log_std"], before["log_std"])
    for prefix in ("mlp_extractor.policy_net.", "action_net."):  # the mean network learned
        assert any(not torch.equal(after[key], before[key]) for key in before if key.startswith(prefix)), prefix
    assert model.ent_coef == canonical_sb3.ctx.cfg.ppo_ent_coef == 0.01  # unchanged, and without effect
    saved_model = PPO.load(str(canonical_sb3.ctx.cfg.model_path(saved)), device="cpu")
    assert torch.equal(saved_model.policy.log_std.detach(), sigma0)
    assert (payload["policy_std"], payload["log_std_trainable"]) == (SIGMA0, False)
    assert payload["policy_log_std"] == [LOG_SIGMA0_FLOAT32] * 2
    assert "D-21" in payload["exploration_noise"] and "ent_coef has no effect" in payload["exploration_noise"]
    # D-23 test (b): B1's 3e-4, although the pipeline exports FINE_TUNE_LR = 0.003 to every job; SB3 set the Adam
    # step size from the schedule at every update, so the group's lr is the one it trained with.
    assert snellius_common.fine_tune_lr() == 0.003
    assert model.learning_rate == saved_model.learning_rate == canonical_sb3.ctx.cfg.ppo_learning_rate == 3e-4
    assert [group["lr"] for group in model.policy.optimizer.param_groups] == [3e-4]


def test_b2_b3_and_b1plus_inherit_sigma0_from_b1(canonical_sb3, monkeypatch):
    """D-21 test (b): a B2 candidate (from B1_cand), final B2, B1+, B3 and an eta task, each warm-started from the
    script-trained B1 of test (a), report frozen_std == [0.3, 0.3] (the float32 exp of log 0.3, as log sigma is
    stored) and policy_std 0.3."""

    from scripts.pipeline import profile_b3_dual

    monkeypatch.setenv("B2_UPDATES", "2")
    monkeypatch.setenv("B3_UPDATES", "2")
    monkeypatch.setattr(profile_b3_dual, "PROFILE_UPDATES", 2)  # production 10 (D-12a); two are enough here
    canonical_sb3.run("train_formula_b1", "--phase", "final")
    canonical_sb3.run("train_formula_b1", "--phase", "candidate")
    canonical_sb3.write_threshold()
    canonical_sb3.ctx.cfg.result_path("b2_best_alpha.txt").write_text("1.0", encoding="utf-8")
    for script, argv in (
        ("train_formula_b2", ("--phase", "candidate", "--alpha", "1.0")),
        ("train_formula_b2", ("--phase", "final")),
        ("train_formula_b2", ("--phase", "final", "--control", "b1plus")),
        ("train_formula_b3", ()),
        ("profile_b3_dual", ("--eta-index", "0")),
    ):
        payload = canonical_sb3.run(script, *argv)
        assert payload["frozen_std"] == [SIGMA0_FLOAT32] * 2, script
        np.testing.assert_allclose(payload["frozen_std"], [SIGMA0, SIGMA0], rtol=0, atol=1e-7)
        assert payload["frozen_log_std"] == [LOG_SIGMA0_FLOAT32] * 2, script
        assert payload["policy_std"] == SIGMA0, script


def test_a_b1_with_another_sigma_is_refused_by_every_warm_start_and_by_the_d_job(canonical_sb3, monkeypatch):
    """D-21 test (c): B1 and B1_cand trained with sigma 0.5 are refused, naming D-21, by a B2 candidate, final B2,
    B1+, B3, an eta task and the d job, before any of them trains or writes a file."""

    from stable_baselines3 import PPO

    monkeypatch.setenv("B2_UPDATES", "2")
    monkeypatch.setenv("B3_UPDATES", "2")
    monkeypatch.setenv("POLICY_STD", "0.5")
    canonical_sb3.run("train_formula_b1", "--phase", "final")
    canonical_sb3.run("train_formula_b1", "--phase", "candidate")
    monkeypatch.setenv("POLICY_STD", "0.3")  # D-21's sigma0; these B1s explored with 0.5
    canonical_sb3.write_threshold()
    cfg = canonical_sb3.ctx.cfg
    cfg.result_path("b2_best_alpha.txt").write_text("1.0", encoding="utf-8")
    root = cfg.repo_root
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    learned: list = []
    monkeypatch.setattr(PPO, "learn", lambda self, *args, **kwargs: learned.append(self))

    for script, argv in (
        ("train_formula_b2", ("--phase", "candidate", "--alpha", "1.0")),
        ("train_formula_b2", ("--phase", "final")),
        ("train_formula_b2", ("--phase", "final", "--control", "b1plus")),
        ("train_formula_b3", ()),
        ("profile_b3_dual", ("--eta-index", "0")),
        ("calibrate_cvar_b3", ("--rollouts-per-seed", "2", "--fill-seeds", "42 43")),
    ):
        with pytest.raises(ValueError, match=r"not POLICY_STD = 0\.3 in every action dimension.*D-21"):
            canonical_sb3.run(script, *argv)
    assert learned == []
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before  # nothing written


def test_the_sigma0_check_is_exact_in_log_sigma_s_own_dtype(monkeypatch):
    """D-21: float32 holds log(0.3) as -1.2039728164672852, so a correct B1's sigma is 0.2999999964, 3.6e-9 from
    0.3, and a literal 1e-12 tolerance around 0.3 would refuse every B1. The reference is POLICY_STD as log
    sigma's dtype holds it: in float32 only that exact value passes, one float32 step away does not."""

    import torch

    monkeypatch.setenv("POLICY_STD", "0.3")
    exact = torch.full((2,), math.log(SIGMA0))
    snellius_common.require_policy_std(exact, source="B1")
    assert abs(math.exp(float(exact[0])) - SIGMA0) > 1e-9  # why the reference is not 0.3 itself
    snellius_common.require_policy_std(torch.full((2,), math.log(SIGMA0), dtype=torch.float64), source="B1")
    one_step = exact.clone()
    one_step[1] = torch.nextafter(exact[1], torch.tensor(0.0))  # one float32 step in one dimension
    for refused in (one_step, torch.zeros(2), torch.full((2,), math.log(0.5)), torch.zeros(0)):
        with pytest.raises(ValueError, match="D-21"):
            snellius_common.require_policy_std(refused, source="B1")
    for raw in ("", "0", "-0.3", "nan", "inf", "0.3x"):
        monkeypatch.setenv("POLICY_STD", raw)
        with pytest.raises(ValueError, match="POLICY_STD must be a positive number.*D-21"):
            snellius_common.policy_std()
    monkeypatch.delenv("POLICY_STD")
    with pytest.raises(ValueError, match="D-21"):
        snellius_common.policy_std()


def test_the_pinned_share_counts_the_deterministic_decisions_at_a_bound():
    """D-21 (e): predict(deterministic=True) returns clip(mu), so |mu_i| >= 1 exactly where an action is -1 or
    +1. The counter passes every action through unchanged and counts the deterministic pass only."""

    from types import SimpleNamespace

    from gymnasium.spaces import Box

    from scripts.pipeline.calibrate_cvar_b3 import PinnedDecisionCounter

    clipped = np.array([[1.0, 0.5], [-1.0, -1.0], [0.25, 0.99999994]], dtype=np.float32)  # clip(mu), float32
    replies = iter([clipped, np.array([[1.0, 1.0]], dtype=np.float32)])  # then a stochastic pass, not counted
    box = Box(low=-np.ones(2, dtype=np.float32), high=np.ones(2, dtype=np.float32), dtype=np.float32)
    model = SimpleNamespace(action_space=box, device="cpu", predict=lambda observation, deterministic: (next(replies), None))
    counter = PinnedDecisionCounter(model)
    actions, _ = counter.predict(np.zeros((3, 7), dtype=np.float32), deterministic=True)
    assert actions is clipped  # passed through unchanged
    counter.predict(np.zeros((1, 7), dtype=np.float32), deterministic=False)
    assert counter.device == "cpu" and counter.share() == [2 / 3, 1 / 3]
    with pytest.raises(RuntimeError, match="no decision.*D-21"):
        PinnedDecisionCounter(model).share()
    wide = Box(low=-2 * np.ones(2, dtype=np.float32), high=2 * np.ones(2, dtype=np.float32), dtype=np.float32)
    with pytest.raises(ValueError, match="action box.*D-21"):
        PinnedDecisionCounter(SimpleNamespace(action_space=wide, device="cpu"))


def test_action_shift_vs_b1_is_zero_at_warm_start_reproducible_and_observes_only(two_days, monkeypatch):
    """D-21 test (f): action_shift_vs_b1 is 0 before the first update, has two non-negative entries per update,
    and is identical for two runs with the same seed. B1's reference actions are the saved B1's own, the last
    update's shift is recomputed from the trained policy, and turning the diagnostic off leaves B3 bit for bit
    unchanged: it draws from no generator training uses."""

    from stable_baselines3 import PPO

    from cvar_mm.gym.drawdown_objectives import ACTION_SHIFT_SAMPLE, CompleteDayRiskCallback
    from cvar_mm.gym.formula_as import FORMULA_FEATURE_NAMES
    from scripts.pipeline import train_formula_b3

    two_days.train_b1()
    data = snellius_common.load_train_arrays(two_days.ctx)

    def b3():  # d = 1e-9 binds: lambda > 0 from the second update on
        return train_formula_b3.run_b3(
            two_days.ctx, *data, threshold_d=1e-9, lambda_lr=0.5, updates=3, fill_paths=2, seed=42,
            tensorboard_log=None,
        )

    runs = [b3(), b3()]
    for run in runs:
        callback = run.callback
        assert callback.action_shift_before_updates == [0.0, 0.0]
        shifts = [row["action_shift_vs_b1"] for row in callback.diagnostics]
        assert len(shifts) == 3 and all(len(shift) == 2 and min(shift) >= 0.0 for shift in shifts)
        assert max(max(shift) for shift in shifts) > 0  # the mean moved, so the zeros above are informative
        steps = sum(len(synthetic_day(date)[0]) - 1 for date in ("2025-01-01", "2025-01-02"))
        assert callback._shift_observations.shape == (min(ACTION_SHIFT_SAMPLE, steps * 2), len(FORMULA_FEATURE_NAMES))
    assert [row["action_shift_vs_b1"] for row in runs[0].callback.diagnostics] == [
        row["action_shift_vs_b1"] for row in runs[1].callback.diagnostics
    ]
    np.testing.assert_array_equal(runs[0].callback._shift_observations, runs[1].callback._shift_observations)

    callback = runs[0].callback
    b1 = PPO.load(str(two_days.ctx.cfg.model_path("ppo_b1_doge.zip")), device="cpu")
    b1_actions, _ = b1.policy.predict(callback._shift_observations, deterministic=True)
    np.testing.assert_array_equal(callback._b1_actions, b1_actions)  # taken before any update: B1's own actions
    trained, _ = runs[0].model.policy.predict(callback._shift_observations, deterministic=True)
    expected = np.abs(trained.astype(np.float64) - b1_actions.astype(np.float64)).mean(axis=0)
    assert callback.diagnostics[-1]["action_shift_vs_b1"] == expected.tolist()

    monkeypatch.setattr(CompleteDayRiskCallback, "_take_action_shift_sample", lambda self, observations: None)
    monkeypatch.setattr(CompleteDayRiskCallback, "_action_shift", lambda self: [0.0, 0.0])
    unobserved = b3()
    assert unobserved.callback._shift_observations is None  # the diagnostic was off
    assert_same_tensors(unobserved.model.policy.state_dict(), runs[0].model.policy.state_dict())
    for name in ("lambda_used_history", "nu_history", "cvar_history"):
        assert getattr(unobserved.callback, name) == getattr(runs[0].callback, name), name


def test_b2_and_b1plus_record_the_action_shift_per_update(two_days, monkeypatch):
    """D-21 (f) through the scripts: final B2 and B1+ record action_shift_vs_b1 in every update's diagnostics,
    two non-negative values, the same for a rerun with the same seed."""

    monkeypatch.setenv("B2_UPDATES", "2")
    monkeypatch.setenv("B3_UPDATES", "2")
    two_days.train_b1()
    two_days.ctx.cfg.result_path("b2_best_alpha.txt").write_text("1.0", encoding="utf-8")
    for argv in (("--phase", "final"), ("--phase", "final", "--control", "b1plus")):
        first, second = (
            [row["action_shift_vs_b1"] for row in two_days.run("train_formula_b2", *argv)["training_diagnostics"]]
            for _ in range(2)
        )
        assert first == second and len(first) == 2, argv
        assert all(len(shift) == 2 and min(shift) >= 0.0 for shift in first), argv


def test_every_fine_tune_records_sb3_value_loss_and_explained_variance_per_update(canonical_sb3, monkeypatch):
    """D-23 test (e): a B2 candidate, final B2 (as the pipeline trains it, D-22), B1+, B3 and an eta task, two
    updates each, record learning_rate 0.003 in their metadata and, in every per-update diagnostics row (metadata;
    the B3 and eta histories), value_loss and explained_variance: present, finite, and exactly SB3's
    train/value_loss and train/explained_variance of that update's own PPO.train()."""

    from stable_baselines3 import PPO

    from scripts.pipeline import profile_b3_dual

    monkeypatch.setenv("B2_UPDATES", "2")
    monkeypatch.setenv("B3_UPDATES", "2")
    monkeypatch.setattr(profile_b3_dual, "PROFILE_UPDATES", 2)  # production 10 (D-12a); two are enough here
    canonical_sb3.train_b1()
    canonical_sb3.train_b1("ppo_b1_cand_doge", "vecnorm_b1_cand")
    canonical_sb3.write_threshold()
    cfg = canonical_sb3.ctx.cfg
    cfg.result_path("b2_best_alpha.txt").write_text("1.0", encoding="utf-8")
    logged: list[tuple] = []  # installed after the B1 stand-ins, so only the fine-tunes' updates
    original_train = PPO.train

    def recording_train(self):
        original_train(self)
        values = self.logger.name_to_value  # .get only, as the callback reads it
        logged.append((values.get("train/value_loss"), values.get("train/explained_variance")))

    monkeypatch.setattr(PPO, "train", recording_train)
    eta = profile_b3_dual.ETA_GRID[0]
    histories = {
        "train_formula_b3": [cfg.result_path("b3_cvar_lagrangian_history.csv")],
        "profile_b3_dual": [  # the D-12a history and the eta run's B3 history (D-20)
            profile_b3_dual.history_path(cfg.results_dir / "profiling" / "b3_dual", eta),
            snellius_common.eta_run_paths(cfg, eta)["history_path"],
        ],
    }
    for script, argv in (
        ("train_formula_b2", ("--phase", "candidate", "--alpha", "1.0")),
        ("train_formula_b2", ("--phase", "final", "--alpha", "1.0")),
        ("train_formula_b2", ("--phase", "final", "--control", "b1plus")),
        ("train_formula_b3", ()),
        ("profile_b3_dual", ("--eta-index", "0")),
    ):
        updates = len(logged)
        payload = canonical_sb3.run(script, *argv)
        assert (payload["learning_rate"], payload["ppo_kwargs"]["learning_rate"]) == (0.003, 0.003), (script, argv)
        rows = payload["training_diagnostics"]
        json.dumps(rows, allow_nan=False)  # plain, finite JSON in the job metadata
        assert len(rows) == 2 and len(logged) == updates + 2, (script, argv)
        assert [(row["value_loss"], row["explained_variance"]) for row in rows] == logged[updates:], (script, argv)
        for row in rows:
            assert math.isfinite(row["value_loss"]) and row["value_loss"] >= 0.0, (script, argv)
            assert math.isfinite(row["explained_variance"]) and row["explained_variance"] <= 1.0, (script, argv)
        for path in histories.get(script, []):
            history = pd.read_csv(path, float_precision="round_trip")
            for key in ("value_loss", "explained_variance"):
                assert history[key].tolist() == pytest.approx([row[key] for row in rows], rel=1e-12), (path.name, key)


# D-22: the three final B2s train in parallel; b2_promote copies the selected alpha's to the final B2's names.
def assert_same_sb3_archive(left: bytes, right: bytes) -> None:
    """Two SB3 model archives hold the same model: the same members, each byte for byte, except "data", whose
    values may differ in start_time only (the wall-clock time learn() started). The zip headers hold each
    member's save time, so two separately trained archives are never equal as files."""

    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(left)) as first, zipfile.ZipFile(io.BytesIO(right)) as second:
        assert first.namelist() == second.namelist()
        for name in first.namelist():
            if name == "data":
                mine, theirs = json.loads(first.read(name)), json.loads(second.read(name))
                assert mine.keys() == theirs.keys()
                assert {key for key in mine if mine[key] != theirs[key]} <= {"start_time"}
            else:
                assert first.read(name) == second.read(name), name


def test_the_promoted_final_b2_is_the_model_the_sequential_path_trains(two_days, monkeypatch):
    """D-22 test (a): one final B2 per alpha trains without reading the B2 selection, and b2_promote copies the
    selected alpha's model and normalizer to the final B2's names byte for byte, with SHA-256. That final B2 is
    the one the sequential path (final B2 trained after the selection, same seed and inputs) produces: the
    normalizer file byte for byte, and every member of the SB3 archive byte for byte except data's start_time.
    The promotion first moves the earlier final B2 aside."""

    monkeypatch.setenv("B2_UPDATES", "2")
    monkeypatch.setenv("B3_UPDATES", "2")
    two_days.train_b1()
    cfg = two_days.ctx.cfg
    selection = snellius_common.b2_selection_paths(cfg, 42)
    selection["b2_best_alpha"].write_text("1.0", encoding="utf-8")
    selection["b2_alpha_sweep_summary"].write_text("alpha,mean_sharpe\n0.1,0.2\n1.0,0.5\n10.0,0.1\n", encoding="utf-8")
    final = snellius_common.b2_final_paths(cfg, 42)

    sequential = two_days.run("train_formula_b2", "--phase", "final")  # trained after the selection
    assert Path(sequential["b2_selection"]) == selection["b2_best_alpha"] and sequential["alpha"] == 1.0
    sequential_files = {key: path.read_bytes() for key, path in final.items()}

    metadata = cfg.results_dir / "job_metadata"
    parallel = {}
    for index, alpha in enumerate((0.1, 1.0, 10.0)):
        payload = two_days.run("train_formula_b2", "--phase", "final", "--alpha", str(alpha))
        assert "b2_selection" not in payload and payload["final_b2"].startswith("parallel (D-22)")
        assert payload["provenance_at_start"]["input_sha256"].keys() == snellius_common.final_b2_inputs(cfg, 42).keys()
        runs = snellius_common.b2_final_run_paths(cfg, alpha, 42)
        assert {key: Path(payload[key]) for key in runs} == runs
        store_metadata(metadata, f"train_formula_b2_alpha{alpha}", payload, index)
        parallel[alpha] = payload
    assert {key: path.read_bytes() for key, path in final.items()} == sequential_files  # the finals left them alone

    promotion = two_days.run("train_formula_b2", "--promote")

    root = cfg.repo_root
    sources = snellius_common.b2_final_run_paths(cfg, 1.0, 42)
    assert (promotion["decision"], promotion["alpha"]) == ("D-22", 1.0)
    assert promotion["source_metadata"] == "train_formula_b2_alpha1.0_local_1.json"
    for key, target in final.items():
        assert target.read_bytes() == sources[key].read_bytes()  # a byte copy of the selected run
        assert promotion["files"][key] == {
            "from": sources[key].relative_to(root).as_posix(),
            "to": target.relative_to(root).as_posix(),
            "sha256": snellius_common.file_sha256(target),
        }
    assert promotion["selection_sha256"] == {
        path.relative_to(root).as_posix(): snellius_common.file_sha256(path) for path in selection.values()
    }
    recorded = json.loads(snellius_common.b2_promotion_path(cfg, 42).read_text(encoding="utf-8"))
    assert {key: recorded[key] for key in ("alpha", "files", "selection_sha256")} == {
        key: promotion[key] for key in ("alpha", "files", "selection_sha256")
    }
    (archive,) = (cfg.results_dir / "b2_promotion_superseded").iterdir()  # the earlier final B2, kept
    assert {key: (archive / path.name).read_bytes() for key, path in final.items()} == sequential_files

    # The promoted final B2 is the sequential final B2: same normalizer bytes, same model bytes but start_time.
    assert final["vecnorm_path"].read_bytes() == sequential_files["vecnorm_path"]
    assert_same_sb3_archive(final["model_path"].read_bytes(), sequential_files["model_path"])
    for key in ("training_diagnostics", "frozen_log_std", "reward_scale_s", "ppo_kwargs", "total_timesteps"):
        if key == "training_diagnostics":
            for mine, theirs in zip(parallel[1.0][key], sequential[key]):
                assert {**mine, "elapsed_seconds": None} == {**theirs, "elapsed_seconds": None}
        else:
            assert parallel[1.0][key] == sequential[key], key
    # ... and not the other alphas' models, so the equality is informative.
    import io
    import zipfile

    def policy_bytes(path: Path) -> bytes:
        with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as archive_file:
            return archive_file.read("policy.pth")

    for alpha in (0.1, 10.0):
        assert policy_bytes(snellius_common.b2_final_run_paths(cfg, alpha, 42)["model_path"]) != policy_bytes(
            final["model_path"]), alpha


def test_b2_promote_refuses_a_missing_or_stale_run_and_promotes_nothing(two_days, monkeypatch):
    """D-22: b2_promote moves the earlier final B2 aside first, then refuses, promoting nothing, if the selected
    alpha's run is missing or its job did not load the current B1, normalizer, calibration and manifest. It takes
    no other option."""

    monkeypatch.setenv("B2_UPDATES", "2")
    monkeypatch.setenv("B3_UPDATES", "2")
    two_days.train_b1()
    cfg = two_days.ctx.cfg
    selection = snellius_common.b2_selection_paths(cfg, 42)
    selection["b2_best_alpha"].write_text("1.0", encoding="utf-8")
    selection["b2_alpha_sweep_summary"].write_text("alpha,mean_sharpe\n1.0,0.5\n", encoding="utf-8")
    final = snellius_common.b2_final_paths(cfg, 42)
    for path in final.values():
        path.write_bytes(b"an earlier final B2")

    with pytest.raises(FileNotFoundError, match="alpha = 1.0 final B2 run is missing"):
        two_days.run("train_formula_b2", "--promote")
    assert not any(path.exists() for path in final.values())  # moved aside, nothing promoted
    assert not snellius_common.b2_promotion_path(cfg, 42).exists()

    payload = two_days.run("train_formula_b2", "--phase", "final", "--alpha", "1.0")
    with pytest.raises(RuntimeError, match="No job metadata of a final B2 with alpha = 1.0"):
        two_days.run("train_formula_b2", "--promote")  # no metadata saved these files
    store_metadata(cfg.results_dir / "job_metadata", "train_formula_b2_alpha1.0", payload, 0)
    b1 = cfg.model_path("ppo_b1_doge.zip")
    b1.write_bytes(b1.read_bytes() + b"rerun")  # B1 rerun after the final B2s
    with pytest.raises(RuntimeError, match="did not load the current warm_start_model"):
        two_days.run("train_formula_b2", "--promote")
    assert not any(path.exists() for path in final.values())
    assert len(list((cfg.results_dir / "b2_promotion_superseded").iterdir())) == 1  # only the first had files to move

    for argv in (("--alpha", "1.0"), ("--phase", "final"), ("--control", "b1plus"), ("--allow-unequal-updates",)):
        with pytest.raises(ValueError, match="--promote takes no other option"):
            two_days.run("train_formula_b2", "--promote", *argv)
