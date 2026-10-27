from __future__ import annotations

import json
import hashlib
import math
import os
import platform
import re
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _discover_project_dir() -> Path:
    env_value = os.environ.get("PROJECT_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve()
    start = Path(__file__).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "cvar_mm").exists():
            return candidate.resolve()
    return Path.cwd().resolve()


PROJECT_DIR = _discover_project_dir()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from cvar_mm.gym.calibration import (  # noqa: E402
    FirstPassageCalibrationConfig,
    calibrate_from_days,
)
from cvar_mm.gym.data_loader import (  # noqa: E402
    load_single_day_calibration_arrays,
    load_single_day_with_features,
)
from cvar_mm.gym.drawdown_objectives import vecnormalize_reward_scale  # noqa: E402
from cvar_mm.gym.experiment_config import ReplayExperimentConfig  # noqa: E402
from cvar_mm.gym.formula_as import FORMULA_FEATURE_NAMES  # noqa: E402


DEFAULT_TRAIN_DAYS = 6
DEFAULT_CANDIDATE_TRAIN_DAYS = 4
DEFAULT_EXPECTED_TEST_DAYS = 23
DEFAULT_ALPHAS = (0.1, 1.0, 10.0)
DEFAULT_FORMULA_GAMMA_MIN = 0.01
DEFAULT_FORMULA_GAMMA_MAX = 0.9
DEFAULT_FORMULA_SKEW_TICKS_MAX = 5.0
DEFAULT_TRAINING_SEED = 42
MANIFEST_COLUMNS = (
    "row_index", "split", "usage", "split_index", "date", "input_path", "input_sha256",
)
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
EXPECTED_DEVELOPMENT_AND_RETROSPECTIVE_DATES = tuple(
    day.strftime("%Y-%m-%d") for day in pd.date_range("2025-01-01", "2025-01-29")
)


@dataclass(frozen=True)
class FormulaPipelineContext:
    project_dir: Path
    data_dir: Path
    cfg: ReplayExperimentConfig
    manifest: pd.DataFrame
    train_params: dict[str, Any]
    formula_kwargs: dict[str, float]

    @property
    def train_manifest(self) -> pd.DataFrame:
        return self.manifest[self.manifest["split"] == "train"].reset_index(drop=True)

    @property
    def candidate_train_manifest(self) -> pd.DataFrame:
        return self.manifest[self.manifest["usage"] == "candidate_train"].reset_index(drop=True)

    @property
    def validation_manifest(self) -> pd.DataFrame:
        return self.manifest[self.manifest["usage"] == "validation"].reset_index(drop=True)

    @property
    def test_manifest(self) -> pd.DataFrame:
        return self.manifest[self.manifest["usage"] == "retrospective_holdout"].reset_index(drop=True)

    @property
    def later_holdout_manifest(self) -> pd.DataFrame:
        return self.manifest[self.manifest["usage"] == "later_untouched_holdout"].reset_index(drop=True)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def required_env(name: str) -> str:
    """A required value with no default (k is frozen by D-19; eta comes from the profile stage): fail closed
    when it is missing (AGENTS §15)."""

    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be set in configs/final_pipeline.env")
    return value


DEFAULT_TRAINING_FILL_PATHS = 16


def training_fill_paths() -> int:
    """N fill paths per training day, the same for B2 and B3 (D-12; default 16)."""

    paths = env_int("TRAINING_FILL_PATHS", DEFAULT_TRAINING_FILL_PATHS)
    if paths < 2:
        raise ValueError("TRAINING_FILL_PATHS must be at least 2 for the leave-one-out baseline (D-12)")
    return paths


POLICY_STD_TOLERANCE = 1e-12  # D-21: |exp(log sigma) - POLICY_STD| per action dimension, in log sigma's dtype


def policy_std() -> float:
    """sigma0 = POLICY_STD, the fixed exploration std of every PPO policy in [-1, 1] action units (D-21: 0.3).

    Read from the environment that configs/final_pipeline.env sets, like k; the submit script pins it to 0.3.
    """

    raw = os.environ.get("POLICY_STD", "").strip()
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not (math.isfinite(value) and value > 0):
        raise ValueError(
            f"POLICY_STD must be a positive number set in configs/final_pipeline.env, got {raw!r} "
            "(DECISIONS D-21: fixed exploration sigma0 = 0.3)"
        )
    return value


def require_policy_std(log_std, *, source: str) -> None:
    """D-21: refuse a policy whose sigma = exp(log sigma) is not POLICY_STD in every action dimension.

    SB3 keeps log sigma in float32, which holds log(0.3) only to about 1e-8: a policy built with
    log_std_init = log(0.3) has sigma = 0.2999999964. The reference is therefore exp(log POLICY_STD rounded to
    log sigma's own dtype), within POLICY_STD_TOLERANCE; in float32 only that exact value passes.
    """

    import torch

    sigma0 = policy_std()
    log_std = torch.as_tensor(log_std).detach().cpu()
    sigma = np.exp(log_std.numpy().astype(np.float64))
    expected = np.exp(torch.full_like(log_std, math.log(sigma0)).numpy().astype(np.float64))
    if log_std.ndim != 1 or log_std.numel() == 0 or not np.all(np.abs(sigma - expected) <= POLICY_STD_TOLERANCE):
        raise ValueError(
            f"{source} has sigma = exp(log sigma) = {sigma.tolist()}, not POLICY_STD = {sigma0} in every action "
            "dimension (DECISIONS D-21: fixed exploration sigma0 = 0.3)"
        )


def fixed_std_ppo(env, ppo_kwargs: dict[str, Any], **kwargs):
    """D-21: ``PPO("MlpPolicy", env, **ppo_kwargs, **kwargs)`` exploring with sigma0 = POLICY_STD, fixed.

    The one way B1, B1_cand and the NB6 ablations build their PPO. Its state-independent log sigma starts at
    log(POLICY_STD) and is frozen before learn(), so only the policy mean and the critic train. The Gaussian
    entropy is then constant and ent_coef (0.01) has no effect, as in B2 and B3 (D-15b), which inherit sigma0
    from B1 through warm_start_from.
    """

    from stable_baselines3 import PPO

    sigma0 = policy_std()
    model = PPO("MlpPolicy", env, **ppo_kwargs, policy_kwargs={"log_std_init": math.log(sigma0)}, **kwargs)
    model.policy.log_std.requires_grad_(False)
    require_policy_std(model.policy.log_std, source="The new PPO policy")
    return model


def fixed_std_record(model) -> dict[str, Any]:
    """What B1, B1_cand and NB6 record of their fixed exploration noise (D-21)."""

    return {
        "policy_std": policy_std(),
        "policy_log_std": model.policy.log_std.detach().cpu().numpy().tolist(),
        "log_std_trainable": bool(model.policy.log_std.requires_grad),
        "exploration_noise": "log sigma fixed at log(POLICY_STD), not learned (D-21); ent_coef has no effect, "
                             "since the Gaussian entropy is then constant (as under D-15b)",
    }


def risk_aware_updates(model: str, *, allow_unequal: bool = False) -> int:
    """Return k, the number of B2/B3 PPO iterations (D-11: B2 and B3 use the same k)."""

    own = f"{model.upper()}_UPDATES"
    names = (own,) if allow_unequal else ("B2_UPDATES", "B3_UPDATES")
    values: dict[str, int] = {}
    for name in names:
        raw = required_env(name)
        values[name] = int(raw)
        if values[name] < 1:
            raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    if len(set(values.values())) > 1:
        raise RuntimeError(
            f"B2_UPDATES={values['B2_UPDATES']} differs from B3_UPDATES={values['B3_UPDATES']}: "
            "D-11 gives B2 and B3 the same k. Pass --allow-unequal-updates to override."
        )
    return values[own]


# D-15 schedule of every risk-aware PPO iteration (B2 all phases, B3, the eta profiler, B1+).
RISK_AWARE_MINIBATCHES = 32
RISK_AWARE_EPOCHS = 10
RISK_AWARE_TARGET_KL = 0.02


def fine_tune_lr() -> float:
    """The PPO learning rate of every fine-tune (B2 all phases, B3, the eta profiler, B1+): FINE_TUNE_LR (D-23: 0.003).

    Read from the environment that configs/final_pipeline.env sets, like POLICY_STD and k; the submit script pins
    it to 0.003. B1, B1_cand and the NB6 ablations never read it: they keep cfg.ppo_learning_rate (3e-4).
    """

    raw = os.environ.get("FINE_TUNE_LR", "").strip()
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not (math.isfinite(value) and value > 0):
        raise ValueError(
            f"FINE_TUNE_LR must be a positive number set in configs/final_pipeline.env, got {raw!r} "
            "(DECISIONS D-23: fine-tune learning rate 3e-3)"
        )
    return value


def risk_aware_ppo_kwargs(cfg: ReplayExperimentConfig, complete_batch_steps: int, num_envs: int) -> dict[str, Any]:
    """PPO settings shared by B2, B3, the eta profiler and B1+ (DECISIONS D-15, D-23).

    One iteration is one balanced complete-day batch of Y = n_steps x num_envs
    transitions, split into minibatches of ceil(Y / 32) (the author's choice at
    implementation): exactly 32 per epoch for any Y >= 1,024, the last shorter
    by fewer than 32 transitions, and never more than 32, so with 10 epochs an
    iteration takes at most 320 Adam steps. SB3 stops the remaining epochs once
    the approximate KL exceeds 1.5 x target_kl. The learning rate is
    FINE_TUNE_LR (D-23: 3e-3, ten times B1's 3e-4), so that target_kl, not the
    learning rate, bounds each iteration. The other settings are B1's.
    """

    transitions = int(complete_batch_steps) * int(num_envs)
    if complete_batch_steps < 1 or num_envs < 1 or transitions < RISK_AWARE_MINIBATCHES:
        raise ValueError(
            f"A risk-aware batch needs at least {RISK_AWARE_MINIBATCHES} transitions for "
            f"{RISK_AWARE_MINIBATCHES} minibatches (D-15); got n_steps={complete_batch_steps}, "
            f"num_envs={num_envs}"
        )
    kwargs = cfg.ppo_kwargs()
    kwargs.update(
        n_steps=int(complete_batch_steps),
        batch_size=-(-transitions // RISK_AWARE_MINIBATCHES),  # integer ceil
        n_epochs=RISK_AWARE_EPOCHS,
        learning_rate=fine_tune_lr(),  # D-23; replaces B1's value in place, so the key order is unchanged
        target_kl=RISK_AWARE_TARGET_KL,
        ent_coef=cfg.ppo_ent_coef,
        normalize_advantage=False,
    )
    return kwargs


def parse_seed_list(raw: str | None = None, *, env_name: str = "EVALUATION_FILL_SEEDS") -> list[int]:
    value = raw or os.environ.get(env_name, "42 43")  # F-07
    seeds = [int(part) for part in value.replace(",", " ").split()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"{env_name} must contain one or more unique integer seeds")
    return seeds


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(root: Path, paths: list[Path]) -> str:
    """``sha256sum`` format relative to the project root (checked with ``sha256sum -c``). Write it with
    ``newline="\\n"``: Git Bash's ``sha256sum -c`` rejects CRLF lines."""

    return "".join(f"{file_sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in paths)


PEAK_RSS_UNAVAILABLE = "unavailable on this platform"
PEAK_RSS_LINUX = "resource.getrusage(RUSAGE_SELF).ru_maxrss in KiB (Linux): the process's running maximum"


def peak_rss_so_far_gib(platform_name: str | None = None) -> tuple[float | None, str]:
    """The process's peak resident memory so far, in GiB, and where it comes from.

    Measured on Linux only: ``ru_maxrss`` is in KiB there (bytes on macOS), and
    Windows has no ``resource`` module, so elsewhere the value is None.
    """

    platform_name = sys.platform if platform_name is None else platform_name
    if not platform_name.startswith("linux"):
        return None, PEAK_RSS_UNAVAILABLE
    import resource  # POSIX only, so imported here rather than at module level

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, PEAK_RSS_LINUX


def seeded_stem(stem: str, seed: int) -> str:
    """Preserve canonical seed-42 names while preventing other seeds overwriting them."""

    return stem if seed == DEFAULT_TRAINING_SEED else f"{stem}_seed{seed}"


def seeded_result_name(filename: str, seed: int) -> str:
    if seed == DEFAULT_TRAINING_SEED:
        return filename
    path = Path(filename)
    return f"{path.stem}_seed{seed}{path.suffix}"


def parse_alpha_list(raw: str | None = None) -> list[float]:
    value = raw or os.environ.get("B2_ALPHAS")
    if not value:
        return list(DEFAULT_ALPHAS)
    return [float(part) for part in value.replace(",", " ").split()]


def calibration_config_from_env(tick_size: float) -> FirstPassageCalibrationConfig:
    """Make all execution-proxy profiling choices visible in job metadata."""

    raw_depths = os.environ.get("CALIBRATION_DEPTH_TICKS")
    proxy = os.environ.get("CALIBRATION_PROXY")
    start_rule = os.environ.get("CALIBRATION_START_RULE")
    if not raw_depths or not proxy or not start_rule:
        raise RuntimeError(
            "CALIBRATION_PROXY, CALIBRATION_START_RULE, and CALIBRATION_DEPTH_TICKS "
            "must be explicitly set after training-only calibration profiling"
        )
    depths = tuple(float(part) for part in raw_depths.replace(",", " ").split())
    return FirstPassageCalibrationConfig(
        tick_size=tick_size,
        depth_ticks=depths,
        censor_horizon_seconds=env_float("CALIBRATION_CENSOR_SECONDS", 15.0),
        execution_proxy=proxy,
        exposure_start_rule=start_rule,
        exposure_stride_seconds=env_float("CALIBRATION_START_STRIDE_SECONDS", 15.0),  # F-03
        min_hits_per_depth=env_int("CALIBRATION_MIN_HITS", 10),
        volatility_window_seconds=env_float("CALIBRATION_VOL_WINDOW_SECONDS", 600.0),
    )


def _curve_payload(curve) -> dict[str, Any]:
    return {
        "side": curve.side,
        "depth_ticks": curve.depth_ticks.tolist(),
        "depth_price": curve.depth_price.tolist(),
        "hits": curve.hits.tolist(),
        "total_exposure_seconds": curve.total_exposure_seconds.tolist(),
        "exposures": curve.exposures.tolist(),
        "censored": curve.censored.tolist(),
        "intensity_per_second": curve.intensity_per_second.tolist(),
    }


def require_snellius_paths() -> None:
    data_dir = os.environ.get("DATA_DIR")
    if not data_dir:
        raise RuntimeError("DATA_DIR must be exported on Snellius before running the pipeline.")
    if "\\" in data_dir or re.match(r"^[A-Za-z]:", data_dir):
        raise RuntimeError(f"DATA_DIR must be a Linux path on Snellius, got: {data_dir}")


def build_config() -> ReplayExperimentConfig:
    data_dir = os.environ.get("DATA_DIR")
    cfg = ReplayExperimentConfig(
        repo_root=PROJECT_DIR,
        datasets_subdir=data_dir or ReplayExperimentConfig().datasets_subdir,
    )
    eval_rollouts = env_int("EVALUATION_ROLLOUTS", cfg.evaluation_rollouts)
    feature_window = env_int("FEATURE_WINDOW", cfg.feature_window)
    fixed_clock_seconds = env_float("FIXED_CLOCK_SECONDS", cfg.fixed_clock_seconds)
    gamma_range = (
        env_float("FORMULA_GAMMA_MIN", DEFAULT_FORMULA_GAMMA_MIN),
        env_float("FORMULA_GAMMA_MAX", DEFAULT_FORMULA_GAMMA_MAX),
    )
    return replace(
        cfg,
        evaluation_rollouts=eval_rollouts,
        feature_window=feature_window,
        fixed_clock_seconds=fixed_clock_seconds,
        as_gamma_range=gamma_range,
    )


def manifest_path(cfg: ReplayExperimentConfig) -> Path:
    return cfg.result_path("formula_manifest.csv")


def train_params_json_path(cfg: ReplayExperimentConfig) -> Path:
    return cfg.result_path("formula_train_params.json")


def train_params_csv_path(cfg: ReplayExperimentConfig) -> Path:
    return cfg.result_path("formula_train_params.csv")


def _extract_date(path: Path) -> str:
    match = DATE_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not extract YYYY-MM-DD date from {path.name}")
    return match.group(0)


def prepare_manifest(
    cfg: ReplayExperimentConfig,
    *,
    train_days: int = DEFAULT_TRAIN_DAYS,
    expected_test_days: int = DEFAULT_EXPECTED_TEST_DAYS,
    max_days: int | None = None,
) -> pd.DataFrame:
    files = sorted(cfg.datasets_dir.glob(f"binance_book_snapshot_25_*_{cfg.pair}.csv"))
    if max_days is not None:
        files = files[:max_days]
    if not files:
        raise FileNotFoundError(f"No DOGE files found in DATA_DIR={cfg.datasets_dir}")
    if len(files) < train_days + expected_test_days:
        raise ValueError(
            f"Expected at least {train_days + expected_test_days} files "
            f"({train_days} train + {expected_test_days} test), found {len(files)}."
        )
    observed_dates = tuple(_extract_date(path) for path in files[: train_days + expected_test_days])
    if observed_dates != EXPECTED_DEVELOPMENT_AND_RETROSPECTIVE_DATES:
        raise ValueError(
            "The corrected split requires exactly 2025-01-01..2025-01-06 for "
            "development and 2025-01-07..2025-01-29 for frozen retrospective evaluation"
        )

    rows: list[dict[str, Any]] = []
    for row_index, path in enumerate(files):
        split = "train" if row_index < train_days else "test"
        if row_index < DEFAULT_CANDIDATE_TRAIN_DAYS:
            usage = "candidate_train"
            split_index = row_index
        elif row_index < train_days:
            usage = "validation"
            split_index = row_index - DEFAULT_CANDIDATE_TRAIN_DAYS
        elif row_index < train_days + expected_test_days:
            usage = "retrospective_holdout"
            split_index = row_index - train_days
        else:
            split = "holdout"
            usage = "later_untouched_holdout"
            split_index = row_index - train_days - expected_test_days
        rows.append(
            {
                "row_index": row_index,
                "split": split,
                "usage": usage,
                "split_index": split_index,
                "date": _extract_date(path),
                "input_path": str(path.resolve()),
                "input_sha256": file_sha256(path),
            }
        )

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    out_path = manifest_path(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)
    return manifest


def load_or_prepare_manifest(cfg: ReplayExperimentConfig) -> pd.DataFrame:
    path = manifest_path(cfg)
    if not path.exists():
        return prepare_manifest(cfg)
    manifest = pd.read_csv(path)
    missing = set(MANIFEST_COLUMNS).difference(manifest.columns)
    if missing:
        raise ValueError(f"Formula manifest missing columns: {sorted(missing)}")
    expected_dates = EXPECTED_DEVELOPMENT_AND_RETROSPECTIVE_DATES
    observed_dates = tuple(manifest["date"].astype(str).iloc[: len(expected_dates)])
    path_dates = tuple(_extract_date(Path(path)) for path in manifest["input_path"].astype(str))
    if (
        observed_dates != expected_dates
        or path_dates[: len(expected_dates)] != expected_dates
        or tuple(manifest["date"].astype(str)) != path_dates
        or any(date <= expected_dates[-1] for date in path_dates[len(expected_dates):])
        or len(set(path_dates)) != len(path_dates)
    ):
        raise ValueError(
            "The cached corrected split requires exactly 2025-01-01..2025-01-06 for "
            "development and 2025-01-07..2025-01-29 for frozen retrospective evaluation"
        )
    expected_usage = (
        ("candidate_train",) * DEFAULT_CANDIDATE_TRAIN_DAYS
        + ("validation",) * (DEFAULT_TRAIN_DAYS - DEFAULT_CANDIDATE_TRAIN_DAYS)
        + ("retrospective_holdout",) * DEFAULT_EXPECTED_TEST_DAYS
    )
    expected_split = ("train",) * DEFAULT_TRAIN_DAYS + ("test",) * DEFAULT_EXPECTED_TEST_DAYS
    observed_usage = tuple(manifest["usage"].astype(str).iloc[: len(expected_usage)])
    observed_split = tuple(manifest["split"].astype(str).iloc[: len(expected_split)])
    later_usage = set(manifest["usage"].astype(str).iloc[len(expected_usage):])
    later_split = set(manifest["split"].astype(str).iloc[len(expected_split):])
    if (
        observed_usage != expected_usage
        or observed_split != expected_split
        or later_usage.difference({"later_untouched_holdout"})
        or later_split.difference({"holdout"})
    ):
        raise ValueError("Cached manifest has invalid development, retrospective, or later-holdout usage")
    return manifest


def load_single_manifest_day(row: pd.Series, cfg: ReplayExperimentConfig):
    return load_single_day_with_features(
        str(row["input_path"]),
        tick_size=cfg.tick_size,
    )


def load_train_arrays(ctx: FormulaPipelineContext):
    return load_manifest_arrays(ctx, ctx.train_manifest, label="train")


def load_candidate_train_arrays(ctx: FormulaPipelineContext):
    return load_manifest_arrays(ctx, ctx.candidate_train_manifest, label="candidate train")


def load_validation_arrays(ctx: FormulaPipelineContext):
    return load_manifest_arrays(ctx, ctx.validation_manifest, label="validation")


def load_manifest_arrays(ctx: FormulaPipelineContext, manifest: pd.DataFrame, *, label: str):
    train_S, train_dt, train_dates, train_features = [], [], [], []
    for row in manifest.itertuples(index=False):
        S, dt, _, features = load_single_day_with_features(
            row.input_path,
            tick_size=ctx.cfg.tick_size,
        )
        train_S.append(S)
        train_dt.append(dt)
        train_dates.append(str(row.date))
        train_features.append(features)
        print(f"Loaded {label} {row.date}: {len(S):,} snapshots")
    return train_S, train_dt, train_dates, train_features


def minimum_complete_day_steps(daily_midprices: list) -> int:
    if not daily_midprices:
        raise ValueError("At least one training day is required")
    return int(sum(len(prices) - 1 for prices in daily_midprices))


def require_complete_day_budget(total_timesteps: int, daily_midprices: list) -> int:
    required = minimum_complete_day_steps(daily_midprices)
    if total_timesteps < required:
        raise ValueError(
            f"Training budget {total_timesteps:,} is below the {required:,} transitions "
            "required to complete every declared training day once"
        )
    return required


def assert_complete_day_coverage(env, expected_dates: list[str]) -> dict[str, list[int]]:
    midprice_model = env.model_dynamics.midprice_model
    report = midprice_model.coverage_report
    missing = [date for date, count in zip(expected_dates, report["completed"]) if count < 1]
    if missing:
        raise RuntimeError(f"Training finished without completing declared days: {missing}")
    return report


def load_test_day(ctx: FormulaPipelineContext, day_index: int):
    test_manifest = ctx.test_manifest
    if day_index < 0 or day_index >= len(test_manifest):
        raise IndexError(f"Test day index {day_index} outside 0..{len(test_manifest)-1}")
    row = test_manifest.iloc[day_index]
    S, dt, _, features = load_single_day_with_features(
        str(row["input_path"]),
        tick_size=ctx.cfg.tick_size,
    )
    return S, dt, str(row["date"]), features


def load_validation_day(ctx: FormulaPipelineContext, day_index: int):
    manifest = ctx.validation_manifest
    if day_index < 0 or day_index >= len(manifest):
        raise IndexError(f"Validation day index {day_index} outside 0..{len(manifest)-1}")
    row = manifest.iloc[day_index]
    S, dt, _, features = load_single_day_with_features(
        str(row["input_path"]), tick_size=ctx.cfg.tick_size
    )
    return S, dt, str(row["date"]), features


def load_or_create_train_params(
    cfg: ReplayExperimentConfig,
    manifest: pd.DataFrame,
    *,
    force: bool = False,
) -> dict[str, Any]:
    json_path = train_params_json_path(cfg)
    if json_path.exists() and not force:
        return json.loads(json_path.read_text(encoding="utf-8"))

    daily_midprices, daily_dt, daily_bid, daily_ask, dates = [], [], [], [], []
    for row in manifest[manifest["split"] == "train"].itertuples(index=False):
        S, dt, _, bid, ask = load_single_day_calibration_arrays(row.input_path)
        daily_midprices.append(S)
        daily_dt.append(dt)
        daily_bid.append(bid)
        daily_ask.append(ask)
        dates.append(str(row.date))
        print(f"Loaded calibration train day {row.date}: {len(S):,} snapshots")

    calibration_config = calibration_config_from_env(cfg.tick_size)
    result = calibrate_from_days(
        daily_midprices,
        daily_dt,
        config=calibration_config,
        daily_best_bid=daily_bid,
        daily_best_ask=daily_ask,
    )
    candidate_count = int((manifest["usage"] == "candidate_train").sum())
    candidate_result = calibrate_from_days(
        daily_midprices[:candidate_count],
        daily_dt[:candidate_count],
        config=calibration_config,
        daily_best_bid=daily_bid[:candidate_count],
        daily_best_ask=daily_ask[:candidate_count],
    )
    diagnostics = result.diagnostics
    rows: list[dict[str, Any]] = []
    for date in dates:
        rows.append(
            {"date": date, "sigma": result.sigma, "A": result.A, "kappa": result.kappa}
        )

    frame = pd.DataFrame(rows)
    payload = {
        "sigma_train": result.sigma,
        "A_train": result.A,
        "kappa_train": result.kappa,
        "aggregation": "pooled_bid_ask_hits_and_exposure_over_training_days",
        "calibration_config": asdict(calibration_config),
        "pooled_fit": asdict(diagnostics.pooled_fit),
        "bid_fit_diagnostic": None if diagnostics.bid_fit is None else asdict(diagnostics.bid_fit),
        "ask_fit_diagnostic": None if diagnostics.ask_fit is None else asdict(diagnostics.ask_fit),
        "pooled_curve": _curve_payload(diagnostics.pooled_curve),
        "bid_curve_diagnostic": _curve_payload(diagnostics.bid_curve),
        "ask_curve_diagnostic": _curve_payload(diagnostics.ask_curve),
        "A_dt_quantiles": diagnostics.A_dt_quantiles,
        "A_dt_fraction_above": diagnostics.A_dt_fraction_above,
        "candidate_only": {
            "dates": dates[:candidate_count],
            "sigma_train": candidate_result.sigma,
            "A_train": candidate_result.A,
            "kappa_train": candidate_result.kappa,
            "pooled_fit": asdict(candidate_result.diagnostics.pooled_fit),
            "A_dt_quantiles": candidate_result.diagnostics.A_dt_quantiles,
            "A_dt_fraction_above": candidate_result.diagnostics.A_dt_fraction_above,
        },
        "per_day": rows,
    }
    train_params_csv_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(train_params_csv_path(cfg), index=False)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Pooled calibration: sigma={result.sigma:.8f}, A={result.A:.6f}, "
        f"kappa={result.kappa:.2f}, proxy={calibration_config.execution_proxy}"
    )
    return payload


def formula_kwargs_from_config(
    train_params: dict[str, Any], cfg: ReplayExperimentConfig
) -> dict[str, float]:
    gamma_min, gamma_max = cfg.as_gamma_range
    return {
        "sigma": float(train_params["sigma_train"]),
        "gamma_min": gamma_min,
        "gamma_max": gamma_max,
        "skew_ticks_max": env_float("FORMULA_SKEW_TICKS_MAX", DEFAULT_FORMULA_SKEW_TICKS_MAX),
    }


def primary_b0_gamma_range(
    cfg: ReplayExperimentConfig, formula_kwargs: dict[str, float]
) -> tuple[float, float]:
    """Fail closed if the primary analytical and RL gamma bounds diverge."""

    rl_range = (float(formula_kwargs["gamma_min"]), float(formula_kwargs["gamma_max"]))
    if cfg.as_gamma_range != rl_range:
        raise RuntimeError(
            "Primary B0 gamma bounds must equal the B1/B2/B3 formula-action bounds: "
            f"B0={cfg.as_gamma_range}, RL={rl_range}"
        )
    return rl_range


def require_gamma_within_bounds(
    gamma: float, bounds: tuple[float, float]
) -> float:
    """Reject any tuner result outside the declared primary comparison range."""

    value = float(gamma)
    if not math.isfinite(value) or not bounds[0] <= value <= bounds[1]:
        raise RuntimeError(f"Tuned B0 gamma {value} is outside configured bounds {bounds}")
    return value


def candidate_config_and_formula(ctx: FormulaPipelineContext):
    """Return B2-candidate settings calibrated without validation exposure."""

    candidate = ctx.train_params["candidate_only"]
    cfg = replace(
        ctx.cfg, A=float(candidate["A_train"]), kappa=float(candidate["kappa_train"])
    )
    formula_kwargs = dict(ctx.formula_kwargs)
    formula_kwargs["sigma"] = float(candidate["sigma_train"])
    return cfg, formula_kwargs


def phase_training_inputs(ctx: FormulaPipelineContext, phase: str):
    """Training days and calibration of a phase: candidate = Jan 1-4 only (F-02, D-11)."""

    if phase == "candidate":
        cfg, formula_kwargs = candidate_config_and_formula(ctx)
        return (*load_candidate_train_arrays(ctx), cfg, formula_kwargs)
    if phase == "final":
        return (*load_train_arrays(ctx), ctx.cfg, ctx.formula_kwargs)
    raise ValueError(f"Unknown training phase: {phase!r}")


B1_ARTIFACT_STEMS = {
    "candidate": ("ppo_b1_cand_doge", "vecnorm_b1_cand"),
    "final": ("ppo_b1_doge", "vecnorm_b1"),
}


def b1_artifact_stems(phase: str, seed: int) -> tuple[str, str]:
    """B1 model and VecNormalize stems: B1_cand for candidates, final B1 otherwise (D-11)."""

    model_stem, vecnorm_stem = B1_ARTIFACT_STEMS[phase]
    return seeded_stem(model_stem, seed), seeded_stem(vecnorm_stem, seed)


# The final models evaluated on Jan 7-29; b1plus is the B1+ control (D-18).
FINAL_MODEL_STEMS = {
    "b1": B1_ARTIFACT_STEMS["final"],
    "b1plus": ("ppo_b1plus_doge", "vecnorm_b1plus"),
    "b2": ("ppo_b2_doge", "vecnorm_b2"),
    "b3": ("ppo_b3_doge", "vecnorm_b3"),
}


def final_model_stems(model: str, seed: int) -> tuple[str, str]:
    """Model and VecNormalize stems of a final model; a key outside FINAL_MODEL_STEMS raises."""

    if model not in FINAL_MODEL_STEMS:
        raise ValueError(f"No final model {model!r}; expected one of {sorted(FINAL_MODEL_STEMS)}")
    model_stem, vecnorm_stem = FINAL_MODEL_STEMS[model]
    return seeded_stem(model_stem, seed), seeded_stem(vecnorm_stem, seed)


# D-20: the files of one B3 run, by the metadata payload key that names each (write_metadata hashes them). The
# final B3 is the eta-profiling run D-12a selects, copied to the final names by profile_b3_dual.py --select.
B3_RUN_FILES = ("model_path", "vecnorm_path", "history_path", "training_costs_path", "training_feasibility_path")


def b3_run_paths(cfg: ReplayExperimentConfig, seed: int) -> dict[str, Path]:
    """The final B3's files (D-20): its model and frozen normalizer, per-update history, training costs and
    training feasibility."""

    model_stem, vecnorm_stem = final_model_stems("b3", seed)
    return {
        "model_path": cfg.model_path(f"{model_stem}.zip"),
        "vecnorm_path": cfg.vecnorm_path(vecnorm_stem),
        "history_path": cfg.result_path(seeded_result_name("b3_cvar_lagrangian_history.csv", seed)),
        "training_costs_path": cfg.result_path(seeded_result_name("b3_training_costs.csv", seed)),
        "training_feasibility_path": cfg.result_path(seeded_result_name("b3_training_feasibility.json", seed)),
    }


def eta_run_paths(cfg: ReplayExperimentConfig, eta: float) -> dict[str, Path]:
    """The same files of one eta-profiling run (D-20; training seed 42 only, D-12a), beside its D-12a history."""

    directory = cfg.results_dir / "profiling" / "b3_dual"
    return {
        "model_path": cfg.model_path(f"ppo_b3_eta{eta}_doge.zip"),
        "vecnorm_path": cfg.vecnorm_path(f"vecnorm_b3_eta{eta}"),
        "history_path": directory / f"b3_eta{eta}_cvar_lagrangian_history.csv",
        "training_costs_path": directory / f"b3_eta{eta}_training_costs.csv",
        "training_feasibility_path": directory / f"b3_eta{eta}_training_feasibility.json",
    }


def b3_inputs(cfg: ReplayExperimentConfig, seed: int) -> dict[str, Path]:
    """The files a B3 run loads: the final B1 and its normalizer, the D-17 d pair, the calibration and the
    manifest. A B3 run hashes them when it starts; record_profile_stage.py checks those hashes (D-19, D-20)."""

    b1_model_stem, b1_vecnorm_stem = b1_artifact_stems("final", seed)
    return {
        "warm_start_model": cfg.model_path(f"{b1_model_stem}.zip"),
        "warm_start_vecnorm": cfg.vecnorm_path(b1_vecnorm_stem),
        "threshold_d": cfg.result_path(seeded_result_name("cvar_threshold_d.txt", seed)),
        "threshold_json": cfg.result_path(seeded_result_name("cvar_threshold_b3.json", seed)),
        "train_params": train_params_json_path(cfg),
        "manifest": manifest_path(cfg),
    }


def eta_task_job_name(eta: float) -> str:
    """The job name of one eta-profiling run's metadata (D-20), by which record_profile_stage.py finds it."""

    return f"profile_b3_dual_eta{eta}"


# D-22: each alpha's final B2 trains on Jan 1-6 in parallel with the candidate sweep, under alpha-specific names;
# b2_promote (train_formula_b2.py --promote) copies the selected alpha's to the final B2's names, as D-20 does for B3.
B2_RUN_FILES = ("model_path", "vecnorm_path")


def b2_final_run_stems(alpha: float, seed: int) -> tuple[str, str]:
    """Model and VecNormalize stems of one parallel final B2 (D-22)."""

    return seeded_stem(f"ppo_b2_final_alpha{alpha}_doge", seed), seeded_stem(f"vecnorm_b2_final_alpha{alpha}", seed)


def b2_final_run_paths(cfg: ReplayExperimentConfig, alpha: float, seed: int) -> dict[str, Path]:
    """One parallel final B2's model and frozen normalizer (D-22)."""

    model_stem, vecnorm_stem = b2_final_run_stems(alpha, seed)
    return {"model_path": cfg.model_path(f"{model_stem}.zip"), "vecnorm_path": cfg.vecnorm_path(vecnorm_stem)}


def b2_final_paths(cfg: ReplayExperimentConfig, seed: int) -> dict[str, Path]:
    """The final B2's names, which b2_promote copies the selected run to (D-22)."""

    model_stem, vecnorm_stem = final_model_stems("b2", seed)
    return {"model_path": cfg.model_path(f"{model_stem}.zip"), "vecnorm_path": cfg.vecnorm_path(vecnorm_stem)}


def b2_selection_paths(cfg: ReplayExperimentConfig, seed: int) -> dict[str, Path]:
    """The B2 selection (F-05): the selected alpha and the sweep summary it was chosen from."""

    return {
        "b2_best_alpha": cfg.result_path(seeded_result_name("b2_best_alpha.txt", seed)),
        "b2_alpha_sweep_summary": cfg.result_path(seeded_result_name("b2_alpha_sweep_summary.csv", seed)),
    }


def b2_promotion_path(cfg: ReplayExperimentConfig, seed: int) -> Path:
    """What b2_promote records (D-22): alpha, source and target paths, SHA-256 of each file, the selection's."""

    return cfg.result_path(seeded_result_name("b2_promotion.json", seed))


def final_b2_inputs(cfg: ReplayExperimentConfig, seed: int) -> dict[str, Path]:
    """The files a final B2 or B1+ loads: the final B1 and its normalizer, the calibration and the manifest. They
    hash them when they start (provenance_at_start); b2_promote and record_profile_stage.py check those hashes."""

    b1_model_stem, b1_vecnorm_stem = b1_artifact_stems("final", seed)
    return {
        "warm_start_model": cfg.model_path(f"{b1_model_stem}.zip"),
        "warm_start_vecnorm": cfg.vecnorm_path(b1_vecnorm_stem),
        "train_params": train_params_json_path(cfg),
        "manifest": manifest_path(cfg),
    }


def final_b2_training_record(
    cfg: ReplayExperimentConfig, alpha: float, seed: int, hashes: dict[str, str], *, since: str = ""
) -> tuple[str, dict]:
    """The newest job metadata of the parallel final B2 of ``alpha`` (D-22) that saved files with these SHA-256
    under its alpha-specific names, written at or after ``since`` (a UTC stamp); raises if there is none."""

    names = {key: path.name for key, path in b2_final_run_paths(cfg, alpha, seed).items()}
    matches = []
    for path in sorted((cfg.results_dir / "job_metadata").glob("train_formula_b2_alpha*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        saved = record.get("artifact_sha256", {})
        if (
            (record.get("model"), record.get("phase"), record.get("alpha")) == ("b2", "final", alpha)
            and str(record.get("timestamp_utc", "")) >= since
            and all(
                Path(str(record.get(key, ""))).name == name and saved.get(f"payload.{key}") == hashes[key]
                for key, name in names.items()
            )
        ):
            matches.append((str(record.get("timestamp_utc", "")), path.name, record))
    if not matches:
        raise RuntimeError(
            f"No job metadata of a final B2 with alpha = {alpha} saved these files under {sorted(names.values())}, so "
            "nothing shows what they were trained from (D-22): rerun that alpha's final B2"
        )
    _, name, record = max(matches, key=lambda match: match[:2])
    return name, record


def archive_previous_promotion(cfg: ReplayExperimentConfig, seed: int) -> Path | None:
    """Move the last promotion (the final B2's files and b2_promotion.json) to results/b2_promotion_superseded/<UTC>/,
    so that a refused promotion leaves no final B2 behind (D-22)."""

    present = [path for path in (*b2_final_paths(cfg, seed).values(), b2_promotion_path(cfg, seed)) if path.exists()]
    if not present:
        return None
    archive = cfg.results_dir / "b2_promotion_superseded" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    archive.mkdir(parents=True)  # a new folder; an existing one is an error, never merged into
    for path in present:
        path.replace(archive / path.name)
    print(f"Moved the previous final B2 and its promotion {[path.name for path in present]} to {archive}")
    return archive


def promote_b2(cfg: ReplayExperimentConfig, seed: int) -> dict:
    """D-22: copy the selected alpha's parallel final B2 (model and frozen normalizer) to the final B2's names, byte
    for byte, and record it in b2_promotion.json.

    It first moves the previous promotion aside, then refuses, promoting nothing, unless the B2 selection exists,
    the selected alpha's run exists, and that run's job metadata hashed, when it loaded them, the current B1, its
    normalizer, the calibration and the manifest. The JSON is written last.
    """

    import shutil

    from cvar_mm.gym.model_selection import require_b2_alpha

    archive_previous_promotion(cfg, seed)
    selection = b2_selection_paths(cfg, seed)
    missing = [str(path) for path in selection.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"The B2 selection is missing, so no final B2 is promoted (D-22): {missing}")
    alpha = float(selection["b2_best_alpha"].read_text(encoding="utf-8"))
    require_b2_alpha(alpha)
    sources, targets = b2_final_run_paths(cfg, alpha, seed), b2_final_paths(cfg, seed)
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"The selected alpha = {alpha} final B2 run is missing, so none is promoted (D-22): {missing}")
    hashes = {key: file_sha256(path) for key, path in sources.items()}
    name, record = final_b2_training_record(cfg, alpha, seed, hashes)
    loaded = (record.get("provenance_at_start") or {}).get("input_sha256") or {}
    stale = [key for key, path in final_b2_inputs(cfg, seed).items() if loaded.get(key) != file_sha256(path)]
    if stale:
        raise RuntimeError(
            f"The selected alpha = {alpha} final B2 ({name}) did not load the current {', '.join(stale)}, so it is "
            "not promoted (D-22): rerun it, then b2_promote"
        )

    def relative(path: Path) -> str:
        return path.relative_to(cfg.repo_root).as_posix()

    files = {}
    for key in B2_RUN_FILES:
        shutil.copyfile(sources[key], targets[key])
        digest = file_sha256(targets[key])
        if digest != hashes[key]:
            raise RuntimeError(f"The copy of {sources[key]} differs from it (D-22)")
        files[key] = {"from": relative(sources[key]), "to": relative(targets[key]), "sha256": digest}
    promotion = {
        "decision": "D-22",
        "alpha": alpha,
        "files": files,
        "selection_sha256": {relative(path): file_sha256(path) for path in selection.values()},
        "source_metadata": name,
        "source_slurm_job_id": record.get("slurm_job_id"),
        "source_inputs_sha256": loaded,
        "git_commit": get_git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }
    b2_promotion_path(cfg, seed).write_text(json.dumps(promotion, indent=2, allow_nan=False), encoding="utf-8")
    return promotion


def warm_start_from(model, source_model_path: Path, source_vecnorm) -> float:
    """D-11, D-15a, D-15b, D-15c: start a B2/B3 PPO from the source B1, in B1's units.

    - Copies B1's policy and critic weights only; the new PPO keeps its freshly
      built Adam moments (D-15c) and its own learning rate, FINE_TUNE_LR (D-23),
      not B1's: B1's optimizer and learning rate stay on the loaded copy.
    - The critic stays in B1's units: B2 and B3 train on B1's normalized rewards
      r / s with B1's frozen normalizer (D-15a), so the value head is not rescaled.
    - Freezes log sigma at B1's value (D-15b). ``requires_grad`` is not part of
      a state dict, so it is set after the load.
    - Refuses a source whose sigma is not POLICY_STD (D-21): B1 explores with
      sigma0 fixed, which B2, B3, the eta profiler and B1+ inherit here.
    It is identical for B2 (all phases), B3, the eta profiler and B1+. Returns
    s = sqrt(ret_rms.var + eps), the divisor of B1's normalized rewards.
    """

    import torch
    from stable_baselines3 import PPO

    if source_vecnorm.ret_rms.count <= 1:
        raise ValueError("The source normalizer never updated its return statistics: not a trained B1")
    if not source_vecnorm.norm_reward or source_vecnorm.training:
        raise ValueError(
            "B2/B3 train on B1's normalizer frozen with norm_reward=True and training=False (D-15a)"
        )
    source_policy = PPO.load(str(source_model_path), device="cpu").get_parameters()["policy"]
    if "log_std" not in source_policy:
        raise ValueError(f"The source B1 {source_model_path} has no log sigma (D-21)")
    require_policy_std(source_policy["log_std"], source=f"The source B1 {source_model_path}")
    model.set_parameters({"policy": source_policy}, exact_match=False)
    # exact_match=False loads the module with strict=False, so check that every weight arrived.
    loaded = model.policy.state_dict()
    if loaded.keys() != source_policy.keys() or not all(
        torch.equal(loaded[key], value) for key, value in source_policy.items()
    ):
        raise ValueError(f"The warm-started policy differs from the source B1 {source_model_path} (D-15c)")
    model.policy.log_std.requires_grad_(False)
    return vecnormalize_reward_scale(source_vecnorm)


def load_context(*, force_prepare: bool = False, require_prepared: bool = False) -> FormulaPipelineContext:
    require_snellius_paths()
    cfg = build_config()
    if require_prepared:
        # Preparing the split or calibrating here would read Jan 5-6 (candidate jobs never may).
        missing = [str(path) for path in (manifest_path(cfg), train_params_json_path(cfg)) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Run prepare_formula_pipeline.py first; missing {missing}")
    cfg.ensure_artifact_dirs()
    if force_prepare:
        manifest = prepare_manifest(cfg)
    else:
        manifest = load_or_prepare_manifest(cfg)
    train_params = load_or_create_train_params(cfg, manifest, force=force_prepare)
    cfg = replace(cfg, A=float(train_params["A_train"]), kappa=float(train_params["kappa_train"]))
    return FormulaPipelineContext(
        project_dir=PROJECT_DIR,
        data_dir=cfg.datasets_dir,
        cfg=cfg,
        manifest=manifest,
        train_params=train_params,
        formula_kwargs=formula_kwargs_from_config(train_params, cfg),
    )


def model_key(model: str, alpha: float | None = None) -> str:
    if model == "b2":
        return "b2" if alpha is None else f"b2_alpha{alpha}"
    return model


def eval_part_dir(
    cfg: ReplayExperimentConfig,
    model: str,
    alpha: float | None = None,
    *,
    split: str = "retrospective_holdout",
    training_seed: int = DEFAULT_TRAINING_SEED,
) -> Path:
    key = seeded_stem(model_key(model, alpha), training_seed)
    return cfg.results_dir / "eval_parts" / split / key


def eval_part_path(
    cfg: ReplayExperimentConfig,
    *,
    model: str,
    date: str,
    alpha: float | None = None,
    split: str = "retrospective_holdout",
    training_seed: int = DEFAULT_TRAINING_SEED,
) -> Path:
    return eval_part_dir(
        cfg, model, alpha, split=split, training_seed=training_seed
    ) / f"day_{date}.csv"


def eval_episode_path(
    cfg: ReplayExperimentConfig,
    *,
    model: str,
    date: str,
    alpha: float | None = None,
    split: str = "retrospective_holdout",
    training_seed: int = DEFAULT_TRAINING_SEED,
) -> Path:
    """One row per full-day path (fill seed x rollout) of one evaluation day (K6)."""

    return eval_part_dir(
        cfg, model, alpha, split=split, training_seed=training_seed
    ) / f"episodes_day_{date}.csv"


def b2_alpha_result_path(
    cfg: ReplayExperimentConfig, alpha: float, training_seed: int = DEFAULT_TRAINING_SEED
) -> Path:
    return cfg.result_path(seeded_result_name(
        f"b2_alpha{alpha}_validation_results.csv", training_seed
    ))


def get_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_DIR,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _git_text(*args: str) -> str | None:
    """git's output without its trailing newline. Never strip the start: a `status --porcelain` line begins
    with its two status columns, and " M path" (modified, not staged) would lose its leading space and then
    parse as another path (record_profile_stage.uncommitted_training_code reads `line[3:]`)."""

    try:
        return subprocess.check_output(
            ["git", *args], cwd=PROJECT_DIR, text=True, stderr=subprocess.DEVNULL
        ).rstrip("\n")
    except Exception:
        return None


# The training code that B2, B1+ and B3 share (git pathspecs, D-19): the profile stage runs one commit of it,
# and the final stage refuses another. submit_final_pipeline.sh checks the same list.
TRAINING_CODE_PATHS = (
    "cvar_mm",
    ":(exclude)cvar_mm/gym/statistical_inference.py",
    "scripts/pipeline/common.py",
    "scripts/pipeline/train_formula_b2.py",
    "scripts/pipeline/train_formula_b3.py",
)


def training_code_state() -> dict[str, Any]:
    """This checkout's commit and its uncommitted or untracked training-code changes, now (D-19 provenance).

    write_metadata records the commit when a job ends; a job calls this when it starts. A None commit means
    git could not tell, and record_profile_stage.py then refuses the job.
    """

    return {
        "git_commit": get_git_commit(),
        "uncommitted_training_code": _git_text(
            "status", "--porcelain", "--untracked-files=all", "--", *TRAINING_CODE_PATHS
        ),
    }


def _existing_file_hashes(value: Any, prefix: str = "payload") -> dict[str, str]:
    hashes: dict[str, str] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            hashes.update(_existing_file_hashes(child, f"{prefix}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            hashes.update(_existing_file_hashes(child, f"{prefix}[{index}]"))
    elif isinstance(value, (str, Path)):
        path = Path(value)
        if path.is_file():
            hashes[prefix] = file_sha256(path)
    return hashes


def write_metadata(job_name: str, payload: dict[str, Any]) -> Path:
    """The job's metadata JSON: environment, git state at the end of the job, the payload, and the SHA-256 of
    every existing file the payload names.

    A job that simulates passes the configuration it used as payload["experiment_config"], so the record holds
    the calibrated A and kappa it ran with (Jan 1-6, or Jan 1-4 for candidate jobs; WORK_ORDER item 8). Without
    one, A and kappa are recorded as None: the ReplayExperimentConfig defaults are not a calibration any job uses.
    """

    cfg = build_config()
    meta_dir = cfg.results_dir / "job_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = meta_dir / f"{job_name}_{job_id}_{timestamp}.json"
    base = {
        "job_name": job_name,
        "slurm_job_id": job_id,
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "timestamp_utc": timestamp,
        "hostname": socket.gethostname(),
        "project_dir": str(PROJECT_DIR),
        "data_dir": os.environ.get("DATA_DIR"),
        # Retained for backward compatibility with historical manifests.
        "conda_env": os.environ.get("CONDA_ENV"),
        # Corrected runs use the isolated thesis virtual environment.
        "venv_path": os.environ.get("VENV_PATH") or os.environ.get("VIRTUAL_ENV"),
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "python_executable": sys.executable,
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
        "python_path": os.environ.get("PYTHONPATH"),
        "git_commit": get_git_commit(),
        "git_branch": _git_text("branch", "--show-current"),
        "git_status_porcelain": _git_text("status", "--porcelain"),
        "python_version": platform.python_version(),
        "command": sys.argv,
        "experiment_config": {**asdict(cfg), "A": None, "kappa": None},
        "policy_observation": list(FORMULA_FEATURE_NAMES),
        "policy_action": ["gamma", "quote_skew_price_units"],
        "execution_model": "p=1-exp(-A*exp(-kappa*delta)*dt)",
        "retrospective_holdout": "January 7-29; evaluation only",
    }
    base.update(payload)
    base["artifact_sha256"] = _existing_file_hashes(payload)
    path.write_text(json.dumps(_json_safe(base), indent=2), encoding="utf-8")
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    return value


def archive_once(path: Path, archive_path: Path) -> None:
    if path.exists() and not archive_path.exists():
        import shutil

        archive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, archive_path)


def print_context_banner(ctx: FormulaPipelineContext, job_name: str) -> None:
    print(f"=== {job_name} ===")
    print(f"Project dir : {ctx.project_dir}")
    print(f"Data dir    : {ctx.data_dir}")
    print(f"Results dir : {ctx.cfg.results_dir}")
    print(f"Train days  : {len(ctx.train_manifest)}")
    print(f"Test days   : {len(ctx.test_manifest)}")
    print(f"Later holdout days (untouched): {len(ctx.later_holdout_manifest)}")
    print(f"Formula     : {ctx.formula_kwargs}")
    print(f"Features    : {list(FORMULA_FEATURE_NAMES)}")

