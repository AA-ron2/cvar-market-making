"""The shadow replay of the forward test (PROTOCOL §9, decision D3) and its parity check against the pipeline.

    python -m hummingbot_adapter.shadow_replay run --model {b0,b3} --day-file PATH --date YYYY-MM-DD --params PATH
        --out-dir DIR [--gamma-file PATH] [--model-zip PATH --vecnorm PATH] [--fill-seeds "42 43"] [--rollouts 20]
    python -m hummingbot_adapter.shadow_replay compare --ours EPISODES.csv --reference EPISODES.csv --day YYYY-MM-DD

run: the frozen simulator on one day file in the Tardis book_snapshot_25 layout (.csv or .csv.gz: a bot's own
snapshots file, or the Jan 7 2025 file for the parity check), configured as the pipeline: export
configs/final_pipeline.env first (source it). --params is hb_calibration.json (sigma, A, kappa) or
formula_train_params.json (sigma_train, A_train, kappa_train).
- B0 as scripts/pipeline/run_b0_fair.py: cvar_mm.gym.data_loader.load_single_day, the frozen gamma_B0 of --gamma-file,
  cvar_mm.gym.helpers.fast_rollout.fast_simulate per fill seed, the rows of
  cvar_mm.gym.notebook_support.stats_dict_to_frame.
- B3 as scripts/pipeline/evaluate_formula_model.py: load_single_day_with_features, PPO.load(device="cpu") and
  cvar_mm.gym.notebook_support.evaluate_formula_rl_per_day per fill seed (return_episodes=True).
It writes <out-dir>/shadow_<model>_<date>_episodes.csv (one row per fill seed x rollout, with the columns and the
column order of the pipeline's episode files) and shadow_<model>_<date>.json (provenance), then rebuilds
<out-dir>/SHA256SUMS over the folder's files. Neither output is ever overwritten; the out-dir may not lie inside the
repository or the day file's folder, and may hold nothing but shadow-replay outputs and their SHA256SUMS. Every
refusal (a missing input file included) runs before the day file is read.

compare: one day's rows of two episode files (ours and the pipeline's), matched on (Fill Seed, Rollout), every numeric
column compared.

Exit codes: 0 done (compare: every difference within 1e-9 relative or 1e-12 absolute); 1 compare found a larger
difference or a layout mismatch; 2 a refusal.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import re
import socket
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import date as Date
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np  # numpy and pandas before torch (SB3, scripts.pipeline.common): one OpenMP runtime on Windows
import pandas as pd

from hummingbot_adapter.bot_config import B0_GAMMA_SHA256, MODEL_SHA256, VECNORM_SHA256, protocol_default
from hummingbot_adapter.provenance import REPO_ROOT, file_sha256, git_output, is_inside
from cvar_mm.gym.data_loader import load_single_day, load_single_day_with_features
from cvar_mm.gym.experiment_config import ReplayExperimentConfig
from cvar_mm.gym.helpers.fast_rollout import fast_simulate
from cvar_mm.gym.notebook_support import evaluate_formula_rl_per_day, stats_dict_to_frame
from scripts.pipeline.common import (
    build_config,
    eval_episode_path,
    formula_kwargs_from_config,
    parse_seed_list,
    seeded_result_name,
)

MODELS = ("b0", "b3")
B0_GAMMA = protocol_default("gamma_b0")  # PROTOCOL §2: 0.8908584807525282
PARAM_KEYS = {
    "hb_calibration": ("sigma", "A", "kappa"),
    "formula_train_params": ("sigma_train", "A_train", "kappa_train"),
}
# configs/final_pipeline.env (F-04, F-07): the evaluation settings build_config and formula_kwargs_from_config read
PIPELINE_ENV = {"FORMULA_SKEW_TICKS_MAX": 5.0, "EVALUATION_FILL_SEEDS": [42, 43], "EVALUATION_ROLLOUTS": 20}
RECORDED_ENV = ("FORMULA_GAMMA_MIN", "FORMULA_GAMMA_MAX", "FORMULA_SKEW_TICKS_MAX", "FEATURE_WINDOW",
                "EVALUATION_FILL_SEEDS", "EVALUATION_ROLLOUTS", "FIXED_CLOCK_SECONDS", "TRAINING_SEED")
Q_MAX = 10
LEADING = ["Day", "Fill Seed", "Rollout"]  # first, as run_b0_fair.py and evaluate_formula_model.py write them
KEYS = ["Fill Seed", "Rollout"]
# The header of the pipeline's episode files (the author's check of b0_test_episodes.csv on Snellius, 2026-10-23)
EPISODE_COLUMNS = (
    "Day", "Fill Seed", "Rollout", "Sharpe", "Sortino", "Max DD", "P&L-to-MAP", "Final PnL", "Mean |q|", "Terminal q",
    "Mean spread", "Sharpe Snapshot (supplementary)", "Sortino Snapshot (supplementary)", "Near Cap Fraction",
)
RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE = 1e-9, 1e-12
SUMS_NAME = "SHA256SUMS"
SHADOW_OUTPUT = re.compile(r"shadow_(b0|b3)_\d{4}-\d{2}-\d{2}(_episodes\.csv|\.json)")
VERSION_PACKAGES = ("numpy", "pandas", "torch", "stable-baselines3", "gymnasium")


class ShadowRefused(RuntimeError):
    """An input or a setting fails a rule of the pre-registration (PROTOCOL §2) or of WO-HB3."""


# --- inputs ----------------------------------------------------------------------------------------------------


def parse_date(text: str) -> str:
    try:
        return Date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ShadowRefused(f"--date must be YYYY-MM-DD, got {text!r}") from exc


def read_params(path: str | Path) -> dict[str, Any]:
    """sigma, A and kappa from hb_calibration.json (sigma, A, kappa) or formula_train_params.json (sigma_train,
    A_train, kappa_train), with the file's SHA-256. The top-level object must hold exactly one of the two key sets
    and no key of the other; anything else is refused."""

    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ShadowRefused(f"{path}: not a readable JSON file ({exc})") from exc
    if not isinstance(data, dict):
        raise ShadowRefused(f"{path}: expected a JSON object, got {type(data).__name__}")
    complete = [kind for kind, keys in PARAM_KEYS.items() if all(key in data for key in keys)]
    touched = [kind for kind, keys in PARAM_KEYS.items() if any(key in data for key in keys)]
    if len(complete) != 1 or touched != complete:
        raise ShadowRefused(
            f"{path}: expected the keys {list(PARAM_KEYS['hb_calibration'])} (hb_calibration.json) or "
            f"{list(PARAM_KEYS['formula_train_params'])} (formula_train_params.json); found {sorted(data)[:12]}"
        )
    kind = complete[0]
    values = {}
    for name, key in zip(("sigma", "A", "kappa"), PARAM_KEYS[kind]):
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ShadowRefused(f"{path}: {key} must be a finite positive number, got {value!r}")
        values[name] = float(value)
    return {"path": str(path), "kind": kind, "keys": list(PARAM_KEYS[kind]), "sha256": file_sha256(path), **values}


def read_gamma(path: str | Path) -> dict[str, Any]:
    """gamma_B0 from --gamma-file: refused unless the file is the frozen b0_gamma_fixed.txt (its SHA-256) and its value
    is exactly the frozen 0.8908584807525282 (PROTOCOL §2)."""

    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    try:
        gamma = float(text)
    except ValueError as exc:
        raise ShadowRefused(f"{path}: expected one number, got {text[:40]!r}") from exc
    if gamma != B0_GAMMA:
        raise ShadowRefused(f"{path}: gamma {gamma!r} is not the frozen gamma_B0 {B0_GAMMA!r} (PROTOCOL §2)")
    digest = file_sha256(path)
    if digest != B0_GAMMA_SHA256:
        raise ShadowRefused(f"{path} has SHA-256 {digest}, not the frozen {B0_GAMMA_SHA256} (PROTOCOL §2)")
    return {"path": str(path), "sha256": digest, "gamma": gamma}


def require_sha256(path: str | Path, expected: str, what: str) -> dict[str, str]:
    digest = file_sha256(path)
    if digest != expected:
        raise ShadowRefused(f"{path} has SHA-256 {digest}, not the frozen {what} {expected} (PROTOCOL §2)")
    return {"path": str(path), "sha256": digest}


def pipeline_config(params: dict[str, Any], rollouts: int) -> tuple[ReplayExperimentConfig, dict[str, float]]:
    """The pipeline's configuration (load_context): build_config() from the exported configs/final_pipeline.env,
    refused unless FORMULA_SKEW_TICKS_MAX, EVALUATION_FILL_SEEDS and EVALUATION_ROLLOUTS are the pipeline's; A and
    kappa as load_context sets them, sigma through formula_kwargs_from_config. The rollout count is the run's
    (--rollouts; the pipeline's 20 unless a test asks for fewer)."""

    parsers = {"FORMULA_SKEW_TICKS_MAX": float, "EVALUATION_FILL_SEEDS": parse_seed_list, "EVALUATION_ROLLOUTS": int}
    for name, expected in PIPELINE_ENV.items():
        raw = os.environ.get(name)
        try:
            value = None if raw is None or not raw.strip() else parsers[name](raw)
        except ValueError:
            value = None
        if value != expected:
            raise ShadowRefused(f"{name}={raw!r}, the pipeline sets {expected!r}: "
                                "source configs/final_pipeline.env first")
    cfg = replace(build_config(), A=params["A"], kappa=params["kappa"])
    formula_kwargs = formula_kwargs_from_config({"sigma_train": params["sigma"]}, cfg)
    if cfg.q_max != Q_MAX:
        raise ShadowRefused(f"cfg.q_max is {cfg.q_max}, the pipeline's Q_MAX is {Q_MAX}")
    return replace(cfg, evaluation_rollouts=rollouts), formula_kwargs


# --- the replays -----------------------------------------------------------------------------------------------


def order_episodes(episodes: pd.DataFrame) -> pd.DataFrame:
    """Day, Fill Seed and Rollout first, the rest in their order, as the pipeline writes its episode files."""

    return episodes[LEADING + [column for column in episodes.columns if column not in LEADING]]


def replay_b0(day_file: Path, date: str, cfg: ReplayExperimentConfig, sigma: float, gamma: float,
              fill_seeds: list[int]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """run_b0_fair.py's test-day loop on one day file: fast_simulate per fill seed, one row per rollout."""

    S, dt, _ = load_single_day(str(day_file))
    frames = []
    for fill_seed in fill_seeds:
        stats = fast_simulate(
            midprices=S, dt_array=dt, gamma=gamma, sigma=sigma,
            kappa=cfg.kappa, A=cfg.A, terminal_time=float(dt.sum()),
            tick_size=cfg.tick_size, Q_MAX=cfg.q_max,
            num_trajectories=cfg.evaluation_rollouts, seed=fill_seed,
            use_linear_approximation=False,
            fixed_clock_seconds=cfg.fixed_clock_seconds,
        )
        frame = stats_dict_to_frame(stats)
        frames.append(frame.assign(Day=str(date), Rollout=np.arange(len(frame)), **{"Fill Seed": fill_seed}))
    episodes = order_episodes(pd.concat(frames, ignore_index=True))
    return episodes, {"rows": int(len(S)), "terminal_time": float(dt.sum())}


def replay_b3(day_file: Path, date: str, cfg: ReplayExperimentConfig, formula_kwargs: dict[str, float],
              model_zip: Path, vecnorm: Path, fill_seeds: list[int]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """evaluate_formula_model.py on one day file: evaluate_formula_rl_per_day per fill seed with return_episodes."""

    from stable_baselines3 import PPO  # here, not at import: B0 and compare need no torch model

    S, dt, _, features = load_single_day_with_features(str(day_file), tick_size=cfg.tick_size)
    model = PPO.load(str(model_zip), device="cpu")
    frames = []
    for fill_seed in fill_seeds:
        _, episodes = evaluate_formula_rl_per_day(
            model=model,
            vecnorm_path=vecnorm,
            test_S=[S],
            test_dt=[dt],
            test_dates=[date],
            test_market_features=[features],
            config=cfg,
            seed=fill_seed,
            num_rollouts=cfg.evaluation_rollouts,
            return_episodes=True,
            **formula_kwargs,
        )
        frames.append(episodes.assign(**{"Fill Seed": fill_seed}))
    episodes = order_episodes(pd.concat(frames, ignore_index=True))
    return episodes, {"rows": int(len(S)), "terminal_time": float(dt.sum())}


# --- outputs ---------------------------------------------------------------------------------------------------


def output_paths(out_dir: Path, model: str, date: str) -> tuple[Path, Path]:
    return out_dir / f"shadow_{model}_{date}_episodes.csv", out_dir / f"shadow_{model}_{date}.json"


def rebuild_sums(folder: Path) -> list[str]:
    """SHA256SUMS over every file in the folder (sha256sum format, LF), replaced atomically."""

    names = sorted(path.name for path in folder.iterdir()
                   if path.is_file() and path.name != SUMS_NAME and not path.name.endswith(".tmp"))
    lines = [f"{file_sha256(folder / name)}  {name}" for name in names]
    temporary = folder / (SUMS_NAME + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("".join(line + "\n" for line in lines))
    os.replace(temporary, folder / SUMS_NAME)
    return lines


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in VERSION_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def git_state() -> dict[str, Any]:
    head = git_output(REPO_ROOT, "rev-parse", "HEAD")
    status = git_output(REPO_ROOT, "status", "--porcelain", "--untracked-files=all", "--",
                        "cvar_mm", "scripts/pipeline/common.py", "hummingbot_adapter")
    return {
        "git_commit": head.stdout.strip() if head.returncode == 0 else None,
        "git_status_code_paths": status.stdout.rstrip("\n") if status.returncode == 0 else None,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def run(args: argparse.Namespace) -> int:
    started_utc, started = _utc_now(), time.perf_counter()
    model, date = args.model, parse_date(args.date)
    day_file, out_dir = Path(args.day_file), Path(args.out_dir)
    if not day_file.is_file():
        raise ShadowRefused(f"no such day file: {day_file}")
    if not (day_file.name.endswith(".csv") or day_file.name.endswith(".csv.gz")):
        raise ShadowRefused(f"{day_file.name}: expected a .csv or .csv.gz day file")
    if date not in day_file.name:
        raise ShadowRefused(f"{day_file.name} does not carry the date {date} in its name")
    if model == "b0" and (args.gamma_file is None or args.model_zip is not None or args.vecnorm is not None):
        raise ShadowRefused("--model b0 takes --gamma-file, not --model-zip or --vecnorm")
    if model == "b3" and (args.gamma_file is not None or args.model_zip is None or args.vecnorm is None):
        raise ShadowRefused("--model b3 takes --model-zip and --vecnorm, not --gamma-file")
    try:
        fill_seeds = parse_seed_list(args.fill_seeds)
    except ValueError as exc:
        raise ShadowRefused(f"--fill-seeds {args.fill_seeds!r}: {exc}") from exc
    if args.rollouts < 1:
        raise ShadowRefused(f"--rollouts must be positive, got {args.rollouts}")
    inputs_given = [args.params] + ([args.gamma_file] if model == "b0" else [args.model_zip, args.vecnorm])
    missing = [str(path) for path in inputs_given if not Path(path).is_file()]
    if missing:
        raise ShadowRefused(f"no such input file: {', '.join(missing)}")
    if is_inside(out_dir, REPO_ROOT) or is_inside(out_dir, day_file.resolve().parent):
        raise ShadowRefused(f"--out-dir {out_dir} lies inside the repository or the day file's folder")
    if out_dir.exists():  # shadow outputs only: never a data, bot or analysis folder (its SHA256SUMS is rebuilt)
        foreign = sorted(entry.name for entry in out_dir.iterdir()
                         if entry.name != SUMS_NAME and not (entry.is_file() and SHADOW_OUTPUT.fullmatch(entry.name)))
        if foreign:
            raise ShadowRefused(f"--out-dir {out_dir} holds other things than shadow-replay outputs: {foreign[:5]}")
    episodes_path, record_path = output_paths(out_dir, model, date)
    for path in (episodes_path, record_path):
        if path.exists():
            raise ShadowRefused(f"{path} exists; outputs are never overwritten")
    params = read_params(args.params)
    cfg, formula_kwargs = pipeline_config(params, args.rollouts)
    inputs: dict[str, Any] = {"params": params}
    if model == "b0":
        inputs["gamma_file"] = read_gamma(args.gamma_file)
    else:
        inputs["model_zip"] = require_sha256(args.model_zip, MODEL_SHA256, "B3 policy")
        inputs["vecnorm"] = require_sha256(args.vecnorm, VECNORM_SHA256, "B3 normalizer")
    inputs["day_file"] = {"path": str(day_file), "sha256": file_sha256(day_file)}

    print(f"shadow replay {model.upper()} {date}: {day_file.name} (SHA-256 {inputs['day_file']['sha256']}), "
          f"params {params['kind']} sigma {params['sigma']!r} A {params['A']!r} kappa {params['kappa']!r}, "
          f"fill seeds {fill_seeds} x {cfg.evaluation_rollouts} rollouts")
    if model == "b0":
        episodes, day = replay_b0(day_file, date, cfg, params["sigma"], inputs["gamma_file"]["gamma"], fill_seeds)
    else:
        episodes, day = replay_b3(day_file, date, cfg, formula_kwargs, Path(args.model_zip), Path(args.vecnorm),
                                  fill_seeds)
    if tuple(episodes.columns) != EPISODE_COLUMNS:
        raise RuntimeError(f"episode columns {list(episodes.columns)} differ from the pipeline's "
                           f"{list(EPISODE_COLUMNS)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(episodes_path, "x", encoding="utf-8", newline="") as handle:
        episodes.to_csv(handle, index=False, lineterminator="\n")
    record = {
        "tool": "hummingbot_adapter.shadow_replay run",
        "model": model.upper(),
        "date": date,
        "inputs": inputs,
        "fill_seeds": fill_seeds,
        "rollouts": cfg.evaluation_rollouts,
        "pipeline_settings": fill_seeds == PIPELINE_ENV["EVALUATION_FILL_SEEDS"]
        and cfg.evaluation_rollouts == PIPELINE_ENV["EVALUATION_ROLLOUTS"],
        "parameters": {
            "sigma": formula_kwargs["sigma"], "A": cfg.A, "kappa": cfg.kappa, "tick_size": cfg.tick_size,
            "Q_MAX": cfg.q_max, "fixed_clock_seconds": cfg.fixed_clock_seconds,
            "gamma_b0": inputs["gamma_file"]["gamma"] if model == "b0" else None,
            "formula_kwargs": formula_kwargs if model == "b3" else None,
            "terminal_time": day["terminal_time"] if model == "b0" else "the environment's (the day's total dt)",
        },
        "config": json.loads(json.dumps(asdict(cfg), default=str)),
        "environment": {name: os.environ.get(name) for name in RECORDED_ENV},
        "day_rows": day["rows"],
        "episodes": {"file": episodes_path.name, "rows": int(len(episodes)), "sha256": file_sha256(episodes_path)},
        **git_state(),
        "versions": package_versions(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "command": sys.argv,
        "start_utc": started_utc,
        "end_utc": _utc_now(),
        "wall_s": time.perf_counter() - started,
    }
    with open(record_path, "x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, indent=2) + "\n")
    rebuild_sums(out_dir)
    if not record["pipeline_settings"]:
        print(f"NOTE: fill seeds {fill_seeds} x {cfg.evaluation_rollouts} rollouts are not the pipeline's "
              f"{PIPELINE_ENV['EVALUATION_FILL_SEEDS']} x {PIPELINE_ENV['EVALUATION_ROLLOUTS']} (a test run)")
    print(f"wrote {episodes_path} ({len(episodes)} rows) and {record_path}; wall {record['wall_s']:.1f} s")
    return 0


# --- compare ---------------------------------------------------------------------------------------------------


@dataclass
class Comparison:
    ok: bool
    problems: list[str]
    summary: pd.DataFrame  # one row per metric: max_abs, max_rel, failing rows
    failures: pd.DataFrame  # one row per failing (key, metric)


def read_episodes(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"Day": str}, float_precision="round_trip")  # exact for repr-printed floats


def compare_frames(ours: pd.DataFrame, reference: pd.DataFrame, day: str) -> Comparison:
    """One day's rows of two episode files, matched on (Fill Seed, Rollout): the same columns in the same order and
    the same keys; every non-key column numeric in both and equal within 1e-9 relative or 1e-12 absolute (a NaN
    equals only a NaN)."""

    problems: list[str] = []
    frames = {}
    for name, frame in (("ours", ours), ("reference", reference)):
        missing = [column for column in LEADING if column not in frame.columns]
        if missing:
            problems.append(f"{name} lacks the columns {missing}")
            continue
        frames[name] = frame[frame["Day"] == day].reset_index(drop=True)
        if frames[name].empty:
            problems.append(f"{name} has no rows with Day == {day}")
        elif frames[name].duplicated(KEYS).any():
            problems.append(f"{name} repeats a (Fill Seed, Rollout) key on {day}")
    empty = pd.DataFrame()
    if problems:
        return Comparison(False, problems, empty, empty)
    a, b = frames["ours"], frames["reference"]
    if list(a.columns) != list(b.columns):
        problems.append(f"the columns differ: ours {list(a.columns)}, reference {list(b.columns)}")
    keys_a, keys_b = set(map(tuple, a[KEYS].to_numpy().tolist())), set(map(tuple, b[KEYS].to_numpy().tolist()))
    if keys_a != keys_b:
        problems.append(f"the keys differ: only ours {sorted(keys_a - keys_b)[:10]}, "
                        f"only the reference {sorted(keys_b - keys_a)[:10]}")
    metrics = [column for column in b.columns if column not in LEADING]
    non_numeric = [column for column in metrics if column in a.columns and not (
        pd.api.types.is_numeric_dtype(a[column]) and pd.api.types.is_numeric_dtype(b[column]))]
    if non_numeric:
        problems.append(f"non-numeric metric columns: {non_numeric}")
    if problems:
        return Comparison(False, problems, empty, empty)
    merged = a.merge(b, on=KEYS, suffixes=(" (ours)", " (reference)"), validate="one_to_one")
    rows, failures = [], []
    for metric in metrics:
        x = merged[f"{metric} (ours)"].to_numpy(dtype=np.float64)
        y = merged[f"{metric} (reference)"].to_numpy(dtype=np.float64)
        both_nan = np.isnan(x) & np.isnan(y)
        same = (x == y) | both_nan
        with np.errstate(invalid="ignore", divide="ignore"):
            absolute = np.where(same, 0.0, np.abs(x - y))
            relative = np.where(same, 0.0, absolute / np.abs(y))
        absolute = np.where(np.isnan(absolute), np.inf, absolute)
        relative = np.where(np.isnan(relative), np.inf, relative)
        good = same | (absolute <= ABSOLUTE_TOLERANCE) | (relative <= RELATIVE_TOLERANCE)
        rows.append({"metric": metric, "max_abs": float(absolute.max()), "max_rel": float(relative.max()),
                     "failing_rows": int((~good).sum())})
        for index in np.flatnonzero(~good):
            failures.append({"Fill Seed": merged["Fill Seed"].iloc[index], "Rollout": merged["Rollout"].iloc[index],
                             "metric": metric, "ours": x[index], "reference": y[index],
                             "abs": absolute[index], "rel": relative[index]})
    summary = pd.DataFrame(rows)
    return Comparison(not failures, [], summary, pd.DataFrame(failures))


def compare(args: argparse.Namespace) -> int:
    day = parse_date(args.day)
    frames = []
    for path in (args.ours, args.reference):
        if not Path(path).is_file():
            raise ShadowRefused(f"no such episode file: {path}")
        try:
            frames.append(read_episodes(path))
        except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            raise ShadowRefused(f"{path} is not a readable episode CSV ({exc})") from exc
    result = compare_frames(frames[0], frames[1], day)
    print(f"compare {day}: ours {args.ours} (SHA-256 {file_sha256(args.ours)})")
    print(f"             reference {args.reference} (SHA-256 {file_sha256(args.reference)})")
    for problem in result.problems:
        print(f"MISMATCH: {problem}")
    if not result.summary.empty:
        print(f"{'metric':<34} {'max abs':>12} {'max rel':>12} {'failing':>8}")
        for row in result.summary.itertuples(index=False):
            print(f"{row.metric:<34} {row.max_abs:>12.3e} {row.max_rel:>12.3e} {row.failing_rows:>8d}")
    if not result.failures.empty:
        print("rows beyond 1e-9 relative and 1e-12 absolute:")
        print(result.failures.to_string(index=False))
    print(f"compare {'PASSED' if result.ok else 'FAILED'} ({len(result.summary)} metrics; "
          f"tolerance {RELATIVE_TOLERANCE} relative or {ABSOLUTE_TOLERANCE} absolute)")
    return 0 if result.ok else 1


# --- the reference files of the parity check -------------------------------------------------------------------


def reference_episode_paths(results_root: str, date: str = "2025-01-07", training_seed: int = 42) -> dict[str, str]:
    """The pipeline's final-stage episode files of one retrospective day under a results root (a POSIX path, e.g. on
    Snellius), from scripts.pipeline.common: B0's seeded b0_test_episodes.csv (all test days: select Day == date)
    and B3's file in its eval-part directory (eval_episode_path, through eval_part_dir)."""

    cfg = ReplayExperimentConfig()
    b3 = eval_episode_path(cfg, model="b3", date=date, split="retrospective_holdout", training_seed=training_seed)
    root = PurePosixPath(results_root)
    return {
        "b0": str(root / seeded_result_name("b0_test_episodes.csv", training_seed)),
        "b3": str(root / PurePosixPath(b3.relative_to(cfg.results_dir).as_posix())),
    }


# --- command line ----------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    replay = commands.add_parser("run", help="the shadow replay of one model on one day file")
    replay.add_argument("--model", choices=MODELS, required=True)
    replay.add_argument("--day-file", required=True, type=Path)
    replay.add_argument("--date", required=True)
    replay.add_argument("--params", required=True, type=Path)
    replay.add_argument("--out-dir", required=True, type=Path)
    replay.add_argument("--gamma-file", type=Path)
    replay.add_argument("--model-zip", type=Path)
    replay.add_argument("--vecnorm", type=Path)
    replay.add_argument("--fill-seeds", default="42 43")
    replay.add_argument("--rollouts", type=int, default=20)
    check = commands.add_parser("compare", help="one day's rows of two episode files")
    check.add_argument("--ours", required=True, type=Path)
    check.add_argument("--reference", required=True, type=Path)
    check.add_argument("--day", required=True)
    args = parser.parse_args(argv)
    try:
        return run(args) if args.command == "run" else compare(args)
    except ShadowRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
