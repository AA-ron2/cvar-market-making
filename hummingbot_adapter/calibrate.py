"""The pre-registered calibration rule (PROTOCOL §4, decision D2): the frozen procedure on a Tardis day.

Run mode:
    python -m hummingbot_adapter.calibrate --inputs <file ...> --env-file configs/final_pipeline.env --out <json>
Verify mode (the frozen Jan 1-6 calibration, recomputed):
    python -m hummingbot_adapter.calibrate --verify-frozen --data-dir <dir> --expected <formula_train_params.json>
        --env-file configs/final_pipeline.env --out <json>

Run mode mirrors scripts.pipeline.common.load_or_create_train_params: the config from
calibration_config_from_env(0.00001) after the env file's CALIBRATION_* exports, the arrays of
cvar_mm.gym.data_loader.load_single_day_calibration_arrays, one pooled cvar_mm.gym.calibration.calibrate_from_days.
It refuses unless PROTOCOL.md is committed and unmodified, and any input dated 2026-10-20 or later (the run
days). Every refusal runs before any input is opened. A .csv.gz is stream-decompressed to the .csv next to it.
Exit codes: 2, a refusal; an uncaught exception (1), an error of the frozen loader or fit, the case of PROTOCOL
§4's fallback; 3, the calibration succeeded but --out could not be written (not a §4 case). The output JSON is
never overwritten.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from hummingbot_adapter.provenance import REPO_ROOT, file_sha256, git_output, is_inside
from cvar_mm.gym import calibration, data_loader
from scripts.pipeline import common

PROTOCOL_PATH = "hummingbot_adapter/PROTOCOL.md"
FIRST_RUN_DAY = date(2026, 10, 20)  # PROTOCOL §6: the prospective days start here
TICK_SIZE = 0.00001
CALIBRATION_KEYS = (
    "CALIBRATION_PROXY",
    "CALIBRATION_START_RULE",
    "CALIBRATION_DEPTH_TICKS",
    "CALIBRATION_CENSOR_SECONDS",
    "CALIBRATION_START_STRIDE_SECONDS",
    "CALIBRATION_MIN_HITS",
    "CALIBRATION_VOL_WINDOW_SECONDS",
)
FROZEN_DAYS = tuple(f"2025-01-0{day}" for day in range(1, 7))
FROZEN_FILE = "binance_book_snapshot_25_{date}_DOGEUSDT.csv"
FROZEN_PARAMS_SHA256 = "2d0b4a916bcf52e8641fac0a6e34f93f088166ee87c4b98b4d24975f1f6426f8"  # PROTOCOL §2
VERIFY_TOLERANCE = 1e-12
EXPORT_LINE = re.compile(r"^export\s+(CALIBRATION_\w*)=(.*)$")
DATE_IN_NAME = re.compile(r"\d{4}-\d{2}-\d{2}")


class CalibrationRefused(RuntimeError):
    """An input or the repository state fails a pre-registered rule."""


def _git(repo_root: Path, *args: str) -> str:
    result = git_output(repo_root, *args)
    if result.returncode != 0:
        raise CalibrationRefused(f"git {' '.join(args)} failed in {repo_root}: {result.stderr.strip()}")
    return result.stdout.rstrip("\n")


def require_committed_protocol(repo_root: Path = REPO_ROOT) -> str:
    """PROTOCOL.md's last commit; refuses if it has none or the file differs from it."""

    commit = _git(repo_root, "log", "-1", "--format=%h", "--", PROTOCOL_PATH)
    if not commit:
        raise CalibrationRefused(f"{PROTOCOL_PATH} is not committed in {repo_root}")
    status = _git(repo_root, "status", "--porcelain", "--", PROTOCOL_PATH)
    if status:
        raise CalibrationRefused(f"{PROTOCOL_PATH} differs from its commit {commit}: {status.strip()}")
    return commit


def file_date(path: str | Path) -> date:
    """The one date in the file name; refuses none or several."""

    found = set(DATE_IN_NAME.findall(Path(path).name))
    if len(found) != 1:
        raise CalibrationRefused(f"{Path(path).name}: expected exactly one YYYY-MM-DD in the name, found {sorted(found)}")
    try:
        return date.fromisoformat(found.pop())
    except ValueError as exc:
        raise CalibrationRefused(f"{Path(path).name}: {exc}") from exc


def read_calibration_env(env_file: str | Path) -> dict[str, str]:
    """The env file's `export CALIBRATION_*=VALUE` lines, read-only and CRLF-safe: each line is stripped of
    surrounding whitespace (including a carriage return), then the value of its quotes. Refuses a missing,
    repeated or unknown key and a value with shell syntax this parser does not evaluate."""

    values: dict[str, str] = {}
    for raw in Path(env_file).read_bytes().decode("utf-8").split("\n"):
        match = EXPORT_LINE.match(raw.strip())
        if match is None:
            continue
        key, value = match.group(1), match.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key not in CALIBRATION_KEYS:
            raise CalibrationRefused(f"{env_file}: unknown key {key}")
        if key in values:
            raise CalibrationRefused(f"{env_file}: {key} is exported twice")
        if not value or any(ch in value for ch in "\"'#$`\\"):
            raise CalibrationRefused(f"{env_file}: {key} has a value this parser does not accept: {value!r}")
        values[key] = value
    missing = [key for key in CALIBRATION_KEYS if key not in values]
    if missing:
        raise CalibrationRefused(f"{env_file}: missing {', '.join(missing)}")
    return values


def calibration_config(env_file: str | Path) -> tuple[Any, dict[str, str]]:
    """The env file's values into os.environ, then the pipeline's own FirstPassageCalibrationConfig."""

    values = read_calibration_env(env_file)
    os.environ.update(values)
    return common.calibration_config_from_env(TICK_SIZE), values


def csv_target(path: Path, repo_root: Path = REPO_ROOT) -> Path | None:
    """The .csv a .csv.gz input decompresses to, or None for a .csv input. Refuses another suffix, a target
    inside the repository and a .partial left by a killed run. Opens nothing."""

    if path.name.endswith(".csv"):
        return None
    if not path.name.endswith(".csv.gz"):
        raise CalibrationRefused(f"{path}: expected a .csv or .csv.gz file")
    target = path.with_name(path.name[: -len(".gz")])
    if is_inside(target, repo_root):
        raise CalibrationRefused(f"{target} would be inside the repository; decompress outside it")
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        raise CalibrationRefused(f"{partial} exists, left by an interrupted run; remove it first")
    return target


def materialize_csv(path: str | Path, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """The CSV to load and the SHA-256 of the input (and of the CSV). A .csv.gz is stream-decompressed to the
    .csv next to it; an existing .csv is kept only if it has the same bytes. A failed decompression removes
    its partial output and propagates."""

    path = Path(path)
    target = csv_target(path, repo_root)
    if target is None:
        digest = file_sha256(path)
        return {"input": str(path), "input_sha256": digest, "csv": str(path), "csv_sha256": digest}
    partial = target.with_name(target.name + ".partial")
    input_sha256 = file_sha256(path)
    digest = hashlib.sha256()
    created = False
    try:
        with gzip.open(path, "rb") as reader, open(partial, "xb") as writer:
            created = True
            while chunk := reader.read(1 << 20):
                digest.update(chunk)
                writer.write(chunk)
    except BaseException:
        if created:
            partial.unlink(missing_ok=True)
        raise
    csv_sha256 = digest.hexdigest()
    if target.exists():
        existing = file_sha256(target)
        partial.unlink()
        if existing != csv_sha256:
            raise CalibrationRefused(f"{target} exists with SHA-256 {existing}, not the {csv_sha256} of {path.name}")
    else:
        partial.replace(target)
    return {"input": str(path), "input_sha256": input_sha256, "csv": str(target), "csv_sha256": csv_sha256}


def calibrate(inputs: list[str | Path], env_file: str | Path, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Run mode: the refusals, then the frozen loader and the pooled fit; returns the JSON payload."""

    protocol_commit = require_committed_protocol(repo_root)
    paths = [Path(item) for item in inputs]
    if not paths:
        raise CalibrationRefused("no inputs")
    dates = [file_date(path) for path in paths]
    if len(set(dates)) != len(dates):
        raise CalibrationRefused(f"two inputs have the same date: {[str(day) for day in dates]}")
    for path, day in zip(paths, dates):
        if day >= FIRST_RUN_DAY:
            raise CalibrationRefused(f"{path.name} is dated on or after {FIRST_RUN_DAY}, a run day (PROTOCOL §6)")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise CalibrationRefused(f"no such input file: {', '.join(missing)}")
    for path in paths:
        csv_target(path, repo_root)
    config, env_values = calibration_config(env_file)
    git_commit = _git(repo_root, "rev-parse", "HEAD")
    git_status = _git(
        repo_root, "status", "--porcelain", "--untracked-files=all", "--",
        "cvar_mm", "scripts/pipeline/common.py", "hummingbot_adapter",
    )

    files, mids, dts, bids, asks = [], [], [], [], []
    for path in paths:
        info = materialize_csv(path, repo_root)
        mid, dt, _, best_bid, best_ask = data_loader.load_single_day_calibration_arrays(info["csv"])
        info.update(
            rows=int(len(mid)),
            mid_mean=float(np.mean(mid)),
            mid_min=float(np.min(mid)),
            mid_max=float(np.max(mid)),
        )
        files.append(info)
        mids.append(mid)
        dts.append(dt)
        bids.append(best_bid)
        asks.append(best_ask)
    result = calibration.calibrate_from_days(
        mids, dts, config=config, daily_best_bid=bids, daily_best_ask=asks
    )
    diagnostics = result.diagnostics
    return {
        "sigma": result.sigma,
        "A": result.A,
        "kappa": result.kappa,
        "r_squared_log": diagnostics.pooled_fit.r_squared_log,
        "rmse_log": diagnostics.pooled_fit.rmse_log,
        "procedure": (
            "cvar_mm.gym.calibration.calibrate_from_days on cvar_mm.gym.data_loader.load_single_day_calibration_arrays; "
            "config scripts.pipeline.common.calibration_config_from_env(0.00001) (PROTOCOL §4)"
        ),
        "calibration_config": asdict(config),
        "calibration_env": env_values,
        "pooled_fit": asdict(diagnostics.pooled_fit),
        "bid_fit_diagnostic": None if diagnostics.bid_fit is None else asdict(diagnostics.bid_fit),
        "ask_fit_diagnostic": None if diagnostics.ask_fit is None else asdict(diagnostics.ask_fit),
        "pooled_curve": common._curve_payload(diagnostics.pooled_curve),
        "bid_curve": common._curve_payload(diagnostics.bid_curve),
        "ask_curve": common._curve_payload(diagnostics.ask_curve),
        "A_dt_quantiles": diagnostics.A_dt_quantiles,
        "A_dt_fraction_above": diagnostics.A_dt_fraction_above,
        "files": files,
        "protocol_commit": protocol_commit,
        "git_commit": git_commit,
        "git_status": git_status,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def verify_frozen(
    data_dir: str | Path, expected_path: str | Path, env_file: str | Path, repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    """Verify mode: run mode on Jan 1-6, compared with the frozen sigma_train, A_train and kappa_train."""

    expected_sha256 = file_sha256(expected_path)
    if expected_sha256 != FROZEN_PARAMS_SHA256:
        raise CalibrationRefused(
            f"{expected_path} has SHA-256 {expected_sha256}, not the frozen {FROZEN_PARAMS_SHA256} (PROTOCOL §2)"
        )
    expected = json.loads(Path(expected_path).read_text(encoding="utf-8"))
    inputs = [Path(data_dir) / FROZEN_FILE.format(date=day) for day in FROZEN_DAYS]
    payload = calibrate(inputs, env_file, repo_root)
    comparisons = {}
    for key, reference_key in (("sigma", "sigma_train"), ("A", "A_train"), ("kappa", "kappa_train")):
        value, reference = float(payload[key]), float(expected[reference_key])
        relative = abs(value - reference) / abs(reference)
        comparisons[key] = {"value": value, "expected": reference, "relative_difference": relative}
    payload["verify"] = {
        "expected_path": str(expected_path),
        "expected_sha256": expected_sha256,
        "tolerance": VERIFY_TOLERANCE,
        "comparisons": comparisons,
        "passed": all(item["relative_difference"] <= VERIFY_TOLERANCE for item in comparisons.values()),
    }
    return payload


def write_json(payload: dict[str, Any], out: Path) -> None:
    with open(out, "x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", type=Path)
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--verify-frozen", action="store_true")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--expected", type=Path)
    args = parser.parse_args(argv)
    if args.verify_frozen and (args.inputs or not args.data_dir or not args.expected):
        parser.error("--verify-frozen takes --data-dir and --expected, not --inputs")
    if not args.verify_frozen and (not args.inputs or args.data_dir or args.expected):
        parser.error("run mode takes --inputs, not --data-dir or --expected")
    try:
        if args.out.exists():
            raise CalibrationRefused(f"{args.out} exists; it is never overwritten")
        if not args.out.parent.is_dir():
            raise CalibrationRefused(f"the folder of --out does not exist: {args.out.parent}")
        print(f"git: {_git(REPO_ROOT, 'log', '-1', '--oneline')}")
        if args.verify_frozen:
            payload = verify_frozen(args.data_dir, args.expected, args.env_file)
        else:
            payload = calibrate(args.inputs, args.env_file)
    except CalibrationRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    for item in payload["files"]:
        print(f"{Path(item['input']).name}: {item['rows']} rows, mid mean {item['mid_mean']!r} "
              f"min {item['mid_min']!r} max {item['mid_max']!r}, SHA-256 {item['input_sha256']}")
    print(f"sigma {payload['sigma']!r}  A {payload['A']!r}  kappa {payload['kappa']!r}")
    print(f"log-R2 {payload['r_squared_log']!r}  log-RMSE {payload['rmse_log']!r}  "
          f"valid depths {payload['pooled_fit']['valid_depths']}")
    passed = True
    if args.verify_frozen:
        verify = payload["verify"]
        for key, item in verify["comparisons"].items():
            print(f"{key}: computed {item['value']!r}  frozen {item['expected']!r}  "
                  f"relative difference {item['relative_difference']!r}")
        print(f"verify {'PASSED' if verify['passed'] else 'FAILED'} (tolerance {VERIFY_TOLERANCE})")
        passed = verify["passed"]
    try:
        write_json(payload, args.out)
    except OSError as exc:
        print(f"WRITE FAILED after a successful calibration (not a PROTOCOL §4 loading or fitting error): {exc}",
              file=sys.stderr)
        return 3
    print(f"wrote {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
