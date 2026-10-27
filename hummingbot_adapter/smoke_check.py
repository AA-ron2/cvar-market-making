"""Checks of one bot day's outputs (WO-HB2 step 4 b, c, d and g), from the files the bot wrote:

    python -m hummingbot_adapter.smoke_check <bot dir> <date> --calibration <hb_calibration.json>

- b: the reconciliation residual of every steps row is 0 DOGE, and q equals the position the DOGE balance
  implies (reference from the state or the day's health rows);
- c: the largest number of new orders in any 10 s window, and the day's projected count;
- d: the snapshot file loads with the frozen load_single_day_with_features; one steps row per snapshot row
  with the same timestamps; for B3, the frozen FormulaASActionWrapper run on the file reproduces the logged
  log_mid_rel, rolling_vol_ticks, momentum_ticks, spread_ticks and lob_imbalance bitwise (float32), and
  inventory_norm and time_remaining_frac equal clip(q/10) and clip((T - t)/T) with the bot's own q, its
  t_elapsed = (ts_us - 00:00 UTC)/1e6 and T = 86,400 s;
- g: every submitted limit price is on the 0.00001 grid; round-away counts per side (steps and orders).
Runs on the analysis machine (the frozen wrapper needs gymnasium), never inside the Hummingbot container.
Exit code 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from hummingbot_adapter.execution import US_PER_SECOND, DayClock
from cvar_mm.gym.formula_as import FORMULA_FEATURE_NAMES

TICK = Decimal("0.00001")
MARKET_FEATURES = ("log_mid_rel", "rolling_vol_ticks", "momentum_ticks", "spread_ticks", "lob_imbalance")


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def check_residual(steps: list[dict[str, str]], state: dict[str, Any] | None, date: str) -> dict[str, Any]:
    residuals = [Decimal(row["residual_doge"]) for row in steps]
    worst = max((abs(value) for value in residuals), default=Decimal(0))
    result: dict[str, Any] = {"rows": len(steps), "max_abs_residual_doge": str(worst), "pass": worst == 0}
    if state is not None and state["day"]["date"] == date and steps:
        reference = Decimal(state["ledger"]["doge_reference"])
        last = steps[-1]
        implied = (Decimal(last["doge_balance"]) - reference + Decimal(last["fees_doge"])) / 100
        result.update({"last_q": last["q"], "position_from_balance": str(implied),
                       "q_equals_account": Decimal(last["q"]) == implied})
        result["pass"] = result["pass"] and result["q_equals_account"]
    return result


def check_rate(orders: list[dict[str, str]], steps: list[dict[str, str]]) -> dict[str, Any]:
    submits = sorted(int(row["ts_us"]) for row in orders if row["event"] == "submit")
    worst, start = 0, 0
    for end, ts in enumerate(submits):
        while submits[start] <= ts - 10 * US_PER_SECOND:
            start += 1
        worst = max(worst, end - start + 1)
    span_s = (int(steps[-1]["ts_us"]) - int(steps[0]["ts_us"])) / 1e6 if len(steps) > 1 else 0.0
    projected = len(submits) / span_s * 86_400 if span_s else 0.0
    return {"new_orders": len(submits), "span_s": span_s, "max_new_orders_10s": worst,
            "projected_per_day": round(projected), "skips": sum(row["event"] == "skip" for row in orders),
            "pass": worst < 100 and projected < 200_000}


def check_grid(orders: list[dict[str, str]], steps: list[dict[str, str]]) -> dict[str, Any]:
    limits = [row for row in orders if row["event"] == "submit" and row["order_type"] == "limit"]
    off_grid = [row["order_id"] for row in limits if Decimal(row["price"]) % TICK != 0]
    return {
        "limit_submits": len(limits), "off_grid": off_grid,
        "round_away_steps": {"bid": sum(row["bid_rounded"] == "1" for row in steps),
                             "ask": sum(row["ask_rounded"] == "1" for row in steps)},
        "round_away_orders": {side: sum(row["reason"] == "quote_rounded" and row["side"] == side for row in limits)
                              for side in ("buy", "sell")},
        "pass": not off_grid,
    }


def check_snapshots(day_dir: Path, date: str, steps: list[dict[str, str]], calibration: dict[str, Any],
                    model: str) -> dict[str, Any]:
    from cvar_mm.gym.data_loader import load_single_day_with_features

    snapshot = day_dir / f"snapshots_{date}.csv"
    S, dt, _, features = load_single_day_with_features(str(snapshot), tick_size=1e-5)
    with open(snapshot, encoding="utf-8", newline="") as handle:
        timestamps = [int(row[2]) for row in list(csv.reader(handle))[1:]]
    aligned = len(timestamps) == len(steps) and all(
        ts == int(row["ts_us"]) for ts, row in zip(timestamps, steps))
    mids = np.array([float(row["mid"]) for row in steps])
    result: dict[str, Any] = {"rows": len(S), "steps_aligned": aligned,
                              "mid_mismatches": int(np.sum(mids != S)) if aligned else None}
    ok = aligned and result["mid_mismatches"] == 0
    if model == "B3" and aligned:
        logged = {name: np.array([np.float32(float(row[name])) for row in steps], dtype=np.float32)
                  for name in FORMULA_FEATURE_NAMES}
        wrapper_obs = _wrapper_observations(S, dt, features, calibration)
        for index, name in enumerate(FORMULA_FEATURE_NAMES):
            if name in MARKET_FEATURES:
                mismatches = int(np.sum(wrapper_obs[:, index].view(np.uint32) != logged[name].view(np.uint32)))
                result[f"{name}_mismatches"] = mismatches
                ok = ok and mismatches == 0
        day_start = DayClock.day_start_us(int(steps[0]["ts_us"]))
        t = np.array([float(row["t_elapsed"]) for row in steps])
        exact_t = np.array([(int(row["ts_us"]) - day_start) / 1e6 for row in steps])
        q = np.array([float(row["q"]) for row in steps])
        T = 86_400.0
        expected_time = np.array([np.clip((T - value) / T, 0.0, 1.0) for value in exact_t]).astype(np.float32)
        expected_inventory = np.clip(q / 10.0, -1.0, 1.0).astype(np.float32)
        result["t_elapsed_mismatches"] = int(np.sum(t != exact_t))
        result["time_remaining_frac_mismatches"] = int(np.sum(
            expected_time.view(np.uint32) != logged["time_remaining_frac"].view(np.uint32)))
        result["inventory_norm_mismatches"] = int(np.sum(
            expected_inventory.view(np.uint32) != logged["inventory_norm"].view(np.uint32)))
        ok = ok and result["t_elapsed_mismatches"] == result["time_remaining_frac_mismatches"] == 0
        ok = ok and result["inventory_norm_mismatches"] == 0
    result["pass"] = ok
    return result


def _wrapper_observations(S: np.ndarray, dt: np.ndarray, features: dict[str, np.ndarray],
                          calibration: dict[str, Any]) -> np.ndarray:
    """The frozen FormulaASActionWrapper's raw observation at every row (any action: the market columns do
    not depend on it), as hummingbot_adapter/tests/conftest.py builds the replay."""

    from cvar_mm.agents.agent import Agent
    from cvar_mm.gym.experiment_config import ReplayExperimentConfig
    from cvar_mm.gym.helpers.generate_trajectory_stats import generate_trajectory_stats
    from cvar_mm.gym.notebook_support import build_formula_replay_env
    from cvar_mm.rewards import PnLReward

    config = dataclasses.replace(ReplayExperimentConfig(), A=float(calibration["A"]), kappa=float(calibration["kappa"]))
    env = build_formula_replay_env(S, dt, config, sigma=float(calibration["sigma"]), market_features=features,
                                   reward_fn=PnLReward(), num_trajectories=1)
    observations: list[np.ndarray] = []

    class Recorder(Agent):
        def get_action(self, state: np.ndarray) -> np.ndarray:
            observations.append(np.array(state[0], dtype=np.float32))
            return np.zeros((1, 2), dtype=np.float32)

    generate_trajectory_stats(env, Recorder(), seed=42, fixed_clock_seconds=config.fixed_clock_seconds)
    observations.append(np.array(env._obs()[0], dtype=np.float32))
    return np.array(observations, dtype=np.float32)


def check_day(bot_dir: Path, date: str, calibration: dict[str, Any]) -> dict[str, Any]:
    day_dir = bot_dir / date
    steps = _rows(day_dir / f"steps_{date}.csv")
    orders = _rows(day_dir / f"orders_{date}.csv")
    fills = _rows(day_dir / f"fills_{date}.csv")
    state_path = bot_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
    model = "B3" if "z_inventory_norm" in (steps[0] if steps else {}) else "B0"
    report = {
        "bot_dir": str(bot_dir), "date": date, "model": model,
        "fills": {side: sum(row["side"] == side for row in fills) for side in ("buy", "sell")},
        "b_residual": check_residual(steps, state, date),
        "c_rate": check_rate(orders, steps),
        "d_snapshots": check_snapshots(day_dir, date, steps, calibration, model),
        "g_grid": check_grid(orders, steps),
    }
    report["pass"] = all(report[key]["pass"] for key in ("b_residual", "c_rate", "d_snapshots", "g_grid"))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bot_dir", type=Path)
    parser.add_argument("date")
    parser.add_argument("--calibration", type=Path, required=True)
    args = parser.parse_args(argv)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    report = check_day(args.bot_dir, args.date, calibration)
    print(json.dumps(report, indent=1, default=str))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
