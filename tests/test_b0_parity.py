"""B0 (fast_rollout) and the B1-B3 environment quote, fill and score identically (K5)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("gymnasium")

import cvar_mm.gym.formula_as as formula_as
import cvar_mm.gym.helpers.fast_rollout as fast_rollout
from cvar_mm.gym.experiment_config import ReplayExperimentConfig
from cvar_mm.gym.formula_as import FormulaASActionConfig, compute_formula_as_depths, map_formula_action
from cvar_mm.gym.helpers.generate_trajectory_stats import generate_trajectory_stats
from cvar_mm.gym.notebook_support import build_formula_replay_env
from cvar_mm.stochastic_processes.fill_probability_models import DepthDependentPoissonFillFunction

TICK = 1e-5


def synthetic_day(snapshots: int = 1201):
    rng = np.random.default_rng(0)
    dt = rng.choice([0.0, 0.5, 1.0], snapshots)  # zero-dt rows: the mid moves, nothing fills
    dt[0] = 0.0
    mid = 0.35 + 0.5 * TICK * np.cumsum(np.r_[0, rng.integers(-1, 2, snapshots - 1)])
    return mid, dt


@pytest.mark.parametrize("q_max", [10, 2])  # q_max = 2: the inventory cap binds
def test_b0_fast_rollout_and_env_path_give_identical_quotes_fills_and_metrics(monkeypatch, q_max):
    mid, dt = synthetic_day()
    A, kappa, sigma, paths, seed, clock, action = 1.0, 1e4, 1e-3, 8, 7, 5.0, 0.5
    gamma = float(map_formula_action(np.array([[action, 0.0]]), FormulaASActionConfig(sigma, kappa, TICK))[0][0])

    path = {"name": "env"}
    quotes: dict[str, list] = {"env": [], "fast": []}
    unclamped: dict[str, list] = {"env": [], "fast": []}
    fills: dict[str, list] = {"env": [], "fast": []}
    shared_quotes = formula_as.avellaneda_stoikov_quotes
    shared_fills = DepthDependentPoissonFillFunction.get_fills

    def recording_quotes(**kwargs):
        depths, diagnostics = shared_quotes(**kwargs)
        quotes[path["name"]].append((diagnostics["bid"].copy(), diagnostics["ask"].copy(), depths.copy()))
        # Item 9 (c): where the A-S quote lay before the mid clamp, and the inventory it was quoted at.
        center = diagnostics["reservation"] + diagnostics["skew"]
        bid = np.floor((center - diagnostics["half_spread"]) / TICK) * TICK
        ask = np.ceil((center + diagnostics["half_spread"]) / TICK) * TICK
        mid_now = np.broadcast_to(kwargs["midprice"], bid.shape)
        unclamped[path["name"]].append((
            bid > mid_now - 0.5 * TICK, ask < mid_now + 0.5 * TICK, np.broadcast_to(kwargs["inventory"], bid.shape).copy()
        ))
        return depths, diagnostics

    def recording_fills(self, depths, dt=None):
        drawn = shared_fills(self, depths, dt)
        fills[path["name"]].append(drawn.copy())
        return drawn

    monkeypatch.setattr(formula_as, "avellaneda_stoikov_quotes", recording_quotes)
    monkeypatch.setattr(fast_rollout, "avellaneda_stoikov_quotes", recording_quotes)
    monkeypatch.setattr(DepthDependentPoissonFillFunction, "get_fills", recording_fills)

    cfg = ReplayExperimentConfig(repo_root=Path.cwd(), A=A, kappa=kappa, tick_size=TICK, q_max=q_max)
    env = build_formula_replay_env(mid, dt, cfg, sigma=sigma, num_trajectories=paths)
    agent = SimpleNamespace(
        get_action=lambda obs: np.tile(np.array([action, 0.0], dtype=np.float32), (paths, 1))
    )
    env_stats = generate_trajectory_stats(env, agent, seed=seed, fixed_clock_seconds=clock)
    path["name"] = "fast"
    fast_stats = fast_rollout.fast_simulate(
        midprices=mid, dt_array=dt, gamma=gamma, sigma=sigma, kappa=kappa, A=A,
        terminal_time=float(dt.sum()), tick_size=TICK, Q_MAX=q_max, num_trajectories=paths,
        seed=seed, fixed_clock_seconds=clock,
    )

    # The env quotes on every row; on zero-dt rows the simulator ignores them and B0 skips them.
    positive = [step for step in range(len(mid) - 1) if dt[step + 1] > 0]
    assert len(quotes["fast"]) == len(fills["fast"]) == len(fills["env"]) == len(positive)
    for step, fast_quote in zip(positive, quotes["fast"]):
        for env_part, fast_part in zip(quotes["env"][step], fast_quote):
            np.testing.assert_array_equal(env_part, fast_part)
    for env_fill, fast_fill in zip(fills["env"], fills["fast"]):
        np.testing.assert_array_equal(env_fill, fast_fill)
    assert env_stats.keys() == fast_stats.keys()
    for key in env_stats:
        np.testing.assert_array_equal(env_stats[key], fast_stats[key], err_msg=key)

    # Meaningful only if the clamp binds, something fills, Sortino is defined somewhere,
    # and with q_max = 2 the inventory reaches the cap, where both paths suppress fills.
    assert any(
        np.any(bid == mid[step] - 0.5 * TICK) or np.any(ask == mid[step] + 0.5 * TICK)
        for step, (bid, ask, _) in zip(positive, quotes["fast"])
    )
    assert sum(int(drawn.sum()) for drawn in fills["fast"]) > 0
    assert np.any(fast_stats["sortino"] != 0)
    assert q_max == 10 or np.any(fast_stats["near_cap_fraction"] > 0)

    # Item 9 (c): the mid clamp itself binds. On some quoted rows the unclamped A-S bid lies above mid - tick/2
    # (or the ask below mid + tick/2), and the quote sits at mid -/+ tick/2, half a tick deep, in both paths (the
    # parity above). The equality check above also holds where floor/ceil alone land on mid -/+ tick/2. With
    # q_max = 10 the inventory never reaches the cap, so here the clamp binds below it, not because of it.
    clamped, below_cap, largest_inventory = 0, 0, 0.0
    for step, (bid, ask, depths), (bid_inside, ask_inside, inventory) in zip(positive, quotes["fast"], unclamped["fast"]):
        assert np.all(bid[bid_inside] == mid[step] - 0.5 * TICK) and np.all(ask[ask_inside] == mid[step] + 0.5 * TICK)
        np.testing.assert_allclose(depths[bid_inside, 0], 0.5 * TICK, rtol=1e-6)
        np.testing.assert_allclose(depths[ask_inside, 1], 0.5 * TICK, rtol=1e-6)
        clamped += int(bid_inside.sum() + ask_inside.sum())
        below_cap += int(((bid_inside | ask_inside) & (np.abs(inventory) < q_max)).sum())
        largest_inventory = max(largest_inventory, float(np.abs(inventory).max()))
    assert clamped > 0
    if q_max == 10:
        assert largest_inventory < q_max and below_cap == clamped > 0


def pre_refactor_depths(*, midprice, inventory, time_elapsed, terminal_time, action, config):
    """compute_formula_as_depths as it stood at a48c9af, before the quote formula was shared."""

    mid = np.asarray(midprice, dtype=np.float64).reshape(-1)
    q = np.asarray(inventory, dtype=np.float64).reshape(-1)
    t = np.asarray(time_elapsed, dtype=np.float64).reshape(-1)
    gamma, skew = map_formula_action(action, config)
    tau = np.maximum(float(terminal_time) - t, 0.0)
    reservation = mid - q * gamma * config.sigma**2 * tau
    half_spread = 0.5 * gamma * config.sigma**2 * tau + (1.0 / gamma) * np.log1p(gamma / config.kappa)
    center = reservation + skew
    bid = np.floor((center - half_spread) / config.tick_size) * config.tick_size
    ask = np.ceil((center + half_spread) / config.tick_size) * config.tick_size
    min_delta = 0.5 * config.tick_size
    bid = np.minimum(bid, mid - min_delta)
    ask = np.maximum(ask, mid + min_delta)
    ask = np.where(ask <= bid, bid + config.tick_size, ask)
    depths = np.column_stack([np.maximum(mid - bid, min_delta), np.maximum(ask - mid, min_delta)])
    return depths.astype(np.float32), bid, ask


@pytest.mark.parametrize("sigma", [2.7e-5, 1e-3])
def test_shared_quote_function_leaves_b1_to_b3_quotes_bitwise_unchanged(sigma):
    rng = np.random.default_rng(3)
    size = 512
    config = FormulaASActionConfig(sigma=sigma, kappa=12_000.0, tick_size=TICK)
    state = dict(
        midprice=0.35 + 0.5 * TICK * rng.integers(-50, 50, size),
        inventory=rng.integers(-10, 11, size).astype(float),
        time_elapsed=rng.uniform(0.0, 86_400.0, size),
        terminal_time=86_400.0,
        action=rng.uniform(-1.0, 1.0, (size, 2)),
        config=config,
    )
    depths, diagnostics = compute_formula_as_depths(**state)
    old_depths, old_bid, old_ask = pre_refactor_depths(**state)
    np.testing.assert_array_equal(depths, old_depths)
    np.testing.assert_array_equal(diagnostics["bid"], old_bid)
    np.testing.assert_array_equal(diagnostics["ask"], old_ask)
