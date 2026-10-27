"""T3 and T4: QuoteEngine against the frozen quotes (B3: the wrapper's, every Jan 7 step; B0: a direct
avellaneda_stoikov_quotes call at fast_simulate's tau, every Jan 7 row)."""

from __future__ import annotations

import numpy as np

from hummingbot_adapter.quotes import QuoteEngine
from hummingbot_adapter.tests.support import B0_GAMMA
from cvar_mm.gym.formula_as import avellaneda_stoikov_quotes


def _off_grid(prices: np.ndarray, tick: float) -> np.ndarray:
    """Prices more than float noise away from the tick grid (the mid clamp at an even spread gives S -/+ tick/2)."""

    ticks = prices / tick
    return np.abs(ticks - np.round(ticks)) > 1e-9


def _engine(settings) -> QuoteEngine:
    gamma_min, gamma_max = settings["gamma_range"]
    return QuoteEngine(
        sigma=settings["sigma"], kappa=settings["kappa"], tick_size=settings["tick_size"],
        gamma_min=gamma_min, gamma_max=gamma_max, skew_ticks_max=settings["skew_ticks_max"],
    )


def test_t3_b3_quotes_equal_the_wrapper_at_every_jan7_step(jan7_replay, numpy_actions):
    replay = jan7_replay
    engine = _engine(replay.settings)
    T = replay.terminal_time
    mismatches, numpy_differs = [], 0
    for k, action in enumerate(replay.actions):
        tau = T - replay.t[k]
        quotes = engine.b3(action, replay.mid[k], replay.q[k], tau)
        if not (quotes["bid"] == replay.bid[k] and quotes["ask"] == replay.ask[k]):
            mismatches.append(k)
        from_numpy = engine.b3(numpy_actions[k], replay.mid[k], replay.q[k], tau)
        numpy_differs += (from_numpy["bid"], from_numpy["ask"]) != (quotes["bid"], quotes["ask"])
    tick = replay.settings["tick_size"]
    off_grid = _off_grid(replay.bid, tick) | _off_grid(replay.ask, tick)
    print(f"\nHB-PARITY T3: {len(replay.actions)} steps, {len(mismatches)} bid/ask mismatches; for information, "
          f"quotes from NumpyPolicy actions differ from quotes from SB3 actions on {numpy_differs} steps, and "
          f"{int(off_grid.sum())} steps quote a bid or ask off the tick grid")
    assert mismatches == []


def test_t4_b0_quotes_equal_a_direct_call_at_every_jan7_row(jan7_replay, inputs):
    assert float(inputs["b0_gamma"].read_text(encoding="utf-8").strip()) == B0_GAMMA
    replay = jan7_replay
    settings = replay.settings
    engine = _engine(settings)
    S, dt = replay.S, replay.dt
    terminal_time = float(dt.sum())  # run_b0_fair.py's terminal_time
    tau = np.maximum(terminal_time - np.cumsum(dt), 0.0)  # fast_simulate's tau_arr
    q = np.random.default_rng(42).integers(-10, 11, size=len(S))
    keys = ("bid", "ask", "delta_bid", "delta_ask", "gamma", "skew")
    mismatches, off_grid = [], 0
    for k in range(len(S)):
        quotes = engine.b0(S[k], q[k], tau[k], B0_GAMMA)
        off_grid += bool(_off_grid(np.array([quotes["bid"], quotes["ask"]]), settings["tick_size"]).any())
        _, direct = avellaneda_stoikov_quotes(
            midprice=np.full(1, S[k]),
            inventory=np.full(1, float(q[k])),
            tau=np.full(1, tau[k]),
            gamma=np.full(1, B0_GAMMA),
            skew=np.zeros(1),
            sigma=settings["sigma"],
            kappa=settings["kappa"],
            tick_size=settings["tick_size"],
        )
        if any(quotes[key] != float(direct[key][0]) for key in keys):
            mismatches.append(k)
    print(f"\nHB-PARITY T4: {len(S)} rows, {len(mismatches)} mismatches; q in [{q.min()}, {q.max()}]; "
          f"tau in [{tau.min()!r}, {tau.max()!r}] s; for information, {off_grid} rows quote off the tick grid")
    assert len(S) == len(replay.raw_obs) and np.unique(q).size == 21
    assert mismatches == []
