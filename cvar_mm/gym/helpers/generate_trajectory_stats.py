"""
Streaming trajectory statistics.

Runs the environment once and computes all Falces Marin et al. (2022)
metrics **inline** during the rollout, using O(N) memory instead of
storing the full O(N × T) observation/action/reward history.

This solves two problems:
    1. **Memory**: 1000 trajectories × 714k steps × 4 features × 8 bytes
       = 21.3 GiB with full storage.  With streaming: ~40 KB total.
    2. **Speed**: metrics are computed from a single rollout instead of
       re-running the environment four times.

Running statistics tracked per trajectory (all shape (N,)):
    - Welford accumulators for Sharpe:  Σr, Σr², count
    - Welford accumulators for Sortino: Σr_neg, Σr²_neg, count_neg
    - Max drawdown: running_max_pnl, max_drawdown
    - P&L-to-MAP: Σ|q|, count
    - Spread: Σ(δ_bid + δ_ask), count
    - Terminal state: final cash, inventory, midprice, PnL
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cvar_mm.gym.trading_environment import TradingEnvironment
from cvar_mm.agents.agent import Agent
from cvar_mm.gym.index_names import CASH_INDEX, INVENTORY_INDEX, ASSET_PRICE_INDEX


def _metric_state(env: TradingEnvironment, obs: np.ndarray) -> np.ndarray:
    state = getattr(env, "metric_state", None)
    if callable(state):
        return state()
    if state is not None:
        return np.asarray(state)
    return obs


def _depth_action(env: TradingEnvironment, action: np.ndarray) -> np.ndarray:
    depths = getattr(env, "last_depth_action", None)
    if depths is not None:
        return np.asarray(depths)
    return action


def generate_trajectory_stats(
    env: TradingEnvironment,
    agent: Agent,
    seed: int | None = None,
    fixed_clock_seconds: float = 60.0,
) -> dict[str, np.ndarray]:
    """
    Run one full episode and compute per-trajectory statistics inline.

    Parameters
    ----------
    env : TradingEnvironment
    agent : Agent
    seed : int | None

    Returns
    -------
    stats : dict[str, np.ndarray]
        All values have shape ``(num_trajectories,)``:

        - ``total_pnl``        : final mark-to-market PnL
        - ``terminal_q``       : terminal inventory
        - ``mean_spread``      : time-averaged bid-ask spread
        - ``sharpe``           : Sharpe ratio  (Falces Marin eq. 16)
        - ``sortino``          : Sortino ratio as Falces Marin et al. (2022)
                                 eq. (20) defines it (D-13): the mean of all
                                 fixed-clock (60 s) wealth changes divided by
                                 the population std of the negative changes
                                 about their own mean; 0 with fewer than two
                                 negative changes.  This is not the Sortino &
                                 Price (1994) downside deviation
                                 sqrt(E[min(r, 0)^2]) with target 0.
        - ``max_drawdown``     : Maximum drawdown (§5.3)
        - ``pnl_to_map``       : P&L / mean(|q|) (Falces Marin eq. 21)
        - ``n_steps``          : number of steps executed (scalar)
    """
    if seed is not None:
        env.seed(seed)

    N = env.num_trajectories
    obs, _ = env.reset()
    metric_obs = _metric_state(env, obs)

    # ── Running accumulators (all shape (N,)) ─────────────────
    # PnL path
    pnl = (
        metric_obs[:, CASH_INDEX]
        + metric_obs[:, INVENTORY_INDEX] * metric_obs[:, ASSET_PRICE_INDEX]
    )
    running_max_pnl = pnl.copy()
    max_drawdown = np.zeros(N)

    if fixed_clock_seconds <= 0:
        raise ValueError("fixed_clock_seconds must be positive")

    # Snapshot-return accumulators (supplementary only)
    sum_r = np.zeros(N)
    sum_r2 = np.zeros(N)
    sum_r_neg = np.zeros(N)
    sum_r2_neg = np.zeros(N)
    count_neg = np.zeros(N)

    # Primary fixed-clock economic PnL-return accumulators.
    fixed_sum_r = np.zeros(N)
    fixed_sum_r2 = np.zeros(N)
    fixed_sum_r_neg = np.zeros(N)
    fixed_sum_r2_neg = np.zeros(N)
    fixed_count_neg = np.zeros(N)
    fixed_count = 0
    next_clock_time = float(fixed_clock_seconds)
    last_clock_wealth = pnl.copy()
    previous_wealth = pnl.copy()

    # Inventory and spread
    sum_abs_q = np.zeros(N)
    near_cap_threshold = 0.8 * env.max_inventory
    sum_near_cap = np.zeros(N)
    sum_spread = np.zeros(N)
    elapsed_weight = 0.0
    step_count = 0

    # ── Main loop ─────────────────────────────────────────────
    while True:
        action = agent.get_action(obs)
        if action.ndim == 1:
            action = np.repeat(action.reshape(1, -1), N, axis=0)

        obs, reward, done, truncated, info = env.step(action)
        metric_obs = _metric_state(env, obs)
        depth_action = _depth_action(env, action)
        step_count += 1

        del reward  # evaluation semantics always use economic wealth changes

        current_pnl = (
            metric_obs[:, CASH_INDEX]
            + metric_obs[:, INVENTORY_INDEX] * metric_obs[:, ASSET_PRICE_INDEX]
        )
        r = current_pnl - pnl

        # Sharpe accumulators
        sum_r += r
        sum_r2 += r * r

        # Sortino accumulators
        neg_mask = r < 0
        sum_r_neg += np.where(neg_mask, r, 0.0)
        sum_r2_neg += np.where(neg_mask, r * r, 0.0)
        count_neg += neg_mask.astype(np.float64)

        # PnL path → drawdown
        pnl = current_pnl
        running_max_pnl = np.maximum(running_max_pnl, pnl)
        max_drawdown = np.maximum(max_drawdown, running_max_pnl - pnl)

        # Inventory and spread
        midprice_model = env.model_dynamics.midprice_model
        dt = float(midprice_model.dt_array[midprice_model.step_index])
        if dt > 0:
            elapsed_weight += dt
            sum_abs_q += np.abs(metric_obs[:, INVENTORY_INDEX]) * dt
            sum_near_cap += (
                np.abs(metric_obs[:, INVENTORY_INDEX]) >= near_cap_threshold
            ) * dt
            sum_spread += (depth_action[:, 0] + depth_action[:, 1]) * dt

        current_time = float(metric_obs[0, 2])
        while next_clock_time <= current_time + 1e-12:
            sampled_wealth = (
                pnl if abs(next_clock_time - current_time) <= 1e-12 else previous_wealth
            )
            fixed_return = sampled_wealth - last_clock_wealth
            fixed_sum_r += fixed_return
            fixed_sum_r2 += fixed_return * fixed_return
            fixed_negative = fixed_return < 0
            fixed_sum_r_neg += np.where(fixed_negative, fixed_return, 0.0)
            fixed_sum_r2_neg += np.where(fixed_negative, fixed_return * fixed_return, 0.0)
            fixed_count_neg += fixed_negative
            fixed_count += 1
            last_clock_wealth = sampled_wealth.copy()
            next_clock_time += fixed_clock_seconds
        previous_wealth = pnl.copy()

        if done[0]:
            break

    # ── Compute final metrics ─────────────────────────────────
    T = step_count
    mean_r = sum_r / T
    var_r = sum_r2 / T - mean_r ** 2
    std_r = np.sqrt(np.maximum(var_r, 0.0))

    # Sortino: std of negative returns only
    safe_count_neg = np.maximum(count_neg, 2.0)
    mean_r_neg = sum_r_neg / safe_count_neg
    var_r_neg = sum_r2_neg / safe_count_neg - mean_r_neg ** 2
    std_r_neg = np.sqrt(np.maximum(var_r_neg, 0.0))

    snapshot_sharpe = np.divide(mean_r, std_r, out=np.zeros(N), where=std_r > 1e-15)
    snapshot_sortino = np.divide(
        mean_r,
        std_r_neg,
        out=np.zeros(N),
        where=(std_r_neg > 1e-15) & (count_neg >= 2),
    )

    if fixed_count:
        fixed_mean = fixed_sum_r / fixed_count
        fixed_variance = fixed_sum_r2 / fixed_count - fixed_mean**2
        fixed_std = np.sqrt(np.maximum(fixed_variance, 0.0))
        safe_fixed_negative = np.maximum(fixed_count_neg, 2.0)
        fixed_negative_mean = fixed_sum_r_neg / safe_fixed_negative
        fixed_negative_variance = (
            fixed_sum_r2_neg / safe_fixed_negative - fixed_negative_mean**2
        )
        fixed_negative_std = np.sqrt(np.maximum(fixed_negative_variance, 0.0))
        sharpe = np.divide(
            fixed_mean, fixed_std, out=np.zeros(N), where=fixed_std > 1e-15
        )
        sortino = np.divide(
            fixed_mean,
            fixed_negative_std,
            out=np.zeros(N),
            where=(fixed_negative_std > 1e-15) & (fixed_count_neg >= 2),
        )
    else:
        sharpe = np.zeros(N)
        sortino = np.zeros(N)

    mean_abs_q = sum_abs_q / max(elapsed_weight, 1e-15)
    pnl_to_map = np.divide(
        pnl, mean_abs_q, out=np.zeros(N), where=mean_abs_q > 1e-15
    )

    return {
        "total_pnl": pnl,
        "terminal_q": metric_obs[:, INVENTORY_INDEX],
        "mean_spread": sum_spread / max(elapsed_weight, 1e-15),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "pnl_to_map": pnl_to_map,
        "mean_abs_q": mean_abs_q,
        "near_cap_fraction": sum_near_cap / max(elapsed_weight, 1e-15),
        "snapshot_sharpe": snapshot_sharpe,
        "snapshot_sortino": snapshot_sortino,
        "fixed_clock_seconds": float(fixed_clock_seconds),
        "fixed_clock_return_count": fixed_count,
        "n_steps": T,
    }


def stats_to_summary(stats: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Convert streaming stats dict into a Falces Marin summary DataFrame.

    Same output format as ``backtest_summary`` but computed from
    the O(N)-memory streaming stats.
    """
    df = pd.DataFrame({
        "Sharpe": stats["sharpe"],
        "Sortino": stats["sortino"],
        "Max DD": stats["max_drawdown"],
        "P&L-to-MAP": stats["pnl_to_map"],
        "Final PnL": stats["total_pnl"],
        "Terminal q": stats["terminal_q"],
        "Mean |q|": stats["mean_abs_q"],
        "Mean spread": stats["mean_spread"],
    })
    if "snapshot_sharpe" in stats:
        df["Sharpe Snapshot (supplementary)"] = stats["snapshot_sharpe"]
    if "snapshot_sortino" in stats:
        df["Sortino Snapshot (supplementary)"] = stats["snapshot_sortino"]
    N = len(df)
    df.index = [f"traj_{i}" for i in range(N)]
    df.index.name = "trajectory"

    agg = pd.DataFrame({
        col: [df[col].mean(), df[col].std(), df[col].median()]
        for col in df.columns
    }, index=["Mean", "Std", "Median"])
    agg.index.name = "trajectory"

    return pd.concat([df, agg])


def stats_to_results_table(stats: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Convert streaming stats into the mbt-gym ``generate_results_table_and_hist``
    DataFrame format.
    """
    pnl = stats["total_pnl"]
    q = stats["terminal_q"]
    spread = stats["mean_spread"]

    results = pd.DataFrame(
        index=["Inventory"],
        columns=[
            "Mean spread", "Mean PnL", "Std PnL",
            "Mean terminal inventory", "Std terminal inventory",
        ],
    )
    results.loc["Inventory", "Mean spread"] = float(np.mean(spread))
    results.loc["Inventory", "Mean PnL"] = float(np.mean(pnl))
    results.loc["Inventory", "Std PnL"] = float(np.std(pnl))
    results.loc["Inventory", "Mean terminal inventory"] = float(np.mean(q))
    results.loc["Inventory", "Std terminal inventory"] = float(np.std(q))
    return results
