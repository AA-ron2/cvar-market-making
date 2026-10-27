"""
Fast rollout for market-replay A-S simulation.

Bypasses the Gymnasium ``env.step()`` / ``agent.get_action()`` interface
and runs the entire simulation in a tight NumPy loop. Deterministic terms
are pre-computed once, while the stochastic arrival/fill draws are sampled
inside the loop to avoid allocating impractically large ``(T, N, ...)``
buffers for long replay sessions.

Performance comparison (714k steps, N=1000):
    Gymnasium loop:  ~13 minutes   (Python overhead per step)
    fast_simulate:   ~15-30 seconds (pure NumPy arithmetic)

The replay path uses start-of-interval quotes and the exact depth-dependent
Poisson execution CDF, matching the corrected Gymnasium replay environment.
Quotes come from the shared ``formula_as.avellaneda_stoikov_quotes`` (mid
clamp included) and fills from the environment's own execution model, so B0
and B1-B3 quote, fill and score identically for the same gamma (K5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cvar_mm.gym.formula_as import avellaneda_stoikov_quotes
from cvar_mm.stochastic_processes.fill_probability_models import DepthDependentPoissonFillFunction


def fast_simulate(
    midprices: np.ndarray,
    dt_array: np.ndarray,
    gamma: float,
    sigma: float,
    kappa: float,
    A: float,
    terminal_time: float,
    tick_size: float = 0.00001,
    Q_MAX: int = 50,
    num_trajectories: int = 1,
    seed: int | None = None,
    use_linear_approximation: bool = False,
    fixed_clock_seconds: float = 60.0,
) -> dict[str, np.ndarray]:
    """
    Run a full market-replay A-S simulation with streaming metrics.

    Pre-computes deterministic terms once, then loops over T steps doing
    vectorised NumPy arithmetic on ``(N,)`` arrays with per-step random
    draws for arrivals/fills.

    Parameters
    ----------
    midprices : np.ndarray, shape (M,)
        Session midprice series.
    dt_array : np.ndarray, shape (M,)
        Inter-snapshot Δt in seconds (dt[0] = 0).
    gamma, sigma, kappa, A : float
        A-S model parameters.
    terminal_time : float
        T in seconds.
    tick_size : float
    Q_MAX : int
    num_trajectories : int
        N parallel trajectories (independent fills, shared midprice).
    seed : int | None
    use_linear_approximation : bool
        Deprecated robustness option.  If true, linearises the complete
        depth-dependent probability as ``A exp(-kappa*delta) Δt``.

    Returns
    -------
    stats : dict[str, np.ndarray]
        Same keys as ``generate_trajectory_stats``:
        total_pnl, terminal_q, mean_spread, sharpe, sortino,
        max_drawdown, pnl_to_map, mean_abs_q, near_cap_fraction, n_steps.
        ``sortino`` is Falces Marin et al. (2022) eq. (20) (D-13): the mean
        of all fixed-clock (60 s) wealth changes divided by the population
        std of the negative changes about their own mean; 0 with fewer than
        two negative changes.  It is not the Sortino & Price (1994)
        downside deviation sqrt(E[min(r, 0)^2]) with target 0.
    """
    # The environment's execution model, seeded like TradingEnvironment
    # (``seed + 1``): a named fill seed draws the same uniforms in B0 and in
    # the RL environments (common random numbers).
    fill_model = DepthDependentPoissonFillFunction(
        A, kappa, num_trajectories, seed=None if seed is None else seed + 1
    )
    M = len(midprices)
    T = M - 1                       # number of steps
    N = num_trajectories
    if fixed_clock_seconds <= 0:
        raise ValueError("fixed_clock_seconds must be positive")

    # ══════════════════════════════════════════════════════════
    # 1.  PRE-COMPUTE deterministic arrays
    # ══════════════════════════════════════════════════════════
    S = midprices.astype(np.float64)
    dt = dt_array.astype(np.float64)
    t_cumulative = np.cumsum(dt)     # wall-clock time at each snapshot
    tau_arr = np.maximum(terminal_time - t_cumulative, 0.0)  # (M,)

    # The env path quotes on (N,) arrays; B0 uses the same layout.
    gamma_paths = np.full(N, float(gamma))
    no_skew = np.zeros(N)
    near_cap_threshold = 0.8 * Q_MAX

    # ══════════════════════════════════════════════════════════
    # 2.  INITIALISE state vectors  (N,)
    # ══════════════════════════════════════════════════════════
    cash = np.zeros(N)
    q = np.zeros(N)
    pnl = np.zeros(N)               # cash + q * S

    # Streaming statistics
    running_max_pnl = np.zeros(N)
    max_drawdown = np.zeros(N)
    sum_r = np.zeros(N)
    sum_r2 = np.zeros(N)
    sum_r_neg = np.zeros(N)
    sum_r2_neg = np.zeros(N)
    count_neg = np.zeros(N, dtype=np.float64)
    sum_abs_q = np.zeros(N)          # includes initial q=0
    sum_near_cap = np.zeros(N)
    sum_spread = np.zeros(N)
    elapsed_weight = 0.0
    next_clock_time = fixed_clock_seconds
    last_clock_wealth = pnl.copy()
    fixed_sum_r = np.zeros(N)
    fixed_sum_r2 = np.zeros(N)
    fixed_sum_r_neg = np.zeros(N)
    fixed_sum_r2_neg = np.zeros(N)
    fixed_count_neg = np.zeros(N)
    fixed_count = 0
    step_count = 0

    # ══════════════════════════════════════════════════════════
    # 3.  MAIN LOOP — pure NumPy arithmetic per iteration
    # ══════════════════════════════════════════════════════════
    for t in range(1, M):
        dt_t = dt[t]

        # Skip zero-dt (duplicate timestamps)
        if dt_t <= 0:
            # Midprice may change; update PnL for streaming stats
            S_t = S[t]
            new_pnl = cash + q * S_t
            r = new_pnl - pnl
            pnl = new_pnl

            # Update streaming stats for this zero-dt step
            sum_r += r
            sum_r2 += r * r
            neg_mask = r < 0
            sum_r_neg += np.where(neg_mask, r, 0.0)
            sum_r2_neg += np.where(neg_mask, r * r, 0.0)
            count_neg += neg_mask

            running_max_pnl = np.maximum(running_max_pnl, pnl)
            max_drawdown = np.maximum(max_drawdown, running_max_pnl - pnl)
            step_count += 1
            continue

        S_start = S[t - 1]

        # ── A-S quotes: the shared formula, mid clamp included (K5) ──
        depths, _ = avellaneda_stoikov_quotes(
            midprice=np.full(N, S_start),
            inventory=q,
            tau=np.full(N, tau_arr[t - 1]),
            gamma=gamma_paths,
            skew=no_skew,
            sigma=sigma,
            kappa=kappa,
            tick_size=tick_size,
        )

        # ── Fills: the environment's execution model ──────
        if use_linear_approximation:
            probabilities = np.minimum(A * np.exp(-kappa * depths) * dt_t, 1.0)
            fills = fill_model.rng.uniform(size=depths.shape) < probabilities
        else:
            fills = fill_model.get_fills(depths, dt_t) == 1

        # ── Q_MAX enforcement ─────────────────────────────
        filled_bid = fills[:, 0] & (q < Q_MAX)    # suppress buy if at +Q_MAX
        filled_ask = fills[:, 1] & (q > -Q_MAX)   # suppress sell if at -Q_MAX

        # ── Update state: execution at mid -/+ depth, as in LimitOrderModelDynamics ──
        fb = filled_bid.astype(np.float64)
        fa = filled_ask.astype(np.float64)
        q += fb - fa
        cash += (
            fa * (S_start + depths[:, 1].astype(np.float64))
            - fb * (S_start - depths[:, 0].astype(np.float64))
        )

        # ── PnL and streaming stats ──────────────────────
        previous_endpoint_wealth = pnl.copy()
        new_pnl = cash + q * S[t]
        r = new_pnl - pnl
        pnl = new_pnl

        sum_r += r
        sum_r2 += r * r
        neg_mask = r < 0
        sum_r_neg += np.where(neg_mask, r, 0.0)
        sum_r2_neg += np.where(neg_mask, r * r, 0.0)
        count_neg += neg_mask

        running_max_pnl = np.maximum(running_max_pnl, pnl)
        max_drawdown = np.maximum(max_drawdown, running_max_pnl - pnl)

        elapsed_weight += dt_t
        sum_abs_q += np.abs(q) * dt_t
        sum_near_cap += (np.abs(q) >= near_cap_threshold) * dt_t
        # float32 depths times a Python float, exactly as generate_trajectory_stats accumulates
        sum_spread += (depths[:, 0] + depths[:, 1]) * float(dt_t)

        current_time = float(t_cumulative[t])
        while next_clock_time <= current_time + 1e-12:
            sampled_wealth = (
                pnl if abs(next_clock_time - current_time) <= 1e-12
                else previous_endpoint_wealth
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
        step_count += 1

    # ══════════════════════════════════════════════════════════
    # 5.  COMPUTE final metrics from accumulators
    # ══════════════════════════════════════════════════════════
    mean_r = sum_r / step_count
    var_r = sum_r2 / step_count - mean_r ** 2
    std_r = np.sqrt(np.maximum(var_r, 0.0))

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
        fixed_var = fixed_sum_r2 / fixed_count - fixed_mean**2
        fixed_std = np.sqrt(np.maximum(fixed_var, 0.0))
        safe_fixed_negative = np.maximum(fixed_count_neg, 2.0)
        fixed_negative_mean = fixed_sum_r_neg / safe_fixed_negative
        fixed_negative_var = fixed_sum_r2_neg / safe_fixed_negative - fixed_negative_mean**2
        fixed_negative_std = np.sqrt(np.maximum(fixed_negative_var, 0.0))
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
        "terminal_q": q,
        "mean_spread": sum_spread / max(elapsed_weight, 1e-15),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "pnl_to_map": pnl_to_map,
        "mean_abs_q": mean_abs_q,
        "near_cap_fraction": sum_near_cap / max(elapsed_weight, 1e-15),
        "snapshot_sharpe": snapshot_sharpe,
        "snapshot_sortino": snapshot_sortino,
        "fixed_clock_seconds": fixed_clock_seconds,
        "fixed_clock_return_count": fixed_count,
        "n_steps": step_count,
    }


def fast_simulate_summary(
    midprices: np.ndarray,
    dt_array: np.ndarray,
    gamma: float,
    sigma: float,
    kappa: float,
    A: float,
    terminal_time: float,
    tick_size: float = 0.00001,
    Q_MAX: int = 50,
    num_trajectories: int = 1000,
    seed: int | None = None,
    use_linear_approximation: bool = False,
) -> pd.DataFrame:
    """
    Run fast simulation and return a summary DataFrame
    (same format as ``backtest_summary``).
    """
    stats = fast_simulate(
        midprices=midprices, dt_array=dt_array,
        gamma=gamma, sigma=sigma, kappa=kappa, A=A,
        terminal_time=terminal_time, tick_size=tick_size,
        Q_MAX=Q_MAX, num_trajectories=num_trajectories,
        seed=seed, use_linear_approximation=use_linear_approximation,
    )

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
    df["Sharpe Snapshot (supplementary)"] = stats["snapshot_sharpe"]
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
