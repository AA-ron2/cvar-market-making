"""Helper exports with the numerical fast path importable without Gymnasium."""

from importlib.util import find_spec

from cvar_mm.gym.helpers.fast_rollout import fast_simulate, fast_simulate_summary

if find_spec("gymnasium") is not None:
    from cvar_mm.gym.helpers.generate_trajectory_stats import (
        generate_trajectory_stats,
        stats_to_summary,
        stats_to_results_table,
    )
