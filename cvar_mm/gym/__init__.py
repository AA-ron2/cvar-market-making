"""Gym integrations and dependency-light numerical helpers.

Importing a pure submodule must not require Gymnasium merely because Python
initializes this package first.
"""

from importlib.util import find_spec

from cvar_mm.gym.experiment_config import BMExperimentConfig, ReplayExperimentConfig, find_repo_root
from cvar_mm.gym.formula_as import (
    FORMULA_FEATURE_NAMES,
    FormulaASActionConfig,
    FormulaASActionWrapper,
    FormulaASPolicyAdapter,
    compute_formula_as_depths,
    extract_market_feature_arrays,
    map_formula_action,
)

if find_spec("gymnasium") is not None:
    from cvar_mm.gym.trading_environment import TradingEnvironment
    from cvar_mm.gym.model_dynamics import ModelDynamics, LimitOrderModelDynamics
    from cvar_mm.gym.helpers.generate_trajectory_stats import (
        generate_trajectory_stats,
        stats_to_summary,
        stats_to_results_table,
    )
    from cvar_mm.gym.notebook_support import (
        build_formula_multi_day_replay_env,
        build_formula_replay_env,
        build_multi_day_replay_env,
        build_replay_env,
        build_replay_feature_computer,
        evaluate_agent_over_seeds,
        evaluate_formula_rl_per_day,
        evaluate_as_fast,
        freeze_vecnorm,
        make_vecnorm,
        run_qmax_sensitivity,
        stats_dict_to_frame,
        summarise_agent_frames,
    )

    # SB3 wrapper is only available when stable-baselines3 is installed.
    try:
        from cvar_mm.gym.sb3_wrapper import StableBaselinesTradingEnvironment
    except ImportError:
        pass
