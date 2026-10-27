from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # pragma: no cover - depends on runtime environment
    import gymnasium
    from gymnasium.spaces import Box
    _GYMNASIUM_AVAILABLE = True
except Exception:  # pragma: no cover - imported in lightweight/static checks
    gymnasium = None
    Box = None
    _GYMNASIUM_AVAILABLE = False

from cvar_mm.gym.index_names import (
    ASSET_PRICE_INDEX,
    INVENTORY_INDEX,
    TIME_INDEX,
)


FORMULA_FEATURE_NAMES = (
    "inventory_norm",
    "time_remaining_frac",
    "log_mid_rel",
    "rolling_vol_ticks",
    "momentum_ticks",
    "spread_ticks",
    "lob_imbalance",
)


@dataclass(frozen=True)
class FormulaASActionConfig:
    """Bounds and market parameters for formula-based A-S action mapping."""

    sigma: float
    kappa: float
    tick_size: float
    gamma_min: float = 0.01
    gamma_max: float = 0.9
    skew_ticks_max: float = 5.0

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive.")
        if self.tick_size <= 0:
            raise ValueError("tick_size must be positive.")
        if self.gamma_min <= 0 or self.gamma_max <= 0:
            raise ValueError("gamma bounds must be positive.")
        if self.gamma_min >= self.gamma_max:
            raise ValueError("gamma_min must be smaller than gamma_max.")
        if self.skew_ticks_max < 0:
            raise ValueError("skew_ticks_max must be non-negative.")


def map_formula_action(
    action: np.ndarray,
    config: FormulaASActionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map PPO actions in [-1, 1]^2 to A-S parameters.

    The gamma mapping is log-linear so equal action movement corresponds to a
    multiplicative change in risk aversion. Skew is additive in ticks, which is
    safer for low-priced crypto assets than percentage price skew.
    """
    action = np.asarray(action, dtype=np.float64)
    if action.ndim == 1:
        action = action.reshape(1, -1)
    if action.shape[1] != 2:
        raise ValueError(f"Expected action shape (N, 2), got {action.shape}.")

    clipped = np.clip(action, -1.0, 1.0)
    gamma_u = 0.5 * (clipped[:, 0] + 1.0)
    log_min = np.log(config.gamma_min)
    log_max = np.log(config.gamma_max)
    gamma = np.exp(log_min + gamma_u * (log_max - log_min))
    skew = clipped[:, 1] * config.skew_ticks_max * config.tick_size
    return gamma, skew


def compute_formula_as_depths(
    *,
    midprice: np.ndarray,
    inventory: np.ndarray,
    time_elapsed: np.ndarray,
    terminal_time: float,
    action: np.ndarray,
    config: FormulaASActionConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Convert normalized parameter actions to executable bid/ask depths."""
    mid = np.asarray(midprice, dtype=np.float64).reshape(-1)
    q = np.asarray(inventory, dtype=np.float64).reshape(-1)
    t = np.asarray(time_elapsed, dtype=np.float64).reshape(-1)
    gamma, skew = map_formula_action(action, config)

    tau = np.maximum(float(terminal_time) - t, 0.0)
    return avellaneda_stoikov_quotes(
        midprice=mid,
        inventory=q,
        tau=tau,
        gamma=gamma,
        skew=skew,
        sigma=config.sigma,
        kappa=config.kappa,
        tick_size=config.tick_size,
    )


def avellaneda_stoikov_quotes(
    *,
    midprice: np.ndarray,
    inventory: np.ndarray,
    tau: np.ndarray,
    gamma: np.ndarray,
    skew: np.ndarray,
    sigma: float,
    kappa: float,
    tick_size: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """A-S quotes on the tick grid with the mid clamp, shared by B0 and B1-B3 (K5).

    r = S - q gamma sigma^2 tau and h = gamma sigma^2 tau / 2 + ln(1 + gamma/kappa) / gamma;
    bid = floor((r + skew - h) / tick) tick and ask = ceil((r + skew + h) / tick) tick,
    then bid <= S - tick/2 and ask >= S + tick/2.  Returns the depths in the
    environment's float32 action dtype and float64 diagnostics.
    """
    mid = midprice
    reservation = mid - inventory * gamma * sigma**2 * tau
    half_spread = (
        0.5 * gamma * sigma**2 * tau
        + (1.0 / gamma) * np.log1p(gamma / kappa)
    )
    center = reservation + skew

    bid = np.floor((center - half_spread) / tick_size) * tick_size
    ask = np.ceil((center + half_spread) / tick_size) * tick_size

    min_delta = 0.5 * tick_size
    bid = np.minimum(bid, mid - min_delta)
    ask = np.maximum(ask, mid + min_delta)
    crossed = ask <= bid
    ask = np.where(crossed, bid + tick_size, ask)

    delta_bid = np.maximum(mid - bid, min_delta)
    delta_ask = np.maximum(ask - mid, min_delta)
    depths = np.column_stack([delta_bid, delta_ask]).astype(np.float32)
    diagnostics = {
        "gamma": gamma,
        "skew": skew,
        "reservation": reservation,
        "half_spread": half_spread,
        "bid": bid,
        "ask": ask,
        "delta_bid": delta_bid,
        "delta_ask": delta_ask,
    }
    return depths, diagnostics


def extract_market_feature_arrays(
    data,
    *,
    tick_size: float,
    imbalance_depth: int = 5,
) -> dict[str, np.ndarray]:
    """Extract online L1/L2 features from a Tardis snapshot DataFrame."""
    ask = data["asks[0].price"].to_numpy(dtype=np.float64)
    bid = data["bids[0].price"].to_numpy(dtype=np.float64)
    spread_ticks = (ask - bid) / max(float(tick_size), 1e-12)

    bid_amounts = []
    ask_amounts = []
    for level in range(imbalance_depth):
        bid_col = f"bids[{level}].amount"
        ask_col = f"asks[{level}].amount"
        if bid_col in data.columns and ask_col in data.columns:
            bid_amounts.append(data[bid_col].to_numpy(dtype=np.float64))
            ask_amounts.append(data[ask_col].to_numpy(dtype=np.float64))

    if bid_amounts and ask_amounts:
        bid_qty = np.sum(np.vstack(bid_amounts), axis=0)
        ask_qty = np.sum(np.vstack(ask_amounts), axis=0)
        lob_imbalance = (bid_qty - ask_qty) / np.maximum(bid_qty + ask_qty, 1e-12)
    else:
        lob_imbalance = np.zeros_like(spread_ticks)

    return {
        "spread_ticks": spread_ticks.astype(np.float32),
        "lob_imbalance": lob_imbalance.astype(np.float32),
    }


class FormulaASActionWrapper(gymnasium.Env if _GYMNASIUM_AVAILABLE else object):
    """
    Expose A-S parameter actions while delegating fills/rewards to the base env.

    The wrapped environment still executes bid/ask depths internally. PPO sees
    engineered no-leakage features and chooses normalized ``[gamma, skew]``
    controls; this wrapper computes the A-S quote formula and forwards depths.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        base_env,
        *,
        action_config: FormulaASActionConfig,
        market_features: dict[str, np.ndarray] | None = None,
        daily_market_features: list[dict[str, np.ndarray]] | None = None,
        rolling_window: int = 100,
        enabled_features: tuple[str, ...] | list[str] | None = None,
    ):
        if not _GYMNASIUM_AVAILABLE:
            raise ImportError("gymnasium is required to construct FormulaASActionWrapper.")
        super().__init__()
        self.base_env = base_env
        self.action_config = action_config
        self.market_features = market_features
        self.daily_market_features = daily_market_features
        self.rolling_window = int(max(2, rolling_window))
        self.enabled_features = tuple(enabled_features or FORMULA_FEATURE_NAMES)
        unknown = set(self.enabled_features) - set(FORMULA_FEATURE_NAMES)
        if unknown:
            raise ValueError(f"Unknown formula feature(s): {sorted(unknown)}")
        self._feature_indices = [FORMULA_FEATURE_NAMES.index(name) for name in self.enabled_features]

        self.action_space = Box(
            low=-np.ones(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.enabled_features),),
            dtype=np.float32,
        )

        self._dS: deque[float] = deque(maxlen=self.rolling_window)
        self._dt: deque[float] = deque(maxlen=self.rolling_window)
        self._last_mid = 0.0
        self._initial_mid = 1.0
        self.last_depth_action: np.ndarray | None = None
        self.last_formula_diagnostics: dict[str, np.ndarray] = {}

    def seed(self, seed: int | None = None) -> None:
        self.base_env.seed(seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        _, info = self.base_env.reset(seed=seed, options=options)
        state = self.metric_state
        self._dS = deque(maxlen=self.rolling_window)
        self._dt = deque(maxlen=self.rolling_window)
        self._last_mid = float(state[0, ASSET_PRICE_INDEX])
        self._initial_mid = max(abs(self._last_mid), 1e-12)
        self.last_depth_action = np.zeros((self.num_trajectories, 2), dtype=np.float32)
        self.last_formula_diagnostics = {}
        return self._obs(), info

    def step(self, action: np.ndarray):
        state = self.metric_state
        depths, diagnostics = compute_formula_as_depths(
            midprice=state[:, ASSET_PRICE_INDEX],
            inventory=state[:, INVENTORY_INDEX],
            time_elapsed=state[:, TIME_INDEX],
            terminal_time=self.terminal_time,
            action=action,
            config=self.action_config,
        )
        self.last_depth_action = depths
        self.last_formula_diagnostics = diagnostics

        obs, reward, terminated, truncated, info = self.base_env.step(depths)
        self._update_rolling_features()
        if isinstance(info, dict):
            info = info.copy()
            info["formula_action"] = diagnostics
            info["depth_action"] = depths
        return self._obs(), reward, terminated, truncated, info

    def _update_rolling_features(self) -> None:
        state = self.metric_state
        mid = float(state[0, ASSET_PRICE_INDEX])
        step_index = self.base_env._midprice_model.step_index
        dt_array = self.base_env._midprice_model.dt_array
        dt = float(dt_array[step_index]) if 0 <= step_index < len(dt_array) else 0.0
        self._dS.append(mid - self._last_mid)
        self._dt.append(max(dt, 1e-10))
        self._last_mid = mid

    def _active_market_features(self) -> dict[str, np.ndarray] | None:
        if self.daily_market_features is not None:
            day_index = getattr(self.base_env._midprice_model, "day_index", 0)
            if 0 <= day_index < len(self.daily_market_features):
                return self.daily_market_features[day_index]
        return self.market_features

    def _feature_at_step(self, name: str, default: float = 0.0) -> float:
        features = self._active_market_features()
        if not features or name not in features:
            return default
        values = features[name]
        step_index = self.base_env._midprice_model.step_index
        if step_index < 0 or step_index >= len(values):
            return default
        return float(values[step_index])

    def _obs(self) -> np.ndarray:
        state = self.metric_state
        q = state[:, INVENTORY_INDEX]
        t = state[:, TIME_INDEX]
        mid = state[:, ASSET_PRICE_INDEX]
        terminal_time = max(float(self.terminal_time), 1e-12)

        if len(self._dS) >= 2:
            dt_sum = max(float(np.sum(self._dt)), 1e-12)
            rolling_vol = np.sqrt(float(np.sum(np.square(self._dS))) / dt_sum)
            momentum = float(np.sum(self._dS)) / dt_sum
        else:
            rolling_vol = 0.0
            momentum = 0.0

        shared = np.array([
            np.clip((terminal_time - float(t[0])) / terminal_time, 0.0, 1.0),
            np.log(max(float(mid[0]), 1e-12) / self._initial_mid),
            rolling_vol / self.action_config.tick_size,
            momentum / self.action_config.tick_size,
            self._feature_at_step("spread_ticks", 0.0),
            self._feature_at_step("lob_imbalance", 0.0),
        ], dtype=np.float32)

        obs = np.zeros((self.num_trajectories, len(FORMULA_FEATURE_NAMES)), dtype=np.float32)
        obs[:, 0] = np.clip(q / max(float(self.max_inventory), 1.0), -1.0, 1.0)
        obs[:, 1:] = shared.reshape(1, -1)
        return obs[:, self._feature_indices]

    @property
    def metric_state(self) -> np.ndarray:
        return self.base_env.model_dynamics.state.copy()

    @property
    def model_dynamics(self):
        return self.base_env.model_dynamics

    @property
    def num_trajectories(self) -> int:
        return self.base_env.num_trajectories

    @property
    def n_steps(self) -> int:
        return self.base_env.n_steps

    @property
    def terminal_time(self) -> float:
        return self.base_env.terminal_time

    @property
    def max_inventory(self) -> int:
        return self.base_env.max_inventory

    @property
    def normalise_observation_space_(self) -> bool:
        return False

    @property
    def normalise_action_space_(self) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_env, name)


class FormulaASPolicyAdapter:
    """
    Minimal adapter for offline parity checks and later Hummingbot integration.

    Hummingbot should reconstruct the same formula observation vector, then use
    this adapter to convert model output into executable quote prices.
    """

    def __init__(
        self,
        model,
        *,
        action_config: FormulaASActionConfig,
        terminal_time: float,
        vecnorm_env=None,
        deterministic: bool = True,
    ):
        self.model = model
        self.action_config = action_config
        self.terminal_time = float(terminal_time)
        self.vecnorm_env = vecnorm_env
        self.deterministic = deterministic

    def predict_quote(
        self,
        *,
        formula_observation: np.ndarray,
        raw_state: np.ndarray,
    ) -> dict[str, float]:
        obs = np.asarray(formula_observation, dtype=np.float32).reshape(1, -1)
        if self.vecnorm_env is not None:
            obs = self.vecnorm_env.normalize_obs(obs)
        action, _ = self.model.predict(obs[0], deterministic=self.deterministic)
        state = np.asarray(raw_state, dtype=np.float64).reshape(1, -1)
        depths, diagnostics = compute_formula_as_depths(
            midprice=state[:, ASSET_PRICE_INDEX],
            inventory=state[:, INVENTORY_INDEX],
            time_elapsed=state[:, TIME_INDEX],
            terminal_time=self.terminal_time,
            action=action,
            config=self.action_config,
        )
        return {
            "bid_price": float(diagnostics["bid"][0]),
            "ask_price": float(diagnostics["ask"][0]),
            "delta_bid": float(depths[0, 0]),
            "delta_ask": float(depths[0, 1]),
            "gamma": float(diagnostics["gamma"][0]),
            "skew": float(diagnostics["skew"][0]),
        }
