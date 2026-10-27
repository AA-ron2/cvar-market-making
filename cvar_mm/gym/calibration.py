"""Censored first-passage calibration for the A--S replay model.

At each quote depth, the exponential waiting-time MLE is
``lambda_hat = hits / total_exposure_seconds``.  The symmetric A--S curve is
then fitted as ``lambda(delta) = A exp(-kappa * delta)``.  Depth is converted
from ticks to price units exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple, Sequence

import numpy as np

from cvar_mm.gym.data_loader import (
    read_market_data_csv,
    stable_order_by_exchange_timestamp,
    timestamps_us_to_datetime_and_deltas,
)
from cvar_mm.gym.helpers.fast_rollout import fast_simulate

ExecutionProxy = Literal["midpoint_first_passage", "best_price_reach"]
ExposureStartRule = Literal["all_snapshots", "fixed_stride", "non_overlapping"]


class ASParameters(NamedTuple):
    sigma: float  # price / sqrt(second)
    A: float  # 1 / second at zero depth
    kappa: float  # 1 / price unit


@dataclass(frozen=True)
class FirstPassageCalibrationConfig:
    """Every choice that defines an exposure is explicit and serialisable.

    The defaults are the frozen F-03 calibration of configs/final_pipeline.env (WORK_ORDER item 8). The
    pipeline still passes every field from that file (common.calibration_config_from_env).
    """

    tick_size: float = 0.00001
    depth_ticks: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0)
    censor_horizon_seconds: float = 15.0
    execution_proxy: ExecutionProxy = "best_price_reach"
    exposure_start_rule: ExposureStartRule = "fixed_stride"
    exposure_stride_seconds: float = 15.0
    min_hits_per_depth: int = 10
    volatility_window_seconds: float = 600.0

    def __post_init__(self) -> None:
        if self.tick_size <= 0 or self.censor_horizon_seconds <= 0:
            raise ValueError("tick size and censor horizon must be positive")
        if not self.depth_ticks or any(value <= 0 for value in self.depth_ticks):
            raise ValueError("depth_ticks must contain positive values")
        if self.exposure_stride_seconds <= 0 or self.min_hits_per_depth <= 0:
            raise ValueError("stride and minimum hit count must be positive")
        if self.execution_proxy not in {"midpoint_first_passage", "best_price_reach"}:
            raise ValueError(f"Unsupported execution proxy: {self.execution_proxy}")
        if self.exposure_start_rule not in {"all_snapshots", "fixed_stride", "non_overlapping"}:
            raise ValueError(f"Unsupported exposure-start rule: {self.exposure_start_rule}")

    @property
    def depths_price(self) -> np.ndarray:
        return np.asarray(self.depth_ticks, dtype=np.float64) * self.tick_size


@dataclass(frozen=True)
class IntensityCurve:
    side: str
    depth_ticks: np.ndarray
    depth_price: np.ndarray
    hits: np.ndarray
    total_exposure_seconds: np.ndarray
    exposures: np.ndarray
    censored: np.ndarray

    @property
    def intensity_per_second(self) -> np.ndarray:
        return np.divide(
            self.hits,
            self.total_exposure_seconds,
            out=np.zeros_like(self.total_exposure_seconds),
            where=self.total_exposure_seconds > 0,
        )

    @property
    def censoring_fraction(self) -> np.ndarray:
        """Fraction of completed exposures that ended at the censor horizon."""

        return np.divide(
            self.censored,
            self.exposures,
            out=np.full(self.exposures.shape, np.nan, dtype=np.float64),
            where=self.exposures > 0,
        )


@dataclass(frozen=True)
class ExponentialIntensityFit:
    A: float
    kappa: float
    r_squared_log: float
    rmse_log: float
    valid_depths: int


@dataclass(frozen=True)
class CalibrationDiagnostics:
    config: FirstPassageCalibrationConfig
    pooled_curve: IntensityCurve
    bid_curve: IntensityCurve
    ask_curve: IntensityCurve
    pooled_fit: ExponentialIntensityFit
    bid_fit: ExponentialIntensityFit | None
    ask_fit: ExponentialIntensityFit | None
    A_dt_quantiles: dict[str, float]
    A_dt_fraction_above: dict[str, float]


@dataclass(frozen=True)
class CalibrationResult:
    sigma: float
    A: float
    kappa: float
    diagnostics: CalibrationDiagnostics

    @property
    def parameters(self) -> ASParameters:
        return ASParameters(self.sigma, self.A, self.kappa)


def _validate_arrays(midprices, dt_array, best_bid, best_ask, config):
    mid = np.asarray(midprices, dtype=np.float64)
    dt = np.asarray(dt_array, dtype=np.float64)
    if mid.ndim != 1 or dt.ndim != 1 or len(mid) != len(dt) or len(mid) < 2:
        raise ValueError("midprices and dt_array must be aligned one-dimensional arrays")
    if np.any(~np.isfinite(mid)) or np.any(~np.isfinite(dt)) or np.any(dt < 0):
        raise ValueError("midprices/dt_array must be finite and dt non-negative")
    bid = None if best_bid is None else np.asarray(best_bid, dtype=np.float64)
    ask = None if best_ask is None else np.asarray(best_ask, dtype=np.float64)
    if config.execution_proxy == "best_price_reach":
        if bid is None or ask is None or bid.shape != mid.shape or ask.shape != mid.shape:
            raise ValueError("best_price_reach requires aligned best_bid and best_ask")
    return mid, dt, bid, ask


def _sigma_from_day(mid: np.ndarray, dt: np.ndarray, window_seconds: float) -> float:
    positive = dt[1:][dt[1:] > 0]
    if positive.size == 0:
        raise ValueError("dt_array must contain a positive timestep")
    window = max(1, int(window_seconds / np.median(positive)))
    changes = np.diff(mid)
    estimates: list[float] = []
    for start in range(0, len(changes), window):
        stop = min(start + window, len(changes))
        elapsed = float(np.sum(dt[start + 1 : stop + 1]))
        if elapsed > 0:
            estimates.append(float(np.sqrt(np.sum(changes[start:stop] ** 2) / elapsed)))
    if not estimates:
        raise ValueError("No positive-time volatility window is available")
    return float(np.median(estimates))


def _starts(times: np.ndarray, config: FirstPassageCalibrationConfig) -> np.ndarray:
    if config.exposure_start_rule == "all_snapshots":
        return np.arange(len(times) - 1, dtype=np.int64)
    targets = np.arange(times[0], times[-1], config.exposure_stride_seconds)
    indices = np.searchsorted(times, targets, side="left")
    return np.unique(indices[indices < len(times) - 1])


def _first_hit(values, start, stop, threshold, up):
    future = values[start + 1 : stop + 1]
    locations = np.flatnonzero(future >= threshold if up else future <= threshold)
    return None if locations.size == 0 else int(start + 1 + locations[0])


def _observe(values, mid, times, start, depth, side, config):
    censor_time = min(times[start] + config.censor_horizon_seconds, times[-1])
    if censor_time <= times[start]:
        return None
    stop = int(np.searchsorted(times, censor_time, side="right") - 1)
    threshold = mid[start] + depth if side == "ask" else mid[start] - depth
    hit = _first_hit(values, start, stop, threshold, side == "ask")
    end_time = times[hit] if hit is not None else censor_time
    return hit, float(end_time - times[start]), stop


def _collect_side(mid, times, values, config, side):
    depths = config.depths_price
    hits = np.zeros(len(depths), dtype=np.int64)
    exposure = np.zeros(len(depths), dtype=np.float64)
    counts = np.zeros(len(depths), dtype=np.int64)
    censored = np.zeros(len(depths), dtype=np.int64)
    common = None if config.exposure_start_rule == "non_overlapping" else _starts(times, config)
    for depth_index, depth in enumerate(depths):
        if common is None:
            selected: list[int] = []
            start = 0
            while start < len(mid) - 1:
                selected.append(start)
                observed = _observe(values, mid, times, start, depth, side, config)
                if observed is None:
                    break
                hit, _, stop = observed
                start = hit + 1 if hit is not None else max(stop + 1, start + 1)
            start_indices: Sequence[int] = selected
        else:
            start_indices = common
        for start in start_indices:
            observed = _observe(values, mid, times, int(start), depth, side, config)
            if observed is None:
                continue
            hit, elapsed, _ = observed
            counts[depth_index] += 1
            exposure[depth_index] += elapsed
            if hit is None:
                censored[depth_index] += 1
            else:
                hits[depth_index] += 1
    return IntensityCurve(
        side=side,
        depth_ticks=np.asarray(config.depth_ticks),
        depth_price=depths.copy(),
        hits=hits,
        total_exposure_seconds=exposure,
        exposures=counts,
        censored=censored,
    )


def collect_censored_first_passage(
    midprices: np.ndarray,
    dt_array: np.ndarray,
    *,
    config: FirstPassageCalibrationConfig,
    best_bid: np.ndarray | None = None,
    best_ask: np.ndarray | None = None,
) -> tuple[IntensityCurve, IntensityCurve]:
    """Collect bid/ask hit counts and exposure times, including censoring."""

    mid, dt, bid, ask = _validate_arrays(midprices, dt_array, best_bid, best_ask, config)
    times = np.cumsum(dt)
    bid_values, ask_values = (mid, mid) if config.execution_proxy == "midpoint_first_passage" else (bid, ask)
    return (
        _collect_side(mid, times, bid_values, config, "bid"),
        _collect_side(mid, times, ask_values, config, "ask"),
    )


def sum_intensity_curves(curves: Sequence[IntensityCurve], side: str) -> IntensityCurve:
    """Pool aligned curves by summing sufficient statistics."""

    if not curves:
        raise ValueError("At least one curve is required")
    first = curves[0]
    if any(not np.array_equal(curve.depth_price, first.depth_price) for curve in curves[1:]):
        raise ValueError("Cannot pool different depth grids")
    return IntensityCurve(
        side=side,
        depth_ticks=first.depth_ticks.copy(),
        depth_price=first.depth_price.copy(),
        hits=np.sum([curve.hits for curve in curves], axis=0),
        total_exposure_seconds=np.sum([curve.total_exposure_seconds for curve in curves], axis=0),
        exposures=np.sum([curve.exposures for curve in curves], axis=0),
        censored=np.sum([curve.censored for curve in curves], axis=0),
    )


def valid_fit_depth_mask(curve: IntensityCurve, min_hits_per_depth: int) -> np.ndarray:
    """Return the exact depth mask used by the hit-weighted log fit."""

    intensity = curve.intensity_per_second
    return (
        (curve.hits >= min_hits_per_depth)
        & (curve.total_exposure_seconds > 0)
        & np.isfinite(intensity)
        & (intensity > 0)
    )


def empirical_monotonicity_fraction(
    curve: IntensityCurve,
    *,
    min_hits_per_depth: int,
) -> float:
    """Share of adjacent valid depths with non-increasing empirical intensity."""

    valid_intensity = curve.intensity_per_second[
        valid_fit_depth_mask(curve, min_hits_per_depth)
    ]
    if valid_intensity.size < 2:
        return float("nan")
    return float(np.mean(np.diff(valid_intensity) <= 0.0))


def contiguous_supported_depth_prefix(
    curve: IntensityCurve,
    *,
    min_hits_per_depth: int,
) -> np.ndarray:
    """Largest leading part of the configured grid supported by the fit rules."""

    valid = valid_fit_depth_mask(curve, min_hits_per_depth)
    invalid = np.flatnonzero(~valid)
    stop = len(valid) if invalid.size == 0 else int(invalid[0])
    return curve.depth_ticks[:stop].copy()


def fit_exponential_intensity(curve, *, min_hits_per_depth):
    """Hit-weighted log-WLS fit requiring a strictly decreasing fitted curve."""

    intensity = curve.intensity_per_second
    valid = valid_fit_depth_mask(curve, min_hits_per_depth)
    if int(valid.sum()) < 2:
        raise ValueError(f"Only {int(valid.sum())} depths meet the minimum-hit requirement")
    x, y = curve.depth_price[valid], np.log(intensity[valid])
    weights = np.sqrt(curve.hits[valid].astype(np.float64))
    slope, intercept = np.polyfit(x, y, 1, w=weights)
    if not np.isfinite(slope + intercept) or slope >= 0:
        raise ValueError("Fitted intensity must be finite, positive, and strictly decreasing")
    fitted = intercept + slope * x
    mean = float(np.average(y, weights=weights**2))
    ss_res = float(np.sum(weights**2 * (y - fitted) ** 2))
    ss_tot = float(np.sum(weights**2 * (y - mean) ** 2))
    return ExponentialIntensityFit(
        A=float(np.exp(intercept)),
        kappa=float(-slope),
        r_squared_log=1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
        rmse_log=float(np.sqrt(ss_res / np.sum(weights**2))),
        valid_depths=int(valid.sum()),
    )


def _optional_fit(curve, minimum):
    try:
        return fit_exponential_intensity(curve, min_hits_per_depth=minimum)
    except ValueError:
        return None


def _a_dt(A, daily_dt_arrays):
    values = np.concatenate([np.asarray(dt, dtype=np.float64) for dt in daily_dt_arrays]) * A
    values = values[np.isfinite(values) & (values >= 0)]
    if not values.size:
        raise ValueError("No finite A*dt observations")
    quantiles = {name: float(np.quantile(values, q)) for name, q in (
        ("q50", .50), ("q90", .90), ("q95", .95), ("q99", .99), ("max", 1.0)
    )}
    fractions = {str(level): float(np.mean(values > level)) for level in (.01, .1, 1.0)}
    return quantiles, fractions


def calibrate_from_days(
    daily_midprices: Sequence[np.ndarray],
    daily_dt_arrays: Sequence[np.ndarray],
    *,
    config: FirstPassageCalibrationConfig,
    daily_best_bid: Sequence[np.ndarray] | None = None,
    daily_best_ask: Sequence[np.ndarray] | None = None,
) -> CalibrationResult:
    """Fit one pooled symmetric curve while retaining side diagnostics."""

    if not daily_midprices or len(daily_midprices) != len(daily_dt_arrays):
        raise ValueError("daily arrays must be non-empty and aligned")
    if daily_best_bid is not None and len(daily_best_bid) != len(daily_midprices):
        raise ValueError("daily_best_bid is not aligned")
    if daily_best_ask is not None and len(daily_best_ask) != len(daily_midprices):
        raise ValueError("daily_best_ask is not aligned")
    bids, asks, sigmas = [], [], []
    for index, (mid, dt) in enumerate(zip(daily_midprices, daily_dt_arrays)):
        bid_curve, ask_curve = collect_censored_first_passage(
            mid,
            dt,
            config=config,
            best_bid=None if daily_best_bid is None else daily_best_bid[index],
            best_ask=None if daily_best_ask is None else daily_best_ask[index],
        )
        bids.append(bid_curve)
        asks.append(ask_curve)
        sigmas.append(_sigma_from_day(np.asarray(mid), np.asarray(dt), config.volatility_window_seconds))
    bid_curve = sum_intensity_curves(bids, "bid")
    ask_curve = sum_intensity_curves(asks, "ask")
    pooled = sum_intensity_curves([bid_curve, ask_curve], "pooled")
    fit = fit_exponential_intensity(pooled, min_hits_per_depth=config.min_hits_per_depth)
    quantiles, fractions = _a_dt(fit.A, daily_dt_arrays)
    diagnostics = CalibrationDiagnostics(
        config=config,
        pooled_curve=pooled,
        bid_curve=bid_curve,
        ask_curve=ask_curve,
        pooled_fit=fit,
        bid_fit=_optional_fit(bid_curve, config.min_hits_per_depth),
        ask_fit=_optional_fit(ask_curve, config.min_hits_per_depth),
        A_dt_quantiles=quantiles,
        A_dt_fraction_above=fractions,
    )
    return CalibrationResult(float(np.median(sigmas)), fit.A, fit.kappa, diagnostics)


# The pre-F-03 exposure settings these compatibility wrappers always used (the old dataclass defaults), passed
# explicitly so that F-03 defaults change none of their results; the pipeline uses neither wrapper.
LEGACY_EXPOSURE_SETTINGS = {
    "execution_proxy": "midpoint_first_passage",
    "exposure_start_rule": "non_overlapping",
    "exposure_stride_seconds": 1.0,
}


def calibrate_from_arrays(
    midprices: np.ndarray,
    dt_array: np.ndarray,
    tick_size: float = 0.00001,
    n_depth_ticks: int = 20,
    min_arrivals: int = 10,
    *,
    best_bid: np.ndarray | None = None,
    best_ask: np.ndarray | None = None,
    config: FirstPassageCalibrationConfig | None = None,
    return_diagnostics: bool = False,
):
    """One-day compatibility entry point for the censored estimator."""

    if config is None:
        config = FirstPassageCalibrationConfig(
            tick_size=tick_size,
            depth_ticks=tuple(np.arange(0.5, n_depth_ticks + 0.5, 1.0)),
            min_hits_per_depth=min_arrivals,
            **LEGACY_EXPOSURE_SETTINGS,
        )
    result = calibrate_from_days(
        [midprices], [dt_array], config=config,
        daily_best_bid=None if best_bid is None else [best_bid],
        daily_best_ask=None if best_ask is None else [best_ask],
    )
    return result if return_diagnostics else (result.sigma, result.A, result.kappa)


def calibrate_as_parameters(
    filepath: str,
    tick_size: float = 0.00001,
    fit_depth_max: float | None = None,
    volatility_window_sec: float = 600.0,
    min_arrivals: int = 20,
    plot: bool = True,
    verbose: bool = True,
) -> ASParameters:
    """File-based compatibility wrapper; diagnostics are produced elsewhere."""

    del plot
    frame = read_market_data_csv(filepath)
    frame = stable_order_by_exchange_timestamp(frame)
    _, dt = timestamps_us_to_datetime_and_deltas(frame.index)
    ask = frame["asks[0].price"].to_numpy(dtype=np.float64)
    bid = frame["bids[0].price"].to_numpy(dtype=np.float64)
    mid = (ask + bid) / 2.0
    max_ticks = 20 if fit_depth_max is None else max(2, int(fit_depth_max / tick_size))
    config = FirstPassageCalibrationConfig(
        tick_size=tick_size,
        depth_ticks=tuple(np.arange(0.5, max_ticks + 0.5, 1.0)),
        min_hits_per_depth=min_arrivals,
        volatility_window_seconds=volatility_window_sec,
        **LEGACY_EXPOSURE_SETTINGS,
    )
    result = calibrate_from_arrays(mid, dt, best_bid=bid, best_ask=ask, config=config, return_diagnostics=True)
    if verbose:
        print(f"sigma={result.sigma:.8f}, A={result.A:.6f}, kappa={result.kappa:.2f}")
    return result.parameters


def tune_gamma(
    midprices: np.ndarray,
    dt_array: np.ndarray,
    sigma: float,
    kappa: float,
    A: float,
    tick_size: float = 0.00001,
    Q_MAX: int = 10,
    gamma_range: tuple[float, float] = (0.001, 1.0),
    n_trials: int = 100,
    num_trajectories: int = 50,
    seed: int = 42,
    metric: str = "sharpe",
    verbose: bool = True,
):
    """Tune gamma using training data only.

    ``seed`` fixes both the fill draws and the TPE sampler, so the tuned gamma
    is reproducible (K9).
    """

    try:
        import optuna
    except ImportError as exc:
        raise ImportError("optuna required. pip install optuna") from exc
    terminal_time = float(dt_array.sum())

    def objective(trial):
        gamma = trial.suggest_float("gamma", gamma_range[0], gamma_range[1], log=True)
        stats = fast_simulate(
            midprices=midprices, dt_array=dt_array, gamma=gamma, sigma=sigma,
            kappa=kappa, A=A, terminal_time=terminal_time, tick_size=tick_size,
            Q_MAX=Q_MAX, num_trajectories=num_trajectories, seed=seed,
            use_linear_approximation=False,
        )
        return float(stats[metric].mean())

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best_gamma = study.best_params["gamma"]
    if verbose:
        print(f"Best gamma={best_gamma:.6f}; best {metric}={study.best_value:.6f}")
    return best_gamma, study
