"""
Data loader for multi-day Tardis L2 order book snapshots.

Usage::

    from cvar_mm.gym.data_loader import load_multi_day, load_single_day
    daily_S, daily_dt, dates = load_multi_day(data_dir, pair="DOGEUSDT")
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import glob
import os
import warnings

import numpy as np
import pandas as pd

from cvar_mm.gym.formula_as import extract_market_feature_arrays


def read_market_data_csv(filepath: str) -> pd.DataFrame:
    """Keep exchange timestamps as exact text until timestamp validation."""
    return pd.read_csv(filepath, index_col="timestamp", dtype={"timestamp": "string"})


def _validated_timestamps_us(raw_timestamps: pd.Index) -> np.ndarray:
    """Return validated raw Unix-microsecond timestamps as signed int64."""
    raw_values = np.asarray(raw_timestamps)
    if raw_values.ndim != 1 or raw_values.size == 0:
        raise ValueError("timestamps must be a non-empty one-dimensional sequence")

    if raw_values.dtype.kind in {"O", "U", "S"}:
        int64_info = np.iinfo(np.int64)
        exact_timestamps: list[int] = []
        for value in raw_values:
            if not isinstance(value, (int, np.integer, str)):
                break
            try:
                exact_value = Decimal(str(value))
            except InvalidOperation as exc:
                raise ValueError(
                    "timestamps must be numeric integer microseconds since Unix epoch"
                ) from exc
            if not exact_value.is_finite() or exact_value != exact_value.to_integral_value():
                raise ValueError(
                    "timestamps must be finite integer microseconds since Unix epoch"
                )
            if not int64_info.min <= exact_value <= int64_info.max:
                raise ValueError("timestamps exceed the supported int64 microsecond range")
            exact_timestamps.append(int(exact_value))
        else:
            try:
                pd.to_numeric(raw_timestamps, errors="raise")
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "timestamps must be numeric integer microseconds since Unix epoch"
                ) from exc
            return np.asarray(exact_timestamps, dtype=np.int64)

    try:
        numeric_timestamps = pd.to_numeric(raw_timestamps, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "timestamps must be numeric integer microseconds since Unix epoch"
        ) from exc

    values = np.asarray(numeric_timestamps)
    if np.issubdtype(values.dtype, np.integer):
        if np.issubdtype(values.dtype, np.unsignedinteger) and (
            int(values.max()) > np.iinfo(np.int64).max
        ):
            raise ValueError("timestamps exceed the supported int64 microsecond range")
        timestamps_us = values.astype(np.int64, copy=False)
    else:
        try:
            float_values = values.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "timestamps must be numeric integer microseconds since Unix epoch"
            ) from exc
        if np.any(~np.isfinite(float_values)) or np.any(
            float_values != np.trunc(float_values)
        ):
            raise ValueError(
                "timestamps must be finite integer microseconds since Unix epoch"
            )
        int64_info = np.iinfo(np.int64)
        if np.any(float_values < int64_info.min) or np.any(
            float_values >= float(int64_info.max)
        ):
            raise ValueError("timestamps exceed the supported int64 microsecond range")
        timestamps_us = float_values.astype(np.int64)
    return timestamps_us


def stable_order_by_exchange_timestamp(data: pd.DataFrame) -> pd.DataFrame:
    """Stably order complete rows by the raw exchange timestamp index if needed."""

    timestamps_us = _validated_timestamps_us(data.index)
    inversions = timestamps_us[1:] < timestamps_us[:-1]
    inversion_count = int(np.count_nonzero(inversions))
    if inversion_count == 0:
        return data

    backward_steps = [
        int(previous) - int(current)
        for previous, current in zip(
            timestamps_us[:-1][inversions],
            timestamps_us[1:][inversions],
        )
    ]
    max_backward_us = max(backward_steps)
    transition_label = "transition" if inversion_count == 1 else "transitions"
    warnings.warn(
        f"Detected {inversion_count} out-of-order exchange timestamp "
        f"{transition_label} (max backward step {max_backward_us} us); "
        "stably sorting by exchange timestamp.",
        RuntimeWarning,
        stacklevel=2,
    )
    order = np.argsort(timestamps_us, kind="stable")
    return data.iloc[order]


def timestamps_us_to_datetime_and_deltas(
    raw_timestamps: pd.Index,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Convert raw Unix microseconds to a datetime index and elapsed seconds."""

    timestamps_us = _validated_timestamps_us(raw_timestamps)

    if np.any(timestamps_us[1:] < timestamps_us[:-1]):
        raise ValueError("timestamps must be non-decreasing; input is not sorted")
    delta_us = np.diff(timestamps_us, prepend=timestamps_us[0])
    if np.any(delta_us < 0):
        raise ValueError("timestamp deltas exceed the supported int64 range")

    dt_seconds = delta_us.astype(np.float64) / 1_000_000.0
    dt_seconds[0] = 0.0
    try:
        datetime_index = pd.DatetimeIndex(pd.to_datetime(timestamps_us, unit="us"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("timestamps are outside the supported datetime range") from exc
    return datetime_index, dt_seconds


def load_single_day(filepath: str) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Load one Tardis CSV → (midprices, dt_array, datetime_index)."""
    data = read_market_data_csv(filepath)
    data = stable_order_by_exchange_timestamp(data)
    data.index, dt_sec = timestamps_us_to_datetime_and_deltas(data.index)
    data["midprice"] = (data["asks[0].price"] + data["bids[0].price"]) / 2
    S = data["midprice"].to_numpy()
    return S, dt_sec, data.index


def load_single_day_calibration_arrays(
    filepath: str,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Load midpoint, clock deltas, and L1 prices needed by execution proxies."""

    data = read_market_data_csv(filepath)
    data = stable_order_by_exchange_timestamp(data)
    data.index, dt_seconds = timestamps_us_to_datetime_and_deltas(data.index)
    best_ask = data["asks[0].price"].to_numpy(dtype=np.float64)
    best_bid = data["bids[0].price"].to_numpy(dtype=np.float64)
    midprice = (best_ask + best_bid) / 2.0
    return midprice, dt_seconds, data.index, best_bid, best_ask


def load_single_day_with_features(
    filepath: str,
    *,
    tick_size: float = 0.00001,
    imbalance_depth: int = 5,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, dict[str, np.ndarray]]:
    """Load one Tardis CSV and online market features for formula-AS agents."""
    data = read_market_data_csv(filepath)
    data = stable_order_by_exchange_timestamp(data)
    data.index, dt_sec = timestamps_us_to_datetime_and_deltas(data.index)
    data["midprice"] = (data["asks[0].price"] + data["bids[0].price"]) / 2
    S = data["midprice"].to_numpy()
    features = extract_market_feature_arrays(
        data,
        tick_size=tick_size,
        imbalance_depth=imbalance_depth,
    )
    return S, dt_sec, data.index, features


def load_multi_day(
    data_dir: str,
    pair: str = "DOGEUSDT",
    pattern: str = "binance_book_snapshot_25_{date}_{pair}.csv",
    max_days: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """
    Load all available days from a directory.

    Returns (daily_midprices, daily_dt_arrays, date_strings).
    """
    glob_pat = pattern.replace("{date}", "*").replace("{pair}", pair)
    files = sorted(glob.glob(os.path.join(data_dir, glob_pat)))

    if not files:
        raise FileNotFoundError(
            f"No files matching '{glob_pat}' in '{data_dir}'."
        )
    if max_days is not None:
        files = files[:max_days]

    daily_S, daily_dt, dates = [], [], []
    for f in files:
        fname = os.path.basename(f)
        # Extract YYYY-MM-DD from filename
        date_str = next(
            (p for p in fname.replace(".csv", "").split("_")
             if len(p) == 10 and p[4] == "-"),
            fname,
        )
        try:
            S, dt, _ = load_single_day(f)
            daily_S.append(S)
            daily_dt.append(dt)
            dates.append(date_str)
            sigma = np.sqrt(np.sum(np.diff(S) ** 2) / max(np.sum(dt[1:]), 1e-10))
            print(f"  {date_str}: {len(S):>8,} snapshots, σ={sigma:.6f}")
        except Exception as e:
            print(f"  SKIP {fname}: {e}")

    print(f"\nLoaded {len(dates)} days.")
    return daily_S, daily_dt, dates


def load_multi_day_with_features(
    data_dir: str,
    pair: str = "DOGEUSDT",
    pattern: str = "binance_book_snapshot_25_{date}_{pair}.csv",
    max_days: int | None = None,
    *,
    tick_size: float = 0.00001,
    imbalance_depth: int = 5,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[dict[str, np.ndarray]]]:
    """
    Load all available days plus no-leakage L1/L2 features.

    Features are aligned one-to-one with the returned midprice arrays and are
    computed only from contemporaneous order-book snapshots.
    """
    glob_pat = pattern.replace("{date}", "*").replace("{pair}", pair)
    files = sorted(glob.glob(os.path.join(data_dir, glob_pat)))

    if not files:
        raise FileNotFoundError(
            f"No files matching '{glob_pat}' in '{data_dir}'."
        )
    if max_days is not None:
        files = files[:max_days]

    daily_S, daily_dt, dates, daily_features = [], [], [], []
    for f in files:
        fname = os.path.basename(f)
        date_str = next(
            (p for p in fname.replace(".csv", "").split("_")
             if len(p) == 10 and p[4] == "-"),
            fname,
        )
        try:
            S, dt, _, features = load_single_day_with_features(
                f,
                tick_size=tick_size,
                imbalance_depth=imbalance_depth,
            )
            daily_S.append(S)
            daily_dt.append(dt)
            dates.append(date_str)
            daily_features.append(features)
            sigma = np.sqrt(np.sum(np.diff(S) ** 2) / max(np.sum(dt[1:]), 1e-10))
            print(f"  {date_str}: {len(S):>8,} snapshots, sigma={sigma:.6f}")
        except Exception as e:
            print(f"  SKIP {fname}: {e}")

    print(f"\nLoaded {len(dates)} days.")
    return daily_S, daily_dt, dates, daily_features
