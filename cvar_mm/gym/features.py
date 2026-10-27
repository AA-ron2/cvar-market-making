"""
Modular feature computer for enriched state spaces.

Features are appended to the base state [cash, inventory, time, midprice].
Add/remove features by changing the list passed to ``FeatureComputer``.

Usage::

    fc = FeatureComputer([RollingVolatility(100)])                   # just vol
    fc = FeatureComputer([RollingVolatility(100), Momentum(50)])     # vol + mom
    env = TradingEnvironment(..., feature_computer=fc)

References:
    • Cartea, Jaimungal & Sánchez-Betancourt (2023), alpha signals
    • Spooner et al. (2018), market features for RL market-making
"""

from __future__ import annotations

from collections import deque

import numpy as np


class Feature:
    """Base class for a single feature."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def reset(self, initial_price: float) -> None:
        raise NotImplementedError

    def update(self, price: float, dt: float) -> None:
        raise NotImplementedError

    def compute(self) -> float:
        raise NotImplementedError

    def get_bounds(self) -> tuple[float, float]:
        raise NotImplementedError


class RollingVolatility(Feature):
    """σ = sqrt(Σ(ΔS²) / Σ(Δt)) over a rolling window."""

    def __init__(self, window: int = 100):
        self.window = window
        self._dS2: deque[float] = deque(maxlen=window)
        self._dt: deque[float] = deque(maxlen=window)
        self._last_price: float = 0.0

    def reset(self, initial_price: float) -> None:
        self._dS2 = deque(maxlen=self.window)
        self._dt = deque(maxlen=self.window)
        self._last_price = initial_price

    def update(self, price: float, dt: float) -> None:
        dS = price - self._last_price
        self._dS2.append(dS * dS)
        self._dt.append(max(dt, 1e-10))
        self._last_price = price

    def compute(self) -> float:
        if len(self._dS2) < 2:
            return 0.0
        return float(np.sqrt(sum(self._dS2) / max(sum(self._dt), 1e-15)))

    def get_bounds(self) -> tuple[float, float]:
        return (0.0, 0.01)


class Momentum(Feature):
    """mean(ΔS) / Σ(Δt) over a rolling window."""

    def __init__(self, window: int = 100):
        self.window = window
        self._dS: deque[float] = deque(maxlen=window)
        self._dt: deque[float] = deque(maxlen=window)
        self._last_price: float = 0.0

    def reset(self, initial_price: float) -> None:
        self._dS = deque(maxlen=self.window)
        self._dt = deque(maxlen=self.window)
        self._last_price = initial_price

    def update(self, price: float, dt: float) -> None:
        self._dS.append(price - self._last_price)
        self._dt.append(max(dt, 1e-10))
        self._last_price = price

    def compute(self) -> float:
        if len(self._dS) < 2:
            return 0.0
        return float(sum(self._dS) / max(sum(self._dt), 1e-15))

    def get_bounds(self) -> tuple[float, float]:
        return (-0.001, 0.001)


class LobImbalance(Feature):
    """(bid_qty − ask_qty) / (bid_qty + ask_qty). Set via set_imbalance()."""

    def __init__(self, window: int = 1):
        self.window = window
        self._values: deque[float] = deque(maxlen=window)

    def reset(self, initial_price: float) -> None:
        self._values = deque(maxlen=self.window)

    def update(self, price: float, dt: float) -> None:
        pass

    def set_imbalance(self, bid_qty: float, ask_qty: float) -> None:
        total = bid_qty + ask_qty
        self._values.append((bid_qty - ask_qty) / max(total, 1e-15))

    def compute(self) -> float:
        return float(np.mean(self._values)) if self._values else 0.0

    def get_bounds(self) -> tuple[float, float]:
        return (-1.0, 1.0)


class FeatureComputer:
    """Composes multiple ``Feature`` objects into a single interface."""

    def __init__(self, features: list[Feature]):
        self.features = features
        self.n_features = len(features)

    def reset(self, initial_price: float) -> None:
        for f in self.features:
            f.reset(initial_price)

    def update(self, price: float, dt: float) -> None:
        for f in self.features:
            f.update(price, dt)

    def compute(self) -> np.ndarray:
        return np.array([f.compute() for f in self.features], dtype=np.float64)

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        bounds = [f.get_bounds() for f in self.features]
        low = np.array([b[0] for b in bounds], dtype=np.float32)
        high = np.array([b[1] for b in bounds], dtype=np.float32)
        return low, high

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.features]
