"""A-S quotes for B3 and B0, and the orders they become (PROTOCOL §3, §5.2–5.3).

Both models quote through the imported frozen functions on length-1 arrays: B3 maps its action with
map_formula_action, B0 uses gamma_B0 and skew 0, and avellaneda_stoikov_quotes computes the quotes (tick
grid, mid clamp). tau is clamped at 0, as compute_formula_as_depths and fast_simulate clamp it.

Runtime module: standard library, numpy and cvar_mm.gym.formula_as only (it runs inside the Hummingbot
container, which must not have gymnasium: see features.py).
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any

import numpy as np

from cvar_mm.gym.formula_as import FormulaASActionConfig, avellaneda_stoikov_quotes, map_formula_action

QUOTE_KEYS = ("bid", "ask", "gamma", "skew", "delta_bid", "delta_ask")
MIN_NOTIONAL_USDT = Decimal("1")  # Binance DOGEUSDT NOTIONAL filter (PROTOCOL §3)
GRID_TOLERANCE_TICKS = 1e-9  # float rounding of k * tick; a half-tick price is off the grid


def _finite(name: str, value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


class QuoteEngine:
    """Quotes from the frozen A-S formula with sigma and kappa from the calibration."""

    def __init__(
        self,
        sigma: float,
        kappa: float,
        tick_size: float = 1e-5,
        gamma_min: float = 0.01,
        gamma_max: float = 0.9,
        skew_ticks_max: float = 5.0,
    ):
        self.config = FormulaASActionConfig(
            sigma=_finite("sigma", sigma),
            kappa=_finite("kappa", kappa),
            tick_size=_finite("tick_size", tick_size),
            gamma_min=_finite("gamma_min", gamma_min),
            gamma_max=_finite("gamma_max", gamma_max),
            skew_ticks_max=_finite("skew_ticks_max", skew_ticks_max),
        )
        self._tick = Decimal(repr(self.config.tick_size))

    def b3(self, action: Any, mid: float, q: float, tau: float) -> dict[str, float]:
        """B3: gamma and skew from the policy action in [-1, 1]^2 (map_formula_action)."""

        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError(f"action must be two finite numbers, got {action!r}")
        gamma, skew = map_formula_action(action.reshape(1, 2), self.config)
        return self._quotes(mid, q, tau, gamma, skew)

    def b0(self, mid: float, q: float, tau: float, gamma: float) -> dict[str, float]:
        """B0: the fixed gamma and skew 0, as fast_simulate quotes."""

        gamma = _finite("gamma", gamma)
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma!r}")
        return self._quotes(mid, q, tau, np.full(1, gamma), np.zeros(1))

    def _quotes(self, mid: float, q: float, tau: float, gamma: np.ndarray, skew: np.ndarray) -> dict[str, float]:
        mid = _finite("mid", mid)
        if mid <= 0:
            raise ValueError(f"mid must be positive, got {mid!r}")
        _, diagnostics = avellaneda_stoikov_quotes(
            midprice=np.full(1, mid),
            inventory=np.full(1, _finite("q", q)),
            tau=np.maximum(np.full(1, _finite("tau", tau)), 0.0),
            gamma=gamma,
            skew=skew,
            sigma=self.config.sigma,
            kappa=self.config.kappa,
            tick_size=self.config.tick_size,
        )
        return {key: float(diagnostics[key][0]) for key in QUOTE_KEYS}

    def to_orders(
        self,
        quotes: dict[str, float],
        q: float,
        q_max: int = 10,
        unit: int = 100,
    ) -> list[dict[str, str]]:
        """The orders to post: a bid unless q >= q_max and an ask unless q <= -q_max, each `unit` DOGE
        (1 model unit, PROTOCOL §5.1), at prices on the tick grid as Decimal strings.

        Refuses a crossed pair, and a posted order off the tick grid or under the 1 USDT minimum notional.
        """

        q = _finite("q", q)
        if not q.is_integer():
            raise ValueError(f"q must be an integer number of units, got {q!r}")
        if int(q_max) != q_max or q_max < 1:
            raise ValueError(f"q_max must be a positive integer, got {q_max!r}")
        if int(unit) != unit or unit < 1:
            raise ValueError(f"unit must be a positive integer number of DOGE (lot step 1), got {unit!r}")
        bid, ask = _finite("bid", quotes["bid"]), _finite("ask", quotes["ask"])
        if not bid < ask:
            raise ValueError(f"crossed quotes: bid {bid!r} >= ask {ask!r}")
        orders = []
        if q < q_max:
            orders.append(self._order("buy", self._grid_price("bid", bid), int(unit)))
        if q > -q_max:
            orders.append(self._order("sell", self._grid_price("ask", ask), int(unit)))
        return orders

    def _grid_price(self, name: str, price: Any) -> Decimal:
        price = _finite(name, price)
        if price <= 0:
            raise ValueError(f"{name} must be positive, got {price!r}")
        ticks = price / self.config.tick_size
        nearest = round(ticks)
        if abs(ticks - nearest) > GRID_TOLERANCE_TICKS:
            raise ValueError(f"{name} {price!r} is not on the {self.config.tick_size!r} tick grid")
        return (Decimal(nearest) * self._tick).quantize(self._tick)

    def _order(self, side: str, price: Decimal, unit: int) -> dict[str, str]:
        amount = Decimal(unit)
        if price * amount < MIN_NOTIONAL_USDT:
            raise ValueError(f"{side} notional {price * amount} USDT is below {MIN_NOTIONAL_USDT} USDT")
        return {"side": side, "price": format(price, "f"), "amount": format(amount, "f")}
