"""The Hummingbot forward-test bot (WO-HB2 deliverable 3): one StrategyV2Base script for B0 and B3.

Hummingbot 2.17.0 loads it as scripts.hb_forward_bot (deploy/scripts/hb_forward_bot.py imports the two classes
below; the module must hold exactly one StrategyV2Base and one config subclass). HbForwardBot is a thin shell
around bot_core.BotCore: every 0.1 s tick it reads the top 25 of the raw exchange book (the paper connector's
original_*_entries: its composite book also subtracts the bot's own market fills) and calls BotCore.on_tick;
the connector's events go to BotCore; BotCore's orders go out as LIMIT (quotes) or MARKET (closes) orders of
binance_paper_trade (paper ignores LIMIT_MAKER). Hooks must not raise (a raise skips Hummingbot's own order
tracking), so an exception halts the bot through BotCore.fail and is logged, never re-raised.

At construction the bot refuses to start unless the settings pass bot_config.check_settings (production: all
PROTOCOL.md constants) and the frozen inputs pass bot_config.verify_inputs; start() refuses a Hummingbot clock
whose tick is not sampling_interval_s.
"""

from __future__ import annotations

import os
import time
from decimal import Decimal
from itertools import islice
from typing import Any

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.connector.exchange.binance.binance_api_order_book_data_source import BinanceAPIOrderBookDataSource
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import MarketDict, OrderType, TradeType
from hummingbot.core.data_type.trade_fee import DeductedFromReturnsTradeFee
from hummingbot.core.event.event_forwarder import SourceInfoEventForwarder
from hummingbot.core.event.events import (
    BuyOrderCompletedEvent,
    BuyOrderCreatedEvent,
    MarketOrderFailureEvent,
    OrderBookEvent,
    OrderBookTradeEvent,
    OrderCancelledEvent,
    OrderFilledEvent,
    SellOrderCompletedEvent,
    SellOrderCreatedEvent,
)
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase

from hummingbot_adapter.bot_config import check_settings, from_mapping, protocol_default, verify_inputs
from hummingbot_adapter.bot_core import BotCore
from hummingbot_adapter.execution import CLOCK_RAW_RATE_PPM, StepClock
from hummingbot_adapter.monitor import ClockChecker, ReconnectWatcher
from hummingbot_adapter.policy import NumpyPolicy

# Clarification h: Hummingbot 2.17 re-snapshots a book only when no snapshot message has come for
# FULL_ORDER_BOOK_RESET_DELTA_SECONDS (1 h; core/data_type/order_book_tracker_data_source.py, read at every wait)
# and not after a websocket reconnect, so a missed level removal can leave a phantom level for up to an hour.
# Set at import, before the connector starts: its own restore_from_snapshot_and_diffs path then re-syncs the book
# from a REST snapshot every 60 s. HbForwardBot refuses to start unless the paper connector's data source sees it.
BOOK_RESYNC_S = 60
BinanceAPIOrderBookDataSource.FULL_ORDER_BOOK_RESET_DELTA_SECONDS = BOOK_RESYNC_S


class HbForwardBotConfig(StrategyV2ConfigBase):
    """Every field of bot_config.BotSettings, with its PROTOCOL.md default (test_hb_bot_config.py)."""

    script_file_name: str = os.path.basename(__file__)
    controllers_config: list[str] = []
    model: str
    mode: str
    exchange: str = protocol_default("exchange")
    trading_pair: str = protocol_default("trading_pair")
    artifacts_dir: str = protocol_default("artifacts_dir")
    repo_dir: str = protocol_default("repo_dir")
    data_root: str = protocol_default("data_root")
    tick_size: float = protocol_default("tick_size")
    order_amount_doge: int = protocol_default("order_amount_doge")
    q_max: int = protocol_default("q_max")
    feature_window: int = protocol_default("feature_window")
    imbalance_depth: int = protocol_default("imbalance_depth")
    book_levels: int = protocol_default("book_levels")
    day_seconds: int = protocol_default("day_seconds")
    sampling_interval_s: float = protocol_default("sampling_interval_s")
    gamma_b0: float = protocol_default("gamma_b0")
    gamma_min: float = protocol_default("gamma_min")
    gamma_max: float = protocol_default("gamma_max")
    skew_ticks_max: float = protocol_default("skew_ticks_max")
    requote_min_ticks: int = protocol_default("requote_min_ticks")
    requote_min_age_s: float = protocol_default("requote_min_age_s")
    rate_window_s: int = protocol_default("rate_window_s")
    rate_max_per_window: int = protocol_default("rate_max_per_window")
    rate_max_per_day: int = protocol_default("rate_max_per_day")
    close_seconds_before_midnight: int = protocol_default("close_seconds_before_midnight")
    paper_doge_balance: int = protocol_default("paper_doge_balance")
    paper_usdt_balance: int = protocol_default("paper_usdt_balance")
    fee_rate: float = protocol_default("fee_rate")
    halt_q_max: int = protocol_default("halt_q_max")
    residual_max_doge: float = protocol_default("residual_max_doge")
    stale_pause_s: float = protocol_default("stale_pause_s")
    stale_halt_s: float = protocol_default("stale_halt_s")
    reject_max: int = protocol_default("reject_max")
    reject_window_s: float = protocol_default("reject_window_s")
    drawdown_limit_usdt: float = protocol_default("drawdown_limit_usdt")
    clock_warn_s: float = protocol_default("clock_warn_s")
    clock_pause_s: float = protocol_default("clock_pause_s")
    clock_check_interval_s: float = protocol_default("clock_check_interval_s")
    save_interval_s: float = protocol_default("save_interval_s")
    heartbeat_interval_s: float = protocol_default("heartbeat_interval_s")

    def update_markets(self, markets: MarketDict) -> MarketDict:
        markets[self.exchange] = markets.get(self.exchange, set()) | {self.trading_pair}
        return markets


class _PaperExchange:
    """bot_core.ExchangePort on the strategy's binance_paper_trade connector."""

    def __init__(self, strategy: HbForwardBot):
        self._strategy = strategy
        self._exchange = strategy.settings.exchange
        self._pair = strategy.settings.trading_pair

    @property
    def connector(self) -> ConnectorBase:
        return self._strategy.connectors[self._exchange]

    def submit_limit(self, side: str, price: Decimal, amount: Decimal) -> str:
        place = self._strategy.buy if side == "buy" else self._strategy.sell
        return place(self._exchange, self._pair, amount, OrderType.LIMIT, price)

    def submit_market(self, side: str, amount: Decimal) -> str:
        place = self._strategy.buy if side == "buy" else self._strategy.sell
        return place(self._exchange, self._pair, amount, OrderType.MARKET)

    def cancel(self, order_id: str) -> None:
        self._strategy.cancel(self._exchange, self._pair, order_id)

    def balance(self, asset: str) -> Decimal:
        return Decimal(self.connector.get_balance(asset))

    def set_balance(self, asset: str, amount: Decimal) -> None:
        self.connector.set_balance(asset, Decimal(amount))

    def open_order_ids(self) -> set[str]:
        return {order.client_order_id for order in self.connector.limit_orders}


class HbForwardBot(StrategyV2Base):
    def __init__(self, connectors: dict[str, ConnectorBase], config: HbForwardBotConfig):
        super().__init__(connectors, config)
        self.config = config
        self.settings = from_mapping(config.model_dump())
        check_settings(self.settings)
        subdir = os.environ.get("HB_RUN_SUBDIR")  # set by deploy/docker-compose.yml from the run's .env
        expected = "" if self.settings.mode == "production" else "smoke/"
        if subdir != expected:
            raise ValueError(f"{self.settings.mode} mode needs HB_RUN_SUBDIR={expected!r} (the Hummingbot folders "
                             f"of the same run), found {subdir!r}")
        for name in ("data_root", "artifacts_dir", "repo_dir"):  # the mount points of deploy/docker-compose.yml
            if getattr(self.settings, name) != protocol_default(name):
                raise ValueError(f"{name} must be the container's {protocol_default(name)!r}, "
                                 f"got {getattr(self.settings, name)!r}")
        source = connectors[self.settings.exchange].order_book_tracker.data_source
        if source.FULL_ORDER_BOOK_RESET_DELTA_SECONDS != BOOK_RESYNC_S:  # clarification h
            raise ValueError(f"{type(source).__name__} re-snapshots every {source.FULL_ORDER_BOOK_RESET_DELTA_SECONDS} s, "
                             f"clarification h needs {BOOK_RESYNC_S} s")
        inputs = verify_inputs(self.settings)
        policy = NumpyPolicy.load(inputs.policy_path) if self.settings.model == "B3" else None
        self._base, self._quote = self.settings.trading_pair.split("-")
        self._stopping = False
        self._book = None
        self._snapshot_uid: int | None = None
        self._trade_forwarder = SourceInfoEventForwarder(self._on_public_trade)  # PubSub keeps weak references
        self._watcher = ReconnectWatcher()
        self._watcher.attach()
        # clarification i: the wall-clock anchor + CLOCK_MONOTONIC_RAW; j: RAW's rate against Binance corrected
        self._clock = StepClock(rate_ppm=CLOCK_RAW_RATE_PPM)
        # the §7 check reads the step clock without the day's ceiling (clarification k: ts can be held at a day's end)
        # and logs the uncorrected RAW offset beside it (j)
        self._clock_checker = ClockChecker(self._clock.unheld_us, self._clock.wall_us,
                                           monotonic_ns=self._clock.monotonic_ns, raw_us=self._clock.uncorrected_us,
                                           interval_s=self.settings.clock_check_interval_s,
                                           warn_s=self.settings.clock_warn_s)
        self._clock_checker_started = False
        self._errors: dict[tuple[str, str, str], list[float]] = {}  # (where, type, message) -> [last logged, count]
        self.core = BotCore(self.settings, inputs, _PaperExchange(self), self._clock, policy, probe=self._probe)

    def start(self, clock: Clock, timestamp: float) -> None:
        if abs(clock.tick_size - self.settings.sampling_interval_s) > 1e-12:
            raise ValueError(f"the Hummingbot clock ticks every {clock.tick_size} s; PROTOCOL §5.1 needs "
                             f"{self.settings.sampling_interval_s} s (conf_client.yml tick_size)")
        super().start(clock, timestamp)

    # --- the tick -------------------------------------------------------------------------------------------

    def on_tick(self) -> None:
        if self._stopping or self.core.refused is not None:
            return
        try:
            connector = self.connectors[self.settings.exchange]
            order_book = connector.get_order_book(self.settings.trading_pair)
            if order_book is not self._book:  # (re)subscribe when the tracker replaced the book object
                order_book.add_listener(OrderBookEvent.TradeEvent, self._trade_forwarder)
                self._book = order_book
            levels = self.settings.book_levels
            bids = tuple((row.price, row.amount) for row in islice(order_book.original_bid_entries(), levels))
            asks = tuple((row.price, row.amount) for row in islice(order_book.original_ask_entries(), levels))
            local_us = time.time_ns() // 1000
            self.core.on_tick((bids, asks), local_us)
            uid = order_book.snapshot_uid
            if self._snapshot_uid is not None and uid != self._snapshot_uid:  # a REST snapshot applied (clarification h)
                self.core.on_book_snapshot(self._snapshot_uid, uid, order_book.last_diff_uid)
            self._snapshot_uid = uid
            for result in self._clock_checker.drain():
                self.core.on_clock_check(result)
            for kind, level, message in self._watcher.drain():
                self.core.on_feed_event(kind, level, message)
            if not self._clock_checker_started:
                self._clock_checker.start()
                self._clock_checker_started = True
        except Exception as error:  # halts through BotCore.fail; Hummingbot keeps ticking (data keep flowing)
            self._fail("on_tick", error)

    def _probe(self) -> dict[str, Any]:
        connector = self.connectors[self.settings.exchange]
        tracker = connector.order_book_tracker
        book = self._book
        return {
            "connector_ready": bool(connector.ready),
            "network_status": str(connector.network_status),
            "ready_to_trade": bool(self.ready_to_trade),
            "snapshot_uid": None if book is None else book.snapshot_uid,
            "last_diff_uid": None if book is None else book.last_diff_uid,
            "book_resync_s": tracker.data_source.FULL_ORDER_BOOK_RESET_DELTA_SECONDS,  # clarification h
            "snapshots_processed": tracker.metrics.total_snapshots_processed,  # Hummingbot's own count
        }

    def _fail(self, where: str, error: Exception) -> None:
        """Log the error (the same error at most once a minute, with the count since) and halt through the core;
        a refused start is logged and leaves the bot inert."""

        key = (where, type(error).__name__, str(error)[:200])
        now = time.monotonic()
        seen = self._errors.setdefault(key, [float("-inf"), 0])
        seen[1] += 1
        if now - seen[0] >= 60.0:
            refused = "; start refused, the bot is inert" if self.core.refused is not None else "; halting"
            self.logger().error(f"hb_forward_bot: unexpected error in {where} ({seen[1]} times): {error!r}{refused}",
                                exc_info=error)
            seen[0], seen[1] = now, 0
        self.core.fail(where, error)

    # --- connector events --------------------------------------------------------------------------------

    def _fee(self, event: OrderFilledEvent, side: str) -> tuple[Decimal, str, Decimal]:
        """The fee as the paper engine charged it: a DeductedFromReturnsTradeFee comes out of the asset received
        (DOGE on a buy, USDT on a sell); otherwise the fee is quoted on the notional in USDT (or its token)."""

        fee = event.trade_fee
        if fee.flat_fees:
            raise ValueError(f"unexpected flat fees {fee.flat_fees} on {event.order_id}")
        rate = Decimal(fee.percent)
        if isinstance(fee, DeductedFromReturnsTradeFee):
            asset = self._base if side == "buy" else self._quote
        else:
            asset = fee.percent_token or self._quote
        amount = event.amount * rate if asset == self._base else event.amount * event.price * rate
        return amount, asset, rate

    def _mid(self) -> float | None:
        book = self._book
        if book is None:
            return None
        bid = next(iter(book.original_bid_entries()), None)
        ask = next(iter(book.original_ask_entries()), None)
        return None if bid is None or ask is None else (bid.price + ask.price) / 2

    def did_fill_order(self, event: OrderFilledEvent) -> None:
        try:
            side = "buy" if event.trade_type == TradeType.BUY else "sell"
            fee, asset, rate = self._fee(event, side)
            self.core.on_fill(event.order_id, side, Decimal(event.price), Decimal(event.amount), fee, asset, rate,
                              str(event.exchange_trade_id), self._mid())
        except Exception as error:
            self._fail("did_fill_order", error)

    def did_create_buy_order(self, event: BuyOrderCreatedEvent) -> None:
        try:
            self.core.on_created(event.order_id)
        except Exception as error:
            self._fail("did_create_buy_order", error)

    def did_create_sell_order(self, event: SellOrderCreatedEvent) -> None:
        try:
            self.core.on_created(event.order_id)
        except Exception as error:
            self._fail("did_create_sell_order", error)

    def did_cancel_order(self, event: OrderCancelledEvent) -> None:
        try:
            self.core.on_cancelled(event.order_id)
        except Exception as error:
            self._fail("did_cancel_order", error)

    def did_fail_order(self, event: MarketOrderFailureEvent) -> None:
        try:
            self.core.on_failed(event.order_id, event.error_message)
        except Exception as error:
            self._fail("did_fail_order", error)

    def did_complete_buy_order(self, event: BuyOrderCompletedEvent) -> None:
        try:
            self.core.on_completed(event.order_id, event.base_asset_amount, event.quote_asset_amount)
        except Exception as error:
            self._fail("did_complete_buy_order", error)

    def did_complete_sell_order(self, event: SellOrderCompletedEvent) -> None:
        try:
            self.core.on_completed(event.order_id, event.base_asset_amount, event.quote_asset_amount)
        except Exception as error:
            self._fail("did_complete_sell_order", error)

    def _on_public_trade(self, event_tag: int, order_book: Any, event: OrderBookTradeEvent) -> None:
        try:
            side = "buy" if event.type == TradeType.BUY else "sell"  # the taker's side
            self.core.on_public_trade(event.timestamp, event.trade_id, float(event.price), float(event.amount), side)
        except Exception as error:
            self._fail("public_trade", error)

    # --- lifecycle ------------------------------------------------------------------------------------------

    async def on_stop(self) -> None:
        self._stopping = True
        try:
            self.core.stop()
        except Exception as error:
            self._fail("on_stop", error)
        self._clock_checker.stop()
        self._watcher.detach()
        await super().on_stop()

    def format_status(self) -> str:
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        return self.core.status_text()
