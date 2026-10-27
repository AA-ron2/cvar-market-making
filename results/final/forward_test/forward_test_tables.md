# Hummingbot forward validation: analysis tables (descriptive; forward test; not pre-specified for inference)

- days: 2026-10-22, 2026-10-23
- params: <HB_ARTIFACTS>\hb_calibration.json (SHA-256 d71af506725da1e6dc047d82f58a30dcd3beb34c8cef4abc7f693e809282048c; pre-registered hb_calibration.json: yes)
- calibration day file: <HB_ARTIFACTS>\binance_book_snapshot_25_2026-10-01_DOGEUSDT.csv (SHA-256 a9eb64598825add8abf8a14fca0bf5f6bb4d564ac8e533068ba3f31a77c88015; recorded in --params: yes)
- run log: <HOME>\Documents\hb_analysis\RUN_LOG_3d362e5.md (SHA-256 39bba7d74cd1a5e311ebc3226775ae3f981d7e84c1a24fe18a81244187b25042)
- day folders: hb_b0 2026-10-22: SHA256SUMS 529e06b73539eb3a28aa15bffbe155b9242acfe3e660508cdaa46c02f25455a3; hb_b3 2026-10-22: SHA256SUMS c7293df9938245a79a51ce948320d2f3f79e5e81ee734155e131e65be0a4d783; hb_b0 2026-10-23: SHA256SUMS 049f962dea508221ae02887710ba006104f2022c2f94491cc0bda65c0c17de14; hb_b3 2026-10-23: SHA256SUMS 2f063df2215f59632e623d66915aa41ec9082bc68dc2ef05f2b0121c6a608c77
- shadow replay: <HOME>\Documents\hb_analysis\shadow_snellius
- code: ad68e0f77aa1af25e595f1983d67312f0ffa9c9a (uncommitted in hummingbot_adapter or procs: none)

## Q-A: inventory, P&L and drawdown before the close (descriptive; forward test; not pre-specified for inference)

_Time weights as fast_simulate (each step's q after the fills before it, times the Δt since the previous step); the trading window ends at 23:59:50 UTC; P_pre and MaxDD in USDT for 100-DOGE orders (divide by 100 to compare with the replay, which traded 1 DOGE per unit). maxdd_logged_usdt is the bot's own drawdown column (a check). Flags: B3's mean |q| above B0's on that day; B3's q before the close <= -8 (PROTOCOL §9 Q-A)._

| bot | day | model | mean_abs_q | near_cap_share | q_pre | P_pre_usdt | maxdd_pre_usdt | maxdd_logged_usdt | window_steps | weighted_s | last_window_step_utc | dt_mismatches | flag_b3_mean_abs_q_above_b0 | flag_b3_q_pre_le_minus_8 | downtime_flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hb_b0 | 2026-10-22 | B0 | 1.55948 | 0.0191812 | 8 | -0.336 | 0.6575 | 0.6575 | 458870 | 86389.5 | 2026-10-22T23:59:49.584597Z | 0 | n/a | n/a | no |
| hb_b3 | 2026-10-22 | B3 | 9.94924 | 0.997619 | -10 | -2.613 | 3.66 | 3.66 | 464589 | 86389.5 | 2026-10-22T23:59:49.585450Z | 0 | yes | yes | no |
| hb_b0 | 2026-10-23 | B0 | 1.49082 | 0.0314045 | -8 | -1.129 | 1.4965 | 1.4965 | 433733 | 86389.6 | 2026-10-23T23:59:49.683567Z | 0 | n/a | n/a | no |
| hb_b3 | 2026-10-23 | B3 | 9.93886 | 0.991586 | -10 | 4.082 | 1.325 | 1.325 | 438867 | 86389.6 | 2026-10-23T23:59:49.688447Z | 0 | yes | yes | no |


## Q-B: edge at posting and at the fill, markouts and retention (descriptive; forward test; not pre-specified for inference)

_Limit fills only (close == 0); ticks of 0.00001 USDT; m(h) = s(S(t+h) - p)/tick with S(t) the mid of the last step with ts <= t; a markout beyond the file's last step is dropped (dropped_h) and R(h) sums over the fills whose markout at h is kept. delta_*_mismatches count |recomputed - logged| > 1e-9 tick (expected 0)._

| bot | day | limit_fills | buys | sells | unknown_order_fills | mean_delta_post_ticks | mean_delta_fill_ticks | delta_post_mismatches | delta_fill_mismatches | max_abs_delta_difference | mean_m_1s | R_1s | dropped_1s | mean_m_10s | R_10s | dropped_10s | mean_m_60s | R_60s | dropped_60s | downtime_flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hb_b0 | 2026-10-22 | 1102 | 555 | 547 | 0 | 2.82396 | 1.58666 | 0 | 0 | 0 | -0.348004 | -0.123233 | 0 | -0.227314 | -0.0804949 | 0 | -0.194823 | -0.0689822 | 1 | no |
| hb_b3 | 2026-10-22 | 212 | 101 | 111 | 0 | 2.20991 | 1.12972 | 0 | 0 | 0 | -0.377358 | -0.170758 | 0 | -0.462264 | -0.209178 | 0 | -0.896226 | -0.40555 | 0 | no |
| hb_b0 | 2026-10-23 | 1116 | 554 | 562 | 0 | 2.72939 | 1.23477 | 0 | 0 | 0 | -0.47491 | -0.173999 | 0 | -0.614247 | -0.225049 | 0 | -0.550179 | -0.201576 | 0 | no |
| hb_b3 | 2026-10-23 | 150 | 70 | 80 | 0 | 2.22333 | 1.33333 | 0 | 0 | 0 | -0.286667 | -0.128936 | 0 | -1.86 | -0.836582 | 0 | 1.14 | 0.512744 | 0 | no |


## Q-B: paper fills against the expected fills E (descriptive; forward test; not pre-specified for inference)

_E = Σ 1 - exp(-A e^(-κ δ) Δt) over the window's intervals and resting sides, A and κ from hb_calibration.json; the orders resting after step i-1, at their distance from step i-1's mid, meet the Δt to step i (fast_simulate's timing). An order cancelled inside an interval (a §7 pause) counts for the whole interval, as in the simulator._

| bot | day | limit_fills_in_window | E | E_bid | E_ask | fills_over_E | intervals | downtime_flag |
|---|---|---|---|---|---|---|---|---|
| hb_b0 | 2026-10-22 | 1102 | 4645.08 | 2206.95 | 2438.13 | 0.23724 | 458869 | no |
| hb_b3 | 2026-10-22 | 212 | 1017.91 | 197.687 | 820.222 | 0.20827 | 464588 | no |
| hb_b0 | 2026-10-23 | 1116 | 4713.64 | 2490.96 | 2222.67 | 0.23676 | 433732 | no |
| hb_b3 | 2026-10-23 | 150 | 662.64 | 276.397 | 386.243 | 0.226367 | 438866 | no |


## Q-C: P&L net of a maker fee f on the fee grid (descriptive; forward test; not pre-specified for inference)

_P_pre - f F and P_post - f (F + C), USDT; f in %. s7_close_in_C: a §7 close inside the day is part of C (see the next table)._

| bot | day | fee_pct | P_pre_net_usdt | P_post_net_usdt | s7_close_in_C | downtime_flag |
|---|---|---|---|---|---|---|
| hb_b0 | 2026-10-22 | 0 | -0.336 | -0.34 | no | no |
| hb_b0 | 2026-10-22 | 0.01 | -1.211 | -1.22131 | no | no |
| hb_b0 | 2026-10-22 | 0.02 | -2.086 | -2.10261 | no | no |
| hb_b0 | 2026-10-22 | 0.075 | -6.8985 | -6.94979 | no | no |
| hb_b0 | 2026-10-22 | 0.1 | -9.086 | -9.15305 | no | no |
| hb_b3 | 2026-10-22 | 0 | -2.613 | -2.618 | no | no |
| hb_b3 | 2026-10-22 | 0.01 | -2.78112 | -2.794 | no | no |
| hb_b3 | 2026-10-22 | 0.02 | -2.94924 | -2.97001 | no | no |
| hb_b3 | 2026-10-22 | 0.075 | -3.8739 | -3.93803 | no | no |
| hb_b3 | 2026-10-22 | 0.1 | -4.2942 | -4.37803 | no | no |
| hb_b0 | 2026-10-23 | 0 | -1.129 | -1.133 | no | no |
| hb_b0 | 2026-10-23 | 0.01 | -2.01234 | -2.02263 | no | no |
| hb_b0 | 2026-10-23 | 0.02 | -2.89567 | -2.91226 | no | no |
| hb_b0 | 2026-10-23 | 0.075 | -7.75403 | -7.80522 | no | no |
| hb_b0 | 2026-10-23 | 0.1 | -9.96237 | -10.0293 | no | no |
| hb_b3 | 2026-10-23 | 0 | 4.082 | 4.077 | no | no |
| hb_b3 | 2026-10-23 | 0.01 | 3.96278 | 3.94991 | no | no |
| hb_b3 | 2026-10-23 | 0.02 | 3.84356 | 3.82283 | no | no |
| hb_b3 | 2026-10-23 | 0.075 | 3.18785 | 3.12386 | no | no |
| hb_b3 | 2026-10-23 | 0.1 | 2.8898 | 2.80615 | no | no |


## Q-C: traded notional, break-even fees, paper fees and the close (descriptive; forward test; not pre-specified for inference)

_F: limit fills; C: every market close of the day at its fill price rounded to the tick: the day's close (the 23:59:50 close, or the close of a position carried past 00:00; C_day_close_usdt) and any §7 close (halt or drawdown) inside the day, which s7_close_in_C flags (the author's decision of 2026-10-24); f* = P / notional in % (n/a for a zero notional); P_post: W after the day's last fill (its cash when the day ends flat, also if no step follows the close; flat_at_day_end); the paper fees are the ledger's 0.1 % (USDT fees plus DOGE fees times the fill price); a close's cost is its half-spread against the mid at its submission plus its fee._

| bot | day | F_usdt | C_usdt | C_day_close_usdt | P_pre_usdt | P_post_usdt | f_star_pre_pct | f_star_post_pct | paper_fees_limit_usdt | paper_fees_close_usdt | paper_fees_s7_close_usdt | paper_fees_total_usdt | close_cost_usdt | close_side | close_amount_doge | close_price | close_fill_utc | s7_close_fills | s7_close_purposes | s7_close_notional_usdt | s7_close_cost_usdt | close_fills_unknown_purpose | flat_at_day_end | last_step_after_close_fills | P_post_last_step_usdt | limit_fills_after_last_window_step | s7_close_in_C | downtime_flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hb_b0 | 2026-10-22 | 8750 | 63.056 | 63.056 | -0.336 | -0.34 | -0.00384 | -0.00385791 | 8.75 | 0.063056 | 0 | 8.81305 | 0.067056 | sell | 800 | 0.07882 | 2026-10-22T23:59:55.249894Z | 0 | none | 0 | 0 | 0 | yes | yes | -0.34 | 0 | no | no |
| hb_b3 | 2026-10-22 | 1681.2 | 78.83 | 78.83 | -2.613 | -2.618 | -0.155424 | -0.148747 | 1.6812 | 0.07883 | 0 | 1.76003 | 0.08383 | buy | 1000 | 0.07883 | 2026-10-22T23:59:55.087364Z | 0 | none | 0 | 0 | 0 | yes | yes | -2.618 | 0 | no | no |
| hb_b0 | 2026-10-23 | 8833.37 | 62.92 | 62.92 | -1.129 | -1.133 | -0.0127811 | -0.0127356 | 8.83337 | 0.06292 | 0 | 8.89629 | 0.06692 | buy | 800 | 0.07865 | 2026-10-23T23:59:55.122100Z | 0 | none | 0 | 0 | 0 | yes | yes | -1.133 | 0 | no | no |
| hb_b3 | 2026-10-23 | 1192.2 | 78.65 | 78.65 | 4.082 | 4.077 | 0.342391 | 0.320808 | 1.1922 | 0.07865 | 0 | 1.27086 | 0.08365 | buy | 1000 | 0.07865 | 2026-10-23T23:59:55.084400Z | 0 | none | 0 | 0 | 0 | yes | yes | 4.077 | 0 | no | no |


## §7 afterwards: when each loss limit would have triggered (descriptive; forward test; not pre-specified for inference)

_The realized path of every step of the day; the notional at the cap is 1,000 DOGE times the day's first mid; net equity is W minus f times the notional of the limit fills up to the step._

| bot | day | limit | fee_pct | threshold_usdt | first_utc | downtime_flag |
|---|---|---|---|---|---|---|
| hb_b0 | 2026-10-22 | gross drawdown > 10% of the notional at the cap | n/a | 8.1975 | never | no |
| hb_b0 | 2026-10-22 | gross drawdown > 20% of the notional at the cap | n/a | 16.395 | never | no |
| hb_b0 | 2026-10-22 | gross drawdown > 30% of the notional at the cap | n/a | 24.5925 | never | no |
| hb_b0 | 2026-10-22 | net-equity drawdown > 40 USDT | 0 | 40 | never | no |
| hb_b0 | 2026-10-22 | net-equity drawdown > 40 USDT | 0.01 | 40 | never | no |
| hb_b0 | 2026-10-22 | net-equity drawdown > 40 USDT | 0.02 | 40 | never | no |
| hb_b0 | 2026-10-22 | net-equity drawdown > 40 USDT | 0.075 | 40 | never | no |
| hb_b0 | 2026-10-22 | net-equity drawdown > 40 USDT | 0.1 | 40 | never | no |
| hb_b3 | 2026-10-22 | gross drawdown > 10% of the notional at the cap | n/a | 8.1975 | never | no |
| hb_b3 | 2026-10-22 | gross drawdown > 20% of the notional at the cap | n/a | 16.395 | never | no |
| hb_b3 | 2026-10-22 | gross drawdown > 30% of the notional at the cap | n/a | 24.5925 | never | no |
| hb_b3 | 2026-10-22 | net-equity drawdown > 40 USDT | 0 | 40 | never | no |
| hb_b3 | 2026-10-22 | net-equity drawdown > 40 USDT | 0.01 | 40 | never | no |
| hb_b3 | 2026-10-22 | net-equity drawdown > 40 USDT | 0.02 | 40 | never | no |
| hb_b3 | 2026-10-22 | net-equity drawdown > 40 USDT | 0.075 | 40 | never | no |
| hb_b3 | 2026-10-22 | net-equity drawdown > 40 USDT | 0.1 | 40 | never | no |
| hb_b0 | 2026-10-23 | gross drawdown > 10% of the notional at the cap | n/a | 7.8825 | never | no |
| hb_b0 | 2026-10-23 | gross drawdown > 20% of the notional at the cap | n/a | 15.765 | never | no |
| hb_b0 | 2026-10-23 | gross drawdown > 30% of the notional at the cap | n/a | 23.6475 | never | no |
| hb_b0 | 2026-10-23 | net-equity drawdown > 40 USDT | 0 | 40 | never | no |
| hb_b0 | 2026-10-23 | net-equity drawdown > 40 USDT | 0.01 | 40 | never | no |
| hb_b0 | 2026-10-23 | net-equity drawdown > 40 USDT | 0.02 | 40 | never | no |
| hb_b0 | 2026-10-23 | net-equity drawdown > 40 USDT | 0.075 | 40 | never | no |
| hb_b0 | 2026-10-23 | net-equity drawdown > 40 USDT | 0.1 | 40 | never | no |
| hb_b3 | 2026-10-23 | gross drawdown > 10% of the notional at the cap | n/a | 7.8825 | never | no |
| hb_b3 | 2026-10-23 | gross drawdown > 20% of the notional at the cap | n/a | 15.765 | never | no |
| hb_b3 | 2026-10-23 | gross drawdown > 30% of the notional at the cap | n/a | 23.6475 | never | no |
| hb_b3 | 2026-10-23 | net-equity drawdown > 40 USDT | 0 | 40 | never | no |
| hb_b3 | 2026-10-23 | net-equity drawdown > 40 USDT | 0.01 | 40 | never | no |
| hb_b3 | 2026-10-23 | net-equity drawdown > 40 USDT | 0.02 | 40 | never | no |
| hb_b3 | 2026-10-23 | net-equity drawdown > 40 USDT | 0.075 | 40 | never | no |
| hb_b3 | 2026-10-23 | net-equity drawdown > 40 USDT | 0.1 | 40 | never | no |


## Q-D: operation within the exchange rules (descriptive; forward test; not pre-specified for inference)

_Downtime: step gaps over 30 s (from 00:00 and to 24:00 included); uptime = 1 - downtime / 24 h. New orders: every submit (quotes and closes) against 100 per rolling 10 s and 200,000 per day; cancels: the bot's cancel requests. Clock offsets from the hourly checks against Binance (step clock and wall clock)._

| bot | day | uptime_pct | downtime_min | gaps_over_30s | longest_gap_s | longest_step_gap_s | container_restarts | fresh_starts | stops | ws_reconnects | ws_errors | resubscriptions | stale_book_pauses | stale_book_pause_s | quoting_paused_min | rejects | order_failed_rows | new_orders | max_new_orders_10s | rate_skips | not_quoting_skips | cancels | cancelled | s7_triggers | clock_checks | clock_check_errors | max_abs_step_offset_s | max_abs_wall_offset_s | max_abs_residual_doge | downtime_flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hb_b0 | 2026-10-22 | 99.9416 | 0.841659 | 1 | 50.4995 | 50.4995 | 0 | 0 | 0 | 2 | 5 | 2 | 1 | 20.4987 | 0.341644 | 0 | 0 | 36573 | 38 | 0 | 0 | 35470 | 35470 | stale_book:pause=1; stale_book:resume=1 | 24 | 0 | 0.067313 | 0.0635825 | 0 | no |
| hb_b3 | 2026-10-22 | 99.9416 | 0.841643 | 1 | 50.4986 | 50.4986 | 0 | 0 | 0 | 2 | 5 | 2 | 1 | 20.4981 | 0.341636 | 0 | 0 | 27136 | 25 | 0 | 0 | 26923 | 26923 | stale_book:pause=1; stale_book:resume=1 | 24 | 0 | 0.065772 | 0.058005 | 0 | no |
| hb_b0 | 2026-10-23 | 100 | 0 | 0 | 0 | 22.2992 | 0 | 0 | 0 | 1 | 4 | 1 | 0 | 0 | 0 | 0 | 0 | 36079 | 30 | 0 | 0 | 34962 | 34962 | none | 24 | 0 | 0.063994 | 0.079992 | 0 | no |
| hb_b3 | 2026-10-23 | 100 | 0 | 0 | 0 | 22.3 | 0 | 0 | 0 | 1 | 4 | 1 | 0 | 0 | 0 | 0 | 0 | 22331 | 29 | 0 | 0 | 22180 | 22180 | none | 24 | 0 | 0.066561 | 0.0498385 | 0 | no |


## Q-D: step gaps over 30 s (descriptive; forward test; not pre-specified for inference)


| bot | day | start_utc | end_utc | seconds | in_network_window | downtime_flag |
|---|---|---|---|---|---|---|
| hb_b0 | 2026-10-22 | 2026-10-22T22:47:02.783229Z | 2026-10-22T22:47:53.282767Z | 50.4995 | yes | no |
| hb_b3 | 2026-10-22 | 2026-10-22T22:47:02.783635Z | 2026-10-22T22:47:53.282205Z | 50.4986 | yes | no |


## Q-D: the recomputed counts beside the bot's own day_end counters (descriptive; forward test; not pre-specified for inference)

_A check: the bot's counters from its day_end health row (the rolling 10 s window and the daily counts of its RateLimiter; bot_reconnects counts websocket closes and errors). n/a when the day has no day_end row._

| bot | day | new_orders | bot_day_orders | max_new_orders_10s | bot_max_in_window | cancels | bot_cancels | bot_rate_skips | max_abs_residual_doge | bot_max_abs_residual_doge | bot_reconnects | bot_triggers | downtime_flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hb_b0 | 2026-10-22 | 36573 | 36573 | 38 | 38 | 35470 | 35470 | 0 | 0 | 0 | 7 | {"stale_book:pause": 1, "stale_book:resume": 1} | no |
| hb_b3 | 2026-10-22 | 27136 | 27136 | 25 | 25 | 26923 | 26923 | 0 | 0 | 0 | 7 | {"stale_book:pause": 1, "stale_book:resume": 1} | no |
| hb_b0 | 2026-10-23 | 36079 | 36079 | 30 | 30 | 34962 | 34962 | 0 | 0 | 0 | 5 | {} | no |
| hb_b3 | 2026-10-23 | 22331 | 22331 | 29 | 29 | 22180 | 22180 | 0 | 0 | 0 | 5 | {} | no |


## Q-D: §7 quoting pauses and halts (descriptive; forward test; not pre-specified for inference)


| bot | day | trigger | start_utc | end_utc | seconds | downtime_flag |
|---|---|---|---|---|---|---|
| hb_b0 | 2026-10-22 | stale_book | 2026-10-22T22:47:32.784099Z | 2026-10-22T22:47:53.282767Z | 20.4987 | no |
| hb_b3 | 2026-10-22 | stale_book | 2026-10-22T22:47:32.784073Z | 2026-10-22T22:47:53.282205Z | 20.4981 | no |


## Q-D, exploratory: trade fills at the price, book and clock events (descriptive; forward test; not pre-specified for inference)

_Exploratory (not in PROTOCOL §9). Among the opposite-side taker trades logged after a limit fill's order was posted and in the 0.5 s up to the fill, in whole ticks: through, one printed strictly through the order price; at_price (= engine_through + not_engine_through), none through but one exactly at it; no_trade, none reached it (a book cross at a clock tick, or a trigger outside the window). The paper engine fills on a print strictly through the order, comparing the Decimal price with the float print exactly, so an at-tick print can trigger a fill (engine_through) or cannot (not_engine_through). Book re-syncs: applied REST snapshots (clarification h); crossed or locked books from the bot's counters (its health rows stop at 100 a day); round-away quotes: clarification a._

| bot | day | at_price_trade_fills | at_price_engine_through | at_price_not_engine_through | through_trade_fills | no_trade_fills | book_resyncs | snapshots_ahead_of_diffs | snapshot_errors | crossed_or_locked_books | crossed_or_locked_source | crossed_book_rows | skipped_steps | re_anchors | clock_backward | round_bid_steps | round_ask_steps | round_bid_orders | round_ask_orders | downtime_flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hb_b0 | 2026-10-22 | 816 | 707 | 109 | 270 | 16 | 1432 | 11 | 2 | 0 | day_end counters | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no |
| hb_b3 | 2026-10-22 | 149 | 124 | 25 | 51 | 12 | 1432 | 8 | 2 | 0 | day_end counters | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 1 | no |
| hb_b0 | 2026-10-23 | 878 | 697 | 181 | 223 | 15 | 1432 | 20 | 0 | 0 | day_end counters | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no |
| hb_b3 | 2026-10-23 | 107 | 94 | 13 | 32 | 11 | 1432 | 24 | 0 | 0 | day_end counters | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 1 | no |


## Q-B by fill class, exploratory: edge, markouts and fills over E (descriptive; forward test; not pre-specified for inference)

_Exploratory (WO-HB3b; not in PROTOCOL §9). The limit fills by the at-price method's class (the table above), then all of them and the through + no_trade fills together; Q-B's definitions per class (ticks; R(h) over the fills whose markout at h is kept). fills_over_E: the window's fills of the row over E; for through + no_trade a sensitivity to the paper engine's Decimal-against-float comparison. Hypothesis, reported and not tuned on: delta_fill near +1/2 tick for at-price fills and <= -1/2 tick for no_trade (book-cross) fills; a positive delta_fill elsewhere would reflect the depth stream (100 ms batches) lagging the trade stream._

| bot | day | class | fills | share | mean_delta_post_ticks | mean_delta_fill_ticks | mean_m_1s | R_1s | dropped_1s | mean_m_10s | R_10s | dropped_10s | mean_m_60s | R_60s | dropped_60s | fills_over_E | downtime_flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hb_b0 | 2026-10-22 | through | 270 | 0.245009 | 2.96667 | 1.90926 | -1.07222 | -0.361423 | 0 | -0.259259 | -0.0873908 | 0 | 0.0333333 | 0.011236 | 0 | n/a | no |
| hb_b0 | 2026-10-22 | at_price_engine_through | 707 | 0.641561 | 2.79703 | 1.82956 | -0.0615276 | -0.0219975 | 0 | -0.148515 | -0.0530973 | 0 | -0.326025 | -0.116561 | 0 | n/a | no |
| hb_b0 | 2026-10-22 | at_price_not_engine_through | 109 | 0.0989111 | 2.70183 | -0.5 | -0.444954 | -0.164686 | 0 | -0.96789 | -0.358234 | 0 | -0.279817 | -0.103565 | 0 | n/a | no |
| hb_b0 | 2026-10-22 | no_trade | 16 | 0.0145191 | 2.4375 | -0.375 | -0.125 | -0.0512821 | 0 | 1.875 | 0.769231 | 0 | 2.5 | 1.0274 | 1 | n/a | no |
| hb_b0 | 2026-10-22 | all | 1102 | 1 | 2.82396 | 1.58666 | -0.348004 | -0.123233 | 0 | -0.227314 | -0.0804949 | 0 | -0.194823 | -0.0689822 | 1 | 0.23724 | no |
| hb_b0 | 2026-10-22 | through + no_trade | 286 | 0.259528 | 2.93706 | 1.78147 | -1.01923 | -0.347024 | 0 | -0.13986 | -0.047619 | 0 | 0.163158 | 0.0555224 | 1 | 0.0615705 | no |
| hb_b3 | 2026-10-22 | through | 51 | 0.240566 | 2.47059 | 1.31373 | -0.794118 | -0.321429 | 0 | 0.107843 | 0.0436508 | 0 | -2.85294 | -1.15476 | 0 | n/a | no |
| hb_b3 | 2026-10-22 | at_price_engine_through | 124 | 0.584906 | 2.37097 | 1.57258 | -0.0645161 | -0.0272109 | 0 | -0.58871 | -0.248299 | 0 | -0.862903 | -0.363946 | 0 | n/a | no |
| hb_b3 | 2026-10-22 | at_price_not_engine_through | 25 | 0.117925 | 1.54 | -0.5 | -0.94 | -0.61039 | 0 | -1.82 | -1.18182 | 0 | -0.58 | -0.376623 | 0 | n/a | no |
| hb_b3 | 2026-10-22 | no_trade | 12 | 0.0566038 | 0.833333 | -0.833333 | -0.666667 | -0.8 | 0 | 1.25 | 1.5 | 0 | 6.41667 | 7.7 | 0 | n/a | no |
| hb_b3 | 2026-10-22 | all | 212 | 1 | 2.20991 | 1.12972 | -0.377358 | -0.170758 | 0 | -0.462264 | -0.209178 | 0 | -0.896226 | -0.40555 | 0 | 0.20827 | no |
| hb_b3 | 2026-10-22 | through + no_trade | 63 | 0.29717 | 2.15873 | 0.904762 | -0.769841 | -0.356618 | 0 | 0.325397 | 0.150735 | 0 | -1.0873 | -0.503676 | 0 | 0.0618916 | no |
| hb_b0 | 2026-10-23 | through | 223 | 0.199821 | 2.76457 | 1.48206 | -1.02018 | -0.369019 | 0 | -1.35202 | -0.489051 | 0 | -1.2713 | -0.459854 | 0 | n/a | no |
| hb_b0 | 2026-10-23 | at_price_engine_through | 697 | 0.624552 | 2.73099 | 1.64634 | -0.195839 | -0.07171 | 0 | -0.166428 | -0.0609404 | 0 | -0.1901 | -0.0696086 | 0 | n/a | no |
| hb_b0 | 2026-10-23 | at_price_not_engine_through | 181 | 0.162186 | 2.66022 | -0.5 | -0.870166 | -0.327103 | 0 | -1.29006 | -0.484943 | 0 | -0.986188 | -0.370717 | 0 | n/a | no |
| hb_b0 | 2026-10-23 | no_trade | 15 | 0.0134409 | 2.96667 | -0.633333 | -0.566667 | -0.191011 | 0 | -2.3 | -0.775281 | 0 | -1.3 | -0.438202 | 0 | n/a | no |
| hb_b0 | 2026-10-23 | all | 1116 | 1 | 2.72939 | 1.23477 | -0.47491 | -0.173999 | 0 | -0.614247 | -0.225049 | 0 | -0.550179 | -0.201576 | 0 | 0.23676 | no |
| hb_b0 | 2026-10-23 | through + no_trade | 238 | 0.213262 | 2.77731 | 1.34874 | -0.991597 | -0.357035 | 0 | -1.41176 | -0.508321 | 0 | -1.27311 | -0.458396 | 0 | 0.0504918 | no |
| hb_b3 | 2026-10-23 | through | 32 | 0.213333 | 2.40625 | 1.6875 | -1.21875 | -0.506494 | 0 | -4.4375 | -1.84416 | 0 | 2.15625 | 0.896104 | 0 | n/a | no |
| hb_b3 | 2026-10-23 | at_price_engine_through | 94 | 0.626667 | 2.45213 | 1.74468 | 0.0957447 | 0.0390456 | 0 | -0.702128 | -0.286334 | 0 | 1.60638 | 0.655098 | 0 | n/a | no |
| hb_b3 | 2026-10-23 | at_price_not_engine_through | 13 | 0.0866667 | 1.11538 | -0.5 | -0.576923 | -0.517241 | 0 | -0.423077 | -0.37931 | 0 | -5.73077 | -5.13793 | 0 | n/a | no |
| hb_b3 | 2026-10-23 | no_trade | 11 | 0.0733333 | 1.04545 | -1.04545 | -0.5 | -0.478261 | 0 | -5.95455 | -5.69565 | 0 | 2.31818 | 2.21739 | 0 | n/a | no |
| hb_b3 | 2026-10-23 | all | 150 | 1 | 2.22333 | 1.33333 | -0.286667 | -0.128936 | 0 | -1.86 | -0.836582 | 0 | 1.14 | 0.512744 | 0 | 0.226367 | no |
| hb_b3 | 2026-10-23 | through + no_trade | 43 | 0.286667 | 2.05814 | 0.988372 | -1.03488 | -0.502825 | 0 | -4.82558 | -2.34463 | 0 | 2.19767 | 1.0678 | 0 | 0.064892 | no |


## P&L split, exploratory: P_pre into fill edge and inventory (descriptive; forward test; not pre-specified for inference)

_Exploratory (WO-HB3d; not in PROTOCOL §9). USDT for 100-DOGE orders, over the trading window: P_pre = fill edge + inventory (+ the edge of a market fill inside the window, a §7 close; none is expected). inventory = Σ over the window's steps i >= 1 of 100 q_{i-1} (S_i - S_{i-1}); fill edge = Σ over the limit fills of s a (S - p), with S the mid of the first step whose wealth includes the fill: the first step with ts >= the fill's ts (each step marks W over every fill booked before it, and a fill's ts never exceeds that step's; a fill at a step's own ts is taken to be in that step, which fails only while the step clock is held, as at a restart floor; fills_at_step_ts counts them). S is that step's mid, not mid_at_fill (Q-B's delta_fill). A day starts flat, so a fill before its first step enters step 0; the 23:59:50 close and a limit fill after the last window step are left out. mean_fill_edge_ticks: the fill edge per DOGE in ticks. Checks (within 1e-9 USDT): the parts' sum minus P_pre, and the largest |W_i - W_{i-1} - step i's parts| over the window; position_mismatch_steps: window steps whose 100 q differs from the cumulative fills (expected 0)._

| bot | day | model | P_pre_usdt | fill_edge_usdt | inventory_usdt | fill_edge_buys_usdt | fill_edge_sells_usdt | market_fill_edge_usdt | limit_fills | market_fills | fills_at_step_ts | mean_fill_edge_ticks | parts_minus_P_pre_usdt | max_abs_step_mismatch_usdt | position_mismatch_steps | within_tolerance | downtime_flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hb_b0 | 2026-10-22 | B0 | -0.336 | 0.7225 | -1.0585 | 0.459 | 0.2635 | 0 | 1102 | 0 | 0 | 0.655626 | 3.46009e-15 | 1.68615e-14 | 0 | yes | no |
| hb_b3 | 2026-10-22 | B3 | -2.613 | 0.097 | -2.71 | 0.0465 | 0.0505 | 0 | 212 | 0 | 0 | 0.457547 | -6.62805e-15 | 2.59947e-14 | 0 | yes | no |
| hb_b0 | 2026-10-23 | B0 | -1.129 | 0.119 | -1.248 | 0.053 | 0.066 | 0 | 1116 | 0 | 0 | 0.106631 | 3.63164e-15 | 2.12545e-14 | 0 | yes | no |
| hb_b3 | 2026-10-23 | B3 | 4.082 | 0.02 | 4.062 | 0.027 | -0.007 | 0 | 150 | 0 | 0 | 0.133333 | -5.35921e-16 | 2.31135e-14 | 0 | yes | no |


## Q-D, exploratory: clock checks and gaps inside and outside the mobile-hotspot window (descriptive; forward test; not pre-specified for inference)

_Exploratory: the window 2026-10-22T22:47:00Z to 2026-10-23T05:39:00Z (iPhone hotspot, RUN_LOG 2026-10-23T05:49:24Z); a gap belongs to the part where it starts._

| bot | day | part | clock_checks | max_abs_step_offset_s | max_abs_wall_offset_s | mean_wall_offset_s | gaps_over_30s | downtime_s | downtime_flag |
|---|---|---|---|---|---|---|---|---|---|
| hb_b0 | 2026-10-22 | inside | 1 | 0.0201415 | 0.0635825 | -0.0635825 | 1 | 50.4995 | no |
| hb_b0 | 2026-10-22 | outside | 23 | 0.067313 | 0.0370205 | -0.012301 | 0 | 0 | no |
| hb_b3 | 2026-10-22 | inside | 1 | 0.02597 | 0.058005 | -0.058005 | 1 | 50.4986 | no |
| hb_b3 | 2026-10-22 | outside | 23 | 0.065772 | 0.0378685 | -0.0135469 | 0 | 0 | no |
| hb_b0 | 2026-10-23 | inside | 6 | 0.0367145 | 0.0456055 | -0.0389085 | 0 | 0 | no |
| hb_b0 | 2026-10-23 | outside | 18 | 0.063994 | 0.079992 | -0.0227122 | 0 | 0 | no |
| hb_b3 | 2026-10-23 | inside | 6 | 0.0420745 | 0.0498385 | -0.042017 | 0 | 0 | no |
| hb_b3 | 2026-10-23 | outside | 18 | 0.066561 | 0.0276005 | -0.0174878 | 0 | 0 | no |


## RUN_LOG.md entries about the run's days (descriptive; forward test; not pre-specified for inference)

_Every RUN_LOG.md entry written from 2026-10-22T00:00:00Z on, listed under each analysed day whose date appears in its stamp or text (an entry about a day is written after it); an entry that names one bot only is listed for it, any other for both. The CSV and results.json hold the full text; this table cuts it at 500 characters._

| bot | day | written_utc | entry | downtime_flag |
|---|---|---|---|---|
| hb_b0 | 2026-10-22 | 2026-10-22T21:57:08Z | Planned network change (pre-announced). - Planned ISP maintenance at the author's home: 2026-10-22T23:00Z → 2026-10-23T05:00Z (01:00–07:00 Amsterdam), internet only (power unaffected). - Mitigation, decided before the event (no rule, model, parameter or config change): (1) ≈ 22:30Z: the laptop's connection changed so that it can switch networks without blocking traffic, for the rest of the main run; (2) ≈ 22:40Z: the laptop's Wi-Fi switches to the author's phone hotspot; (3) after 05:… | no |
| hb_b3 | 2026-10-22 | 2026-10-22T21:57:08Z | Planned network change (pre-announced). - Planned ISP maintenance at the author's home: 2026-10-22T23:00Z → 2026-10-23T05:00Z (01:00–07:00 Amsterdam), internet only (power unaffected). - Mitigation, decided before the event (no rule, model, parameter or config change): (1) ≈ 22:30Z: the laptop's connection changed so that it can switch networks without blocking traffic, for the rest of the main run; (2) ≈ 22:40Z: the laptop's Wi-Fi switches to the author's phone hotspot; (3) after 05:… | no |
| hb_b0 | 2026-10-23 | 2026-10-22T21:57:08Z | Planned network change (pre-announced). - Planned ISP maintenance at the author's home: 2026-10-22T23:00Z → 2026-10-23T05:00Z (01:00–07:00 Amsterdam), internet only (power unaffected). - Mitigation, decided before the event (no rule, model, parameter or config change): (1) ≈ 22:30Z: the laptop's connection changed so that it can switch networks without blocking traffic, for the rest of the main run; (2) ≈ 22:40Z: the laptop's Wi-Fi switches to the author's phone hotspot; (3) after 05:… | no |
| hb_b3 | 2026-10-23 | 2026-10-22T21:57:08Z | Planned network change (pre-announced). - Planned ISP maintenance at the author's home: 2026-10-22T23:00Z → 2026-10-23T05:00Z (01:00–07:00 Amsterdam), internet only (power unaffected). - Mitigation, decided before the event (no rule, model, parameter or config change): (1) ≈ 22:30Z: the laptop's connection changed so that it can switch networks without blocking traffic, for the rest of the main run; (2) ≈ 22:40Z: the laptop's Wi-Fi switches to the author's phone hotspot; (3) after 05:… | no |
| hb_b0 | 2026-10-22 | 2026-10-23T05:49:24Z | Network window, actual (pre-announced in 01dec99); incident: reconnect gap at 22:47Z. Times from the author (Amsterdam = UTC+2): connection changed (step (1)) 2026-10-22T22:45Z; laptop on the phone hotspot 22:47Z (no internet in between); back on home Wi-Fi 2026-10-23T05:39Z. From both bots' health, step, order and fill files: - 1. Incident, hb_b0 and hb_b3 (the same times on both): reconnect gap at 22:47Z. At 22:45:40.53Z the websocket closed (the connection change) and re-subscribed at 22:45:42.38Z on the changed connection; … | no |
| hb_b3 | 2026-10-22 | 2026-10-23T05:49:24Z | Network window, actual (pre-announced in 01dec99); incident: reconnect gap at 22:47Z. Times from the author (Amsterdam = UTC+2): connection changed (step (1)) 2026-10-22T22:45Z; laptop on the phone hotspot 22:47Z (no internet in between); back on home Wi-Fi 2026-10-23T05:39Z. From both bots' health, step, order and fill files: - 1. Incident, hb_b0 and hb_b3 (the same times on both): reconnect gap at 22:47Z. At 22:45:40.53Z the websocket closed (the connection change) and re-subscribed at 22:45:42.38Z on the changed connection; … | no |
| hb_b0 | 2026-10-23 | 2026-10-23T05:49:24Z | Network window, actual (pre-announced in 01dec99); incident: reconnect gap at 22:47Z. Times from the author (Amsterdam = UTC+2): connection changed (step (1)) 2026-10-22T22:45Z; laptop on the phone hotspot 22:47Z (no internet in between); back on home Wi-Fi 2026-10-23T05:39Z. From both bots' health, step, order and fill files: - 1. Incident, hb_b0 and hb_b3 (the same times on both): reconnect gap at 22:47Z. At 22:45:40.53Z the websocket closed (the connection change) and re-subscribed at 22:45:42.38Z on the changed connection; … | no |
| hb_b3 | 2026-10-23 | 2026-10-23T05:49:24Z | Network window, actual (pre-announced in 01dec99); incident: reconnect gap at 22:47Z. Times from the author (Amsterdam = UTC+2): connection changed (step (1)) 2026-10-22T22:45Z; laptop on the phone hotspot 22:47Z (no internet in between); back on home Wi-Fi 2026-10-23T05:39Z. From both bots' health, step, order and fill files: - 1. Incident, hb_b0 and hb_b3 (the same times on both): reconnect gap at 22:47Z. At 22:45:40.53Z the websocket closed (the connection change) and re-subscribed at 22:45:42.38Z on the changed connection; … | no |
| hb_b0 | 2026-10-22 | 2026-10-24T08:11:31Z | Run end. - Run window: the main run was 2026-10-22T00:00:00Z → 2026-10-24T00:00:00Z (prospective days 2026-10-22 and 2026-10-23), on run code b9de69c with no code, config or rule change. Container restarts over the run: 0 (hb_b0 and hb_b3 up since 2026-10-21T12:14:03Z); no start, restart or stop row in the run. - Day 2026-10-23, final. hb_b0: q −8 at the last trade step (23:59:49.68Z); at 23:59:50.08Z both quotes were cancelled and 800 DOGE bought at market (mid 0.078645), filled at 23:59:55.12Z… | no |
| hb_b3 | 2026-10-22 | 2026-10-24T08:11:31Z | Run end. - Run window: the main run was 2026-10-22T00:00:00Z → 2026-10-24T00:00:00Z (prospective days 2026-10-22 and 2026-10-23), on run code b9de69c with no code, config or rule change. Container restarts over the run: 0 (hb_b0 and hb_b3 up since 2026-10-21T12:14:03Z); no start, restart or stop row in the run. - Day 2026-10-23, final. hb_b0: q −8 at the last trade step (23:59:49.68Z); at 23:59:50.08Z both quotes were cancelled and 800 DOGE bought at market (mid 0.078645), filled at 23:59:55.12Z… | no |
| hb_b0 | 2026-10-23 | 2026-10-24T08:11:31Z | Run end. - Run window: the main run was 2026-10-22T00:00:00Z → 2026-10-24T00:00:00Z (prospective days 2026-10-22 and 2026-10-23), on run code b9de69c with no code, config or rule change. Container restarts over the run: 0 (hb_b0 and hb_b3 up since 2026-10-21T12:14:03Z); no start, restart or stop row in the run. - Day 2026-10-23, final. hb_b0: q −8 at the last trade step (23:59:49.68Z); at 23:59:50.08Z both quotes were cancelled and 800 DOGE bought at market (mid 0.078645), filled at 23:59:55.12Z… | no |
| hb_b3 | 2026-10-23 | 2026-10-24T08:11:31Z | Run end. - Run window: the main run was 2026-10-22T00:00:00Z → 2026-10-24T00:00:00Z (prospective days 2026-10-22 and 2026-10-23), on run code b9de69c with no code, config or rule change. Container restarts over the run: 0 (hb_b0 and hb_b3 up since 2026-10-21T12:14:03Z); no start, restart or stop row in the run. - Day 2026-10-23, final. hb_b0: q −8 at the last trade step (23:59:49.68Z); at 23:59:50.08Z both quotes were cancelled and 800 DOGE bought at market (mid 0.078645), filled at 23:59:55.12Z… | no |


## Data description: the snapshot files (descriptive; forward test; not pre-specified for inference)

_Through the frozen loader (load_single_day_calibration_arrays); sigma by the frozen estimator (_sigma_from_day, 600 s blocks), USDT per sqrt(s); the share of rows with a 1-tick spread._

| bot | day | file | rows | mid_mean | mid_min | mid_max | sigma | one_tick_spread_share | downtime_flag |
|---|---|---|---|---|---|---|---|---|---|
| hb_b0 | 2026-10-22 | snapshots_2026-10-22.csv.gz | 458903 | 0.0795136 | 0.077855 | 0.082185 | 5.34865e-06 | 0.999976 | no |
| hb_b3 | 2026-10-22 | snapshots_2026-10-22.csv.gz | 464623 | 0.0795127 | 0.077855 | 0.082185 | 5.51672e-06 | 0.999972 | no |
| hb_b0 | 2026-10-23 | snapshots_2026-10-23.csv.gz | 433773 | 0.0796462 | 0.0774 | 0.080695 | 5.36494e-06 | 0.999984 | no |
| hb_b3 | 2026-10-23 | snapshots_2026-10-23.csv.gz | 438909 | 0.079644 | 0.0774 | 0.080695 | 5.17225e-06 | 0.999982 | no |
| tardis_calibration | 2026-10-01 | binance_book_snapshot_25_2026-10-01_DOGEUSDT.csv | 540022 | 0.0947043 | 0.093505 | 0.096095 | 7.55645e-06 | 0.999985 | n/a |


## Feature shift of B3's normalized observation (descriptive; forward test; not pre-specified for inference)

_z = the frozen VecNormalize normalization (clip 10) of each feature at each step of the trading window (the steps file's z columns); shares of steps._

| bot | day | feature | mean_z | share_abs_z_gt_3 | share_abs_z_clipped | downtime_flag |
|---|---|---|---|---|---|---|
| hb_b3 | 2026-10-22 | inventory_norm | 0.873563 | 0 | 0 | no |
| hb_b3 | 2026-10-22 | time_remaining_frac | 0.0274134 | 0 | 0 | no |
| hb_b3 | 2026-10-22 | log_mid_rel | -1.46266 | 0 | 0 | no |
| hb_b3 | 2026-10-22 | rolling_vol_ticks | -1.0309 | 0 | 0 | no |
| hb_b3 | 2026-10-22 | momentum_ticks | -0.0120084 | 0 | 0 | no |
| hb_b3 | 2026-10-22 | spread_ticks | -0.0126017 | 2.79817e-05 | 2.79817e-05 | no |
| hb_b3 | 2026-10-22 | lob_imbalance | 0.143853 | 0 | 0 | no |
| hb_b3 | 2026-10-23 | inventory_norm | -0.602 | 0 | 0 | no |
| hb_b3 | 2026-10-23 | time_remaining_frac | 0.00517915 | 0 | 0 | no |
| hb_b3 | 2026-10-23 | log_mid_rel | -0.207064 | 0 | 0 | no |
| hb_b3 | 2026-10-23 | rolling_vol_ticks | -1.02777 | 0 | 0 | no |
| hb_b3 | 2026-10-23 | momentum_ticks | -0.0090692 | 0 | 0 | no |
| hb_b3 | 2026-10-23 | spread_ticks | -0.0126993 | 1.82288e-05 | 1.82288e-05 | no |
| hb_b3 | 2026-10-23 | lob_imbalance | 0.0947167 | 0 | 0 | no |


## Shadow replay against paper: the Q-A quantities and their PIT (descriptive; forward test; not pre-specified for inference)

_PIT = (#{m < m_paper} + ½ #{m = m_paper}) / 40 over the shadow paths (fill seeds 42 and 43 x 20 rollouts) of the frozen simulator on that bot's own snapshot file; flagged below 0.05 or above 0.95 (descriptive). PROTOCOL §9's PIT set is P_pre/100, MaxDD/100 and mean |q|; the near-cap share's PIT is exploratory; the shadow's q is at the end of the file, so it gets no PIT. P_pre and MaxDD divided by 100 (USDT per DOGE unit, as the replay). The paper window ends at 23:59:50, the replay runs to the end of the file. 5-95 %: numpy percentiles (linear interpolation)._

| bot | day | model | metric | role | paper | shadow_mean | shadow_p05 | shadow_p95 | below | ties | paths | pit | pit_flag | shadow_share_q_le_minus_8 | downtime_flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hb_b0 | 2026-10-22 | B0 | P_pre/100 | PIT (PROTOCOL §9) | -0.00336 | 0.135248 | 0.129483 | 0.141036 | 0 | 0 | 40 | 0 | yes | n/a | no |
| hb_b0 | 2026-10-22 | B0 | MaxDD/100 | PIT (PROTOCOL §9) | 0.006575 | 0.00238412 | 0.00153425 | 0.003832 | 40 | 0 | 40 | 1 | yes | n/a | no |
| hb_b0 | 2026-10-22 | B0 | mean \|q\| | PIT (PROTOCOL §9) | 1.55948 | 2.02632 | 1.84513 | 2.23074 | 0 | 0 | 40 | 0 | yes | n/a | no |
| hb_b0 | 2026-10-22 | B0 | near-cap share | PIT, exploratory | 0.0191812 | 0.0219955 | 0.00866316 | 0.0351414 | 15 | 0 | 40 | 0.375 | no | n/a | no |
| hb_b0 | 2026-10-22 | B0 | q (paper: before the close; shadow: end of file) | descriptive, no PIT | 8 | 0.55 | -9 | 10 | 34 | 0 | 40 | n/a | n/a | 0.15 | no |
| hb_b3 | 2026-10-22 | B3 | P_pre/100 | PIT (PROTOCOL §9) | -0.02613 | 0.0344083 | 0.0116945 | 0.064958 | 0 | 0 | 40 | 0 | yes | n/a | no |
| hb_b3 | 2026-10-22 | B3 | MaxDD/100 | PIT (PROTOCOL §9) | 0.0366 | 0.0138811 | 0.0080885 | 0.018641 | 40 | 0 | 40 | 1 | yes | n/a | no |
| hb_b3 | 2026-10-22 | B3 | mean \|q\| | PIT (PROTOCOL §9) | 9.94924 | 9.82839 | 9.75634 | 9.90055 | 40 | 0 | 40 | 1 | yes | n/a | no |
| hb_b3 | 2026-10-22 | B3 | near-cap share | PIT, exploratory | 0.997619 | 0.986463 | 0.975816 | 0.996025 | 40 | 0 | 40 | 1 | yes | n/a | no |
| hb_b3 | 2026-10-22 | B3 | q (paper: before the close; shadow: end of file) | descriptive, no PIT | -10 | -9.95 | -10 | -10 | 0 | 39 | 40 | n/a | n/a | 1 | no |
| hb_b0 | 2026-10-23 | B0 | P_pre/100 | PIT (PROTOCOL §9) | -0.01129 | 0.134821 | 0.12402 | 0.143396 | 0 | 0 | 40 | 0 | yes | n/a | no |
| hb_b0 | 2026-10-23 | B0 | MaxDD/100 | PIT (PROTOCOL §9) | 0.014965 | 0.0047725 | 0.0024945 | 0.00786275 | 40 | 0 | 40 | 1 | yes | n/a | no |
| hb_b0 | 2026-10-23 | B0 | mean \|q\| | PIT (PROTOCOL §9) | 1.49082 | 2.00135 | 1.83522 | 2.19743 | 0 | 0 | 40 | 0 | yes | n/a | no |
| hb_b0 | 2026-10-23 | B0 | near-cap share | PIT, exploratory | 0.0314045 | 0.0187624 | 0.00661057 | 0.0302572 | 38 | 0 | 40 | 0.95 | no | n/a | no |
| hb_b0 | 2026-10-23 | B0 | q (paper: before the close; shadow: end of file) | descriptive, no PIT | -8 | -0.9 | -8 | 9 | 1 | 3 | 40 | n/a | n/a | 0.1 | no |
| hb_b3 | 2026-10-23 | B3 | P_pre/100 | PIT (PROTOCOL §9) | 0.04082 | 0.0356075 | 0.029132 | 0.0569455 | 33 | 0 | 40 | 0.825 | no | n/a | no |
| hb_b3 | 2026-10-23 | B3 | MaxDD/100 | PIT (PROTOCOL §9) | 0.01325 | 0.0124311 | 0.0120587 | 0.012834 | 40 | 0 | 40 | 1 | yes | n/a | no |
| hb_b3 | 2026-10-23 | B3 | mean \|q\| | PIT (PROTOCOL §9) | 9.93886 | 9.90011 | 9.81958 | 9.93035 | 40 | 0 | 40 | 1 | yes | n/a | no |
| hb_b3 | 2026-10-23 | B3 | near-cap share | PIT, exploratory | 0.991586 | 0.995028 | 0.987102 | 0.998462 | 6 | 0 | 40 | 0.15 | no | n/a | no |
| hb_b3 | 2026-10-23 | B3 | q (paper: before the close; shadow: end of file) | descriptive, no PIT | -10 | -9.95 | -10 | -9.95 | 0 | 38 | 40 | n/a | n/a | 1 | no |


## Figure: fig_forward_inventory (descriptive; forward test; not pre-specified for inference)

Inventory q (units of 100 DOGE; cap 10) of B0 (A–S, charcoal) and B3 (CVaR-constrained PPO, green with markers) against UTC time on each prospective day of the Hummingbot forward validation (paper trading on binance_paper_trade, DOGE-USDT): the fill ledger's q at every step, as a step function. Dotted: the near-cap level |q| = 8; dash-dot: the cap; dashed: the 23:59:50 close; grey band: the mobile-hotspot network window (exploratory). One bot per line, one day per panel; descriptive; forward test; not pre-specified for inference.
