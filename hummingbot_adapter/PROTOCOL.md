# Hummingbot forward validation: pre-registered protocol

**Thesis:** *Beyond Reward Shaping: CVaR-Constrained Reinforcement Learning for Avellaneda–Stoikov Market Making* (UvA MSc Computational Science).
**Stage:** RQ6 / PRD §10, the paper-trading deployment test.
**Branch:** `feat/hummingbot-forward-test`, from `cf3096d` (head of `fix/pre-production`). Commit tag: [robustness].
**Status:** PRE-REGISTERED. Committed 2026-10-18T21:55:11Z on feat/hummingbot-forward-test.

---

## 0. Pre-registration statement

**The commit time of this file is the pre-registration time.** After it:
- no parameter, threshold, model, metric or run window in this file changes;
- nothing is re-tuned, retrained, extended or re-run;
- every departure is logged as a deviation (§8);
- any analysis not listed in §9 is labelled *exploratory*.

**Seen before this commit:**
- the retrospective out-of-sample replay results for Jan 7–29 2025;
- live top-of-book quotes for DOGEUSDT and the 24 h ticker on 2026-10-16 and 2026-10-18 (price level only);
- the source code of Hummingbot 2.17.0;
- the frozen thesis code at `cf3096d`.

**Not seen before this commit:**
- the contents of the Tardis files for 2026-09-01 and 2026-10-01. They were downloaded on 2026-10-17/18 and hashed (§2), but not opened.
- any market data from the run days.

## 1. Purpose and limits

**Question (RQ6).** Does a frozen strategy from the thesis behave plausibly once real order management, exchange rules and prospective live data are involved?

This is an external-validity and engineering check, not a new experiment and not a tuning stage.

**What it can support (evidence for, not proof):**
- the frozen pipeline runs on a real exchange stack within the exchange's rules;
- whether the replay's main behavioural finding (B3 holds more inventory than B0) survives a large change of regime;
- a bracket on the simulated edge per fill, using the shadow replay and the paper fills;
- the maker fee at which each model breaks even;
- the cost of the daily horizon.

**What it cannot support:**
- anything about the tail risk of B3 against B1 or B2;
- profitability;
- statistical significance;
- real fills or queue position. Paper only: Binance does not serve Dutch residents, and no real orders are placed.

**Vocabulary:**
- "Hummingbot forward validation" or "paper-trading deployment test", never "live trading results";
- "evidence for", not "proof";
- the trading day (UTC) is the unit;
- the calibration uses the "best-price-reach execution proxy";
- Jan 7–29 2025 is "retrospective out-of-sample", and the run days are "prospective";
- B2 is "penalty-based reward design".

## 2. Frozen artifacts and versions

| Item | Value |
|---|---|
| Models | **B0 and B3 only** (author's decision D1, 2026-10-18). Adding a model is a deviation. |
| B3 policy | `models/ppo_b3_doge.zip`, SHA-256 `05ee4c97524456a6445f54cfe0a4bf353ccd539c20060f1176b83455cebe4e53` |
| B3 normalizer | `models/vecnorm_b3.pkl`, SHA-256 `d2be858e002d3b1e5a32bbe65822de3a4686cdbb91a81b8636241460a9b009b9` |
| Hash record | Both hashes above must equal their lines in Snellius `thesis_corrections_v1/results/profiling/profile_stage.sha256` (written by the profile stage, `fe5f9fe`/`cf3096d`). That file's SHA-256 is `f0b870b532e6727ee7b7728ab6e186460131a9b10aa7de8bf925f3f7edd629f8`. The copies committed to the repo are stale and are never used. |
| B0 | γ = **0.8908584807525282**, skew = 0, no policy. From Snellius `results/b0_gamma_fixed.txt`, SHA-256 `34cbb018dd15a9112f8c6f4f2cbb2874bea4efefbc1de3ad7c1895b6abe226c4`. The committed repo copy (`0.975143028728294`) is stale and is never used. |
| Parity inputs only | `results/formula_train_params.json` (Jan 1–6 calibration), SHA-256 `2d0b4a916bcf52e8641fac0a6e34f93f088166ee87c4b98b4d24975f1f6426f8`; `binance_book_snapshot_25_2025-01-07_DOGEUSDT.csv`, SHA-256 `87f86e54ed8587a8418d1668dd72934699ad09b665e7be8d7b6532ea458e9b31` |
| Thesis code | `cf3096d`. The frozen functions are imported from `procs/`, never copied or edited. |
| Adapter code | `hummingbot_adapter/` on this branch. The commit used for the run is recorded in `RUN_LOG.md` at the start of the warm-up. |
| Hummingbot | 2.17.0 = git tag `v2.17.0` = commit `9af100d`. The image is **built locally from that tag** with the repo's own Dockerfile (`docker build` with labels branch `v2.17.0` and commit `9af100d6822da7d2d0291a906c730ef172284ee2`), tag `hummingbot-local:2.17.0-9af100d`, image ID `sha256:49d7cb7ff1a981dad664f2abf9f9382daea63a33fefb72eb3b75ca1af450d014`. The Docker Hub tag `version-2.17.0` is **not** used: on 2026-10-18 it contained a development build (`hbot dev-2.18.0`, pushed 2026-10-07). If the image ID is not available at commit time, it is recorded in `RUN_LOG.md` before the smoke test as documentation; this is not a design change. |
| Calibration inputs | `binance_book_snapshot_25_2026-10-01_DOGEUSDT.csv.gz`, SHA-256 `4bf7ee3a5332d0f1bf9824c9313e2597780396b09f37399117e8efaad0d2b01b`; `binance_book_snapshot_25_2026-09-01_DOGEUSDT.csv.gz`, SHA-256 `86540e227ade1a654c1930d1d09a15958ce07a938cade691f95d155968a186a2` (fallback only) |

## 3. Venue

- **Connector:** Binance Spot DOGE-USDT through Hummingbot's `binance_paper_trade`. It uses public market data only, with no API keys.
- **Paper fills:** a resting limit order fills in full, at its own price, in either of two cases: at a clock tick the opposite best price has crossed it, or a public trade prints strictly through it. There is no queue and there are no partial fills.
- **Paper balances:** the defaults (DOGE 1,000,000; USDT 100,000). Each bot runs in its own container with its own paper account. Both see the same market path, so the days are paired.
- **Fees:** the paper default, 0.1 % for maker and taker orders. Fees are also recomputed afterwards on a grid (§9 Q-C).
- **Exchange rules (`exchangeInfo`, 2026-10-16/18):**
  - price tick 0.00001 USDT;
  - lot step 1 DOGE, minimum notional 1 USDT;
  - ORDERS limits of 100 per 10 s and 200,000 per day.

## 4. Calibration rule (decision D2)

- **Data:** the Tardis `binance` / `book_snapshot_25` / `DOGEUSDT` file for **2026-10-01**: same source, format and exchange clock as the thesis data.
- **Procedure:** the frozen `procs.gym.calibration.calibrate_from_days`, unchanged.
  - Configuration: the `FirstPassageCalibrationConfig` that `scripts.snellius.common.calibration_config_from_env` builds from `configs/final_pipeline.env` with tick 0.00001.
  - Depths {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20} ticks; the best-price-reach execution proxy; exposures start every 15 s (`fixed_stride`) and are censored at 15 s; at least 10 hits per depth; 600 s blocks for σ.
  - Inputs from the frozen `procs.gym.data_loader.load_single_day_calibration_arrays` on the decompressed CSV.
- **Output:** `hb_calibration.json`, containing:
  - σ, A, κ, the log-R² and the log-RMSE;
  - hits, exposure and censored counts per depth and side;
  - the day's number of rows, and its mean, minimum and maximum mid;
  - the input file's SHA-256;
  - the code commit.
- **Acceptance:** the fit is used as returned. No R² threshold or other criterion can reject it, and nothing is adjusted by hand.
- **Fallback (pre-registered):**
  1. If loading or fitting 2026-10-01 raises an error, the result is the frozen procedure on the pool of 2026-09-01 and 2026-10-01.
  2. If that raises too, stop: the run does not start until the author decides, and the decision is logged as a deviation.
- **Frozen, never recalibrated:** γ_B0, the B3 weights and the VecNormalize statistics.
- **Use:** σ and κ enter the quotes; A enters only the expected-fill analysis (§9 Q-B) and the shadow replay.

## 5. Execution

### 5.1 Inputs and stepping

- **Book sampling:** every **0.1 s** clock tick, the bot reads the connector's order book.
- **When a step is taken:** only when the top-25 book has changed since the last step. This mirrors the Tardis rows, which are written when the book changes.
- **Δt:** the time since the previous step, floored at 1e-10 s.
- **Mid:** S = (best bid + best ask) / 2.
- **Order of updates within a step:** exactly as `procs/gym/formula_as.py` (`FormulaASActionWrapper.step`, `_update_rolling_features`, `_obs`). The Jan 7 parity test is the arbiter, not this text.
- **Inventory** is q in model units, with **1 unit = 100 DOGE**, and **q_max = 10 units**.
  - **q is computed from the bot's own fill ledger:** Σ signed filled quantity / 100, which is an integer.
  - It is never computed from the account's DOGE balance, because the paper engine may take the buy fee in DOGE.
  - The account balance is reconciled against the ledger continuously (§7) and in the reports (§9 Q-D).

### 5.2 B3 observation, policy and action

**The seven features,** as in the replay:

| Feature | Definition |
|---|---|
| `inventory_norm` | clip(q / q_max, −1, 1) |
| `time_remaining_frac` | τ / T, with τ = seconds to the next 00:00 UTC and T = 86,400 s. In the replay T is the day's total Δt; this difference is documented. |
| `log_mid_rel` | ln(S / S₀), where S₀ is the first mid after 00:00 UTC |
| `rolling_vol_ticks` | √(ΣΔS² / ΣΔt) / tick, over the last 100 steps |
| `momentum_ticks` | (ΣΔS / ΣΔt) / tick, over the last 100 steps |
| `spread_ticks` | (ask₀ − bid₀) / tick |
| `lob_imbalance` | over the top 5 levels per side, as in `extract_market_feature_arrays` |

`rolling_vol_ticks` and `momentum_ticks` are 0 while fewer than 2 steps exist.

**Normalization:** the frozen VecNormalize observation statistics, clip 10, no reward normalization, as `freeze_vecnorm` does in evaluation.

**Policy:** the SB3 actor exported to numpy. The deterministic action is clip(μ, −1, 1). Parity with SB3's `predict(deterministic=True)` must be ≤ 1e-6.

**Action map and quotes:** the imported `map_formula_action` gives γ log-linearly in [0.01, 0.9] and skew = 5 ticks × u. The imported `avellaneda_stoikov_quotes` then computes the quotes with S, q (model units), τ, γ, skew and σ, κ from `hb_calibration.json`, at tick 0.00001.

**B0:** the same quote function with γ = γ_B0, skew = 0, and no observation or policy.

### 5.3 Order management

- **Orders:** at most one bid and one ask per bot, 100 DOGE each.
- **Targets:** computed at every step.
- **Re-quoting a side:** only if its target has moved by **at least 1 tick** and its resting order is **at least 1 s old**. After a fill, the replacement is posted at the next step.
- **Inventory cap:** no bid is posted while q ≥ q_max, and no ask while q ≤ −q_max. The replay suppresses those fills in the same way.
- **Order rate:**
  - Every new order counts towards internal caps of **100 per rolling 10 s** and **200,000 per UTC day**. This is conservative: Binance counts only unfilled orders.
  - A re-quote that would exceed a cap is skipped and counted.
  - Cancels are counted and reported.

### 5.4 Restarts

The state persists across a restart: S₀, the rolling window, the fill ledger, the booked DOGE fees, the day-start DOGE balance reference, the day id and any halt flag.
- **A restart within the same UTC day** restores S₀, the window and the ledger. The first step after the gap carries the whole gap as its Δt, as a data gap would in the replay.
- **If the bot was down across 23:59:50 UTC,** it closes any carried position at market at its first step, logs a deviation, and starts the new day as in §5.5.

### 5.5 End of day (decision D4)

- **23:59:50 UTC:**
  1. Cancel both orders.
  2. Close the position to q = 0 with a paper market order.
  3. Log the cost of the close (half-spread plus taker fee).
  4. Post no new quotes until 00:00:00.
- **00:00:00 UTC:** reset τ, T, S₀, the rolling window, the running peak and drawdown, the day's ledger and the day-start DOGE balance reference (to the balance after the close), then resume at the first step.

## 6. Run window

**Fixed-end rule:**
- the main run starts at the **first 00:00 UTC after the smoke test passes**;
- it lasts at least **2 full UTC days**;
- if the start slips past Wed 21 Oct 00:00 UTC, the end becomes start + 2 days.

**Plan:**

| Period | Times |
|---|---|
| Warm-up (excluded from all analyses) | from ≈ Mon 19 Oct 18:00 UTC to Tue 20 Oct 00:00 UTC |
| Main run | Tue 20 Oct 00:00 UTC → Fri 23 Oct 00:00 UTC |
| Prospective days | 2026-10-20, -21 and -22 |

All times are UTC. The run is not stopped early or extended because of results.

## 7. Safety limits

Stop means **halted**:
- cancel all orders;
- close the position at market;
- post no orders until the author clears the halt flag by hand.

A container restart does not clear the flag.

| Trigger | Action | Logged as |
|---|---|---|
| Ledger inventory \|q_ledger\| > q_max | Halt | Deviation (a bug) |
| Reconciliation residual: \|DOGE balance − (day-start DOGE balance + 100·q_ledger − DOGE fees booked)\| > 1 DOGE. The day-start reference is reset at 00:00 UTC, after the daily close. | Halt | Deviation (a bug) |
| No change of the top-25 book for 30 s | Cancel quotes and pause; resume when updates return. Halt after 5 min. | Incident |
| More than 10 rejects within 60 s | Halt | Deviation |
| A new order would exceed an internal rate cap (§5.3) | Skip that order | Counted |
| Daily drawdown of **gross** equity > **40 USDT** per bot (≈ 50 % of the ≈ 78 USDT notional at the cap). Gross equity is W from the fill ledger, before fees, marked at mid: the same W as in §9. | Close, then pause until 00:00 UTC; the day is kept | Finding (a desk's daily loss limit) |
| Clock offset against Binance server time (`GET /api/v3/time`, hourly) > 0.5 s | Warn | Incident |
| The same offset > 2 s | Pause quoting until it is below 2 s | Incident |

**Why gross, not net:**
- At 0.1 %, one 100-DOGE fill pays ≈ 0.008 USDT in fees, so 40 USDT is ≈ 5,000 fills' worth of fees.
- The replay made ≈ 6,000 fills a day, so a net limit could pause a bot on fees alone on an ordinary day.
- The run's data would then depend on a fee level that is applied only afterwards (§9 Q-C).

**Afterwards,** using the realized equity path, report when each of these limits would have triggered:
- daily loss limits of 10 %, 20 % and 30 % of the notional at the cap, at that day's 00:00 mid;
- a 40 USDT limit on net equity, at each fee in the Q-C grid.

**During the smoke test:** the kill switch is triggered once on purpose with a tiny threshold and must close the position.

## 8. Deviation policy

- **`hummingbot_adapter/RUN_LOG.md`** records every incident, restart, reconnect gap over 60 s, halt and manual action, each with:
  - the UTC time;
  - the bot;
  - what happened;
  - the action taken;
  - the minutes affected.
- **No reruns, no extensions, no parameter changes.**
- **A day with more than 1 h of downtime** is kept and flagged in every table.
- **Monitoring:** P&L is visible during monitoring, but no action is taken on it except the triggers in §7.

## 9. Analysis plan

Summaries are per bot and prospective day only, with no tests across fills (PRD PR-8).

**Every table is labelled** "descriptive; forward test; not pre-specified for inference".

**Units:**
- USDT figures are divided by 100 when compared with the replay, which traded 1 DOGE per unit.
- Ticks are also reported.

**Definitions:**
- **Wealth:** W_t = cash_t + 100 q_t S_t, from fills only and before fees. Fees are kept in a separate ledger.
- **Trading window:** 00:00:00 to 23:59:50 UTC.
- **Before close:** the window's end, before the close; comparable with the replay, which marks at mid with no liquidation cost.
- **After close:** including the close's execution price.
- **Time-weighted averages** use Δt, as in `fast_rollout` and `generate_trajectory_stats`.

### Q-A. Does B3 again hold more inventory than B0?

**Per bot and day, in the paper run and in the shadow replay:**
- time-weighted mean |q|;
- the near-cap share, i.e. the time share with |q| ≥ 8 = 0.8 q_max (as in the replay);
- q before the close;
- MaxDD, the maximum of the running peak of W minus W, before the close;
- P&L before the close and before fees.

**Expected:**
- B3's mean |q| is above B0's on every day.
- B3 ends the day before the close with q ≤ −8 on at least 2 of 3 days. In the replay this happened on 61–74 % of days for the PPO variants.

These expectations are less certain than in the replay, because B0's inventory term is weaker in the new regime (§10).

### Q-B. How much of the simulated edge per fill survives?

**Markouts.** Take a paper fill i at time tᵢ and price pᵢ, with side sᵢ = +1 for our buy and −1 for our sell. The filled order was posted at time t_post,i.
- **Edge at posting:** δ_post,i = sᵢ (S(t_post,i) − pᵢ) / tick. This is the depth the simulator would have credited.
- **Edge at fill:** δ_fill,i = sᵢ (S(tᵢ) − pᵢ) / tick.
  - A paper fill happens only after the market has traded through the order, so δ_fill is at most about −½ tick when the spread is 1 tick.
  - It is therefore never used as a denominator; it is reported because it shows the adverse selection directly.
- **Markout at horizon h:** mᵢ(h) = sᵢ (S(tᵢ + h) − pᵢ) / tick, for h ∈ {1, 10, 60} s.
- **Per bot and day, report:**
  - the mean δ_post;
  - the mean δ_fill;
  - the mean m(h);
  - the retention R(h) = Σ mᵢ(h) / Σ δ_post,i.
- The simulator implicitly assumes R ≈ 1, because it credits the posted depth and fills independently of the next mid move.

**Expected against actual fills.** E = Σ over steps and posted sides of 1 − exp(−A e^(−κ δ) Δt), where:
- δ is the resting order's live distance from the mid, in USDT;
- A and κ come from `hb_calibration.json`.

Report the ratio of paper fills to E per bot and day.

**Expected:** R(h) well below 1, and falling with h.

### Q-C. At which maker fee does each model break even?

Define, per bot and day:
- F, the traded notional of the fills;
- C, the notional of the close;
- P_pre, the P&L before the close and before fees;
- P_post, the P&L after the close and before fees.

**Fee curve:**
- before the close: P_pre − f·F;
- after the close: P_post − f·(F + C);
- for f ∈ {0, 0.01, 0.02, 0.075, 0.10} %.

**Break-even fees:**
- f*_pre = P_pre / F;
- f*_post = P_post / (F + C).

Also report the paper ledger's actual 0.1 % fees and the cost of the close.

**Expected:** f* well below the retail 0.1 %.

### Q-D. Does it run unattended within the exchange rules?

Per bot and day:
- uptime;
- restarts;
- forced websocket reconnects;
- stale-book events (count and seconds);
- rejects;
- the largest number of new orders in any 10 s window, and new orders per day, against 100 and 200,000;
- cancels per day;
- §7 trigger events;
- the largest clock offset;
- the largest ledger-to-balance reconciliation residual (as defined in §7).

**Expected:** within every limit, with a reconciliation residual of 0 DOGE.

### Shadow replay and simulator-against-paper (decision D3)

**The shadow replay.** For each bot and day, the frozen simulator runs on **that bot's own** snapshot file, with `hb_calibration.json`:
- **B0:** `fast_simulate` with γ_B0, Q_MAX = 10, fill seeds 42 and 43, and 20 rollouts each, as in `run_b0_fair.py`.
- **B3:** the frozen environment with `FormulaASActionWrapper` and the deterministic policy, with VecNormalize frozen through `freeze_vecnorm` and statistics from `generate_trajectory_stats`, as in `evaluate_formula_rl_per_day`; fill seeds 42 and 43 × 20.

That gives 40 paths per bot and day.

**PIT.** For each m ∈ {P_pre/100, MaxDD/100, mean |q|}:
- PIT = (#{paths with m < m_paper} + ½ #{paths with m = m_paper}) / 40.
- Report all values. Values below 0.05 or above 0.95 are flagged descriptively.
- If the simulator were right, PITs would be uniform (Diebold, Gunther & Tay 1998).
- The paper run also differs in cadence (§5.3), so a flag reflects deployment as a whole, not the fill model alone.

**The three-way reading, per model:**
- replay Jan 7–29 → shadow replay: the effect of the regime;
- shadow replay → paper: the effect of deployment.

### Data description and feature shift

**Per prospective day, from the bot's snapshot file:**
- the number of rows;
- the mean, minimum and maximum mid;
- σ by the frozen estimator;
- the share of rows with a 1-tick spread.

The same quantities are reported for 2026-10-01.

**For B3:**
- the mean z-score of each normalized feature;
- the share of steps with |z| > 3;
- the share clipped at ±10.

## 10. Pre-registered consequences of the regime

These are findings, not reasons to intervene.

- **Price and tick:**
  - DOGE was ≈ 0.078–0.082 USDT on 2026-10-16/18, about 4× below January 2025.
  - One tick is ≈ 1.2–1.3 bp, against ≈ 0.25–0.3 bp in January.
  - Under the January calibration, the A–S half-spread of 8.3–11.2 ticks is ≈ 0.10–0.14 % of price. The 1 Oct calibration will change κ, and therefore the half-spread.
- **γ** is in 1/USDT and stays frozen, as a preference parameter should.
  - If DOGE's percentage volatility is unchanged, σ in USDT/√s is ≈ 4× smaller, so the inventory term qγσ²τ is ≈ 16× weaker in ticks.
  - So B0 will lean on its inventory far less than in the replay.
- **B3's tick features:** `rolling_vol_ticks` and `momentum_ticks` are rates in ticks, so their level falls with the price; B3 sees inputs like those of quiet days. Their z-scores are reported.
- **Fees:** at ≈ 0.08 USDT, a 0.1 % fee is ≈ 8 ticks per fill, roughly the whole simulated edge per fill.

## 11. Outputs and archiving

- **Run outputs** go outside the repo, in `C:\Users\AA-ron2\Documents\hb_data\`, one folder per bot with daily gzipped files. Each folder has a `SHA256SUMS` file. The files are:
  - Tardis-format snapshots;
  - steps;
  - order events;
  - fills;
  - public trades;
  - health records.
- **Raw data is never committed.**
- **The analysis tables and figures** follow `docs/DESIGN_SYSTEM.md` (B0 #333333 solid, B3 #009E73 solid with marker) and carry the labels in §9.
- **The results summary** for the thesis appendix is written from those tables, together with:
  - `RUN_LOG.md`;
  - the deviations;
  - the claim limits in §1.

## References

- Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance* 8(3), 217–224.
- Diebold, F. X., Gunther, T. A. & Tay, A. S. (1998). Evaluating density forecasts with applications to financial risk management. *International Economic Review* 39(4), 863–883.
- Huang, R. D. & Stoll, H. R. (1996). Dealer versus auction markets: a paired comparison of execution costs on NASDAQ and the NYSE. *Journal of Financial Economics* 41(3), 313–357. https://doi.org/10.1016/0304-405X(95)00867-E. This is the realized spread, the academic form of a markout.
- Nosek, B. A., Ebersole, C. R., DeHaven, A. C. & Mellor, D. T. (2018). The preregistration revolution. *PNAS* 115(11), 2600–2606. https://doi.org/10.1073/pnas.1708274114
- Binance Spot API: `exchangeInfo` rate limits and filters; "LIMITS" (the unfilled order count); WebSocket Streams (connections are closed at 24 h).
- Hummingbot 2.17.0 source: `paper_trade_exchange.pyx`, `client_config_map.py`, `binance_constants.py`.
