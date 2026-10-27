# Forward-test analysis and shadow replay (WO-HB3)

> **Public copy.** The analysis record of the forward test, published with redactions (results/final/REDACTIONS.md).
> - Paths are placeholders (the legend is in the paragraph before section 1). The run's data files (the bots' day folders) and the shadow-replay outputs are private; the published outputs are in `results/final/forward_test/`.
> - Module names follow the public package: `cvar_mm` and `scripts/pipeline`. The original analysis ran at commit ad68e0f, where they were `procs` and `scripts.snellius` (hummingbot_adapter/README.md).
> - Two calibration hashes. The run used `hb_calibration.json` with SHA-256 `d71af506…` (`CALIBRATION_SHA256_RUN` in bot_config.py); the published `config/hb_calibration.json` has `ebbe6247…` (`CALIBRATION_SHA256`). The two files differ only in two local paths (`files[0]` "input" and "csv"), where the Windows user name is replaced. `read_params` counts either hash as pre-registered, so the analysis of the original run files gives the same result as before.
> - `results.json` embeds the RUN_LOG text; in the published copy that text is redacted like RUN_LOG.md.

The tools of PROTOCOL.md §9 (and §7, "afterwards") for the Hummingbot forward validation (RQ6), written on
2026-10-23 during day 2 of the main run. PROTOCOL.md stays authoritative; this file records how the tools run, and
the readings this work order fixes where §9 is silent. Every table the tools write is labelled
"descriptive; forward test; not pre-specified for inference": per bot and prospective day only, with no pooling
across days and no test across fills (PRD PR-8).

**When the readings were fixed.** The readings below were fixed before the 2026-10-23 data existed for analysis (that
day's files are finalized at ≈ 2026-10-24T00:01Z) and before any shadow-replay result. Only the finalized 2026-10-22
files were read, for the preliminary run of the work order's step 5. The commit time of this file records it
(76dceaa, 2026-10-23T20:41Z).

Two later additions (WO-HB3b) are the exception: rule 6's §7 close in C (the author's decision) and rule 10's fill
classes. They were fixed on 2026-10-24, after the 2026-10-23 files were finalized but before any analysis of them.
Their code is commit 59bfa59, committed 2026-10-24T09:44:04Z.

WO-HB3d came after the final analysis of both days (WO-HB3c, 2026-10-24), so it is not a reading fixed in advance.
It adds rule 10's exploratory P&L split, whose convention was chosen and checked on the data, and names the inventory
figure's hotspot band in the legend only (the in-panel text is gone). Neither changes any other number. Its code is
commit 28c4119, committed 2026-10-24T19:04:34Z.

| File | What it does |
|---|---|
| `analysis.py` | The per-bot, per-day tables (Q-A to Q-D, data description, feature shift, shadow against paper, §7 afterwards), `results.json`, the appendix table and the inventory figure |
| `shadow_replay.py` | `run`: the frozen simulator on one snapshot file (B0 as `run_b0_fair.py`, B3 as `evaluate_formula_model.py`); `compare`: one day's episode rows against the pipeline's |
| `tests/test_hb_analysis.py`, `tests/test_hb_shadow_replay.py` | Hand-computed examples, the refusals, synthetic day folders, the Jan 7 smoke runs (with `HB_ARTIFACTS`) and the import discipline |

Every path is a command-line argument; the paths below stand for the folders on the author's laptop and on Snellius. In this public copy they are placeholders: `<HB_DATA>` and `<HB_ARTIFACTS>` are the laptop's data and input folders (deploy/.env), `<HOME>` is the home folder of the machine the command runs on, `<PROJECT_DIR>` the pipeline's checkout on Snellius (with its results/ and models/), `<DATA_DIR>` the folder of the Tardis files, and `<VENV>` the Python environment.

## 1. The analysis (laptop)

Run it with the interpreter and environment of the test suites, from the repository root. Outputs go outside the
repository, one new folder per run (a folder that holds files is refused, so nothing is overwritten):

```
python -m hummingbot_adapter.analysis --data-root <HB_DATA> --days 2026-10-22 2026-10-23 --params <HB_ARTIFACTS>/hb_calibration.json --calibration-day-file <HB_ARTIFACTS>/binance_book_snapshot_25_2026-10-01_DOGEUSDT.csv --run-log hummingbot_adapter/RUN_LOG.md --out-dir <HOME>/Documents/hb_analysis/<label> [--shadow-dir <folder with the shadow replay outputs>]
```

- **Inputs, read only.** `<data-root>/hb_b0` (B0) and `<data-root>/hb_b3` (B3). A day is read only if its
  `SHA256SUMS` lists exactly its six files, every hash verifies, the bot folder's `SHA256SUMS` agrees, and each file
  starts with its `telemetry.py` header. Otherwise the run is refused (exit 2).
- **Days.** Days outside the main run (2026-10-22 and 2026-10-23) are refused unless `--dev`. With `--dev` (a
  warm-up day, for development only) everything goes to `<out-dir>/dev/`, and every table title starts with
  "DEVELOPMENT – warm-up, excluded".
- **Provenance.** `results.json` records the SHA-256 of every input. It also records whether `--params` is the
  pre-registered `hb_calibration.json` (the run's file d71af506… or the published copy ebbe6247…; see the note at the top) and whether the calibration day file is the one recorded in it
  (a9eb6459…).
- **Outputs.** `forward_test_tables.md` (all tables), one CSV per table, `results.json`, `appendix_table.tex`
  (booktabs; columns B0 22 Oct | B3 22 Oct | B0 23 Oct | B3 23 Oct), `fig_forward_inventory.pdf/.png` with the
  plotted data (`fig_forward_inventory_data.csv`), and `SHA256SUMS`.

## 2. Readings fixed by WO-HB3 where PROTOCOL §9 is silent

1. **Trading window.** The window is the steps with ts < 23:59:50 UTC of the day. "Before the close" is its last
   step: q_pre and P_pre = W are read there.
   - MaxDD before the close is the maximum over the window of (running peak of W) − W. The peak starts at W = 0 at
     00:00, as the ledger's and fast_simulate's do.
   - W is the steps file's `wealth`: cash + 100·q·S at the step's mid, from fills only, before fees, 0 at 00:00
     (`execution.Ledger`).
2. **Time weights, as fast_simulate.** fast_simulate weights each snapshot's inventory, after the fills of the
   interval that ends there, by that interval's Δt. In the same way, step i ≥ 1 of the window carries
   Δt_i = (ts_i − ts_{i−1}) / 10⁶ s (int64 µs differences, as the frozen loader) with its own q. Step 0, steps with
   Δt = 0 and steps from 23:59:50 carry no weight. This gives the mean |q| and the near-cap share (|q| ≥ 8).
3. **Q-B uses limit fills only** (`close == 0`), not the close.
   - The posting and fill edges δ_post and δ_fill are recomputed from `mid_at_post` and `mid_at_fill` and compared
     with the logged columns (mismatches above 10⁻⁹ tick are counted; 0 is expected).
   - S(t) is the mid of the last step with ts ≤ t: a step at exactly t counts.
4. **Markouts beyond the file end are dropped and counted.** A markout whose t_i + h lies after the file's last
   step is dropped. Its count is reported per horizon. R(h) = Σ m_i(h) / Σ δ_post,i sums over the fills whose
   markout at h is kept, so the numerator and the denominator cover the same fills. The mean m(h) is over the same
   fills.
5. **E has fast_simulate's timing.** fast_simulate exposes the quote of snapshot t−1 over dt[t]. In the same way, the
   orders resting after step i−1 (`resting_bid`, `resting_ask`) meet Δt_i, at their distance δ from step i−1's mid
   in USDT: 1 − exp(−A e^(−κδ) Δt_i). The intervals are the window's (rule 2), and A and κ come from
   `hb_calibration.json`.
   - An order cancelled inside an interval (a §7 pause) counts for the whole interval, as in the simulator.
   - The paper fills compared with E are the window's limit fills.
6. **Q-C.**
   - F is the notional of the day's limit fills (a count of limit fills after the last window step is reported;
     0 is expected).
   - C is the notional of every market close of the day, with the fill prices rounded to the tick. That is the day's
     close (the 23:59:50 close, or the close of a position carried past 00:00, §5.4), and any §7 close (halt or
     drawdown) inside the day. The purpose comes from the order's submit row: `day_close`, `carry`, `halt` or
     `drawdown`.
   - P_post is W after the day's last fill. For a day that ends flat this is its cash, −Σ s·p·a over all its fills
     in the ledger's own Decimal arithmetic. It equals W at the last step whenever a step follows the close, and it
     stays right when none does (a close still in flight at 00:00, or a carry). A day that does not end flat keeps
     W at its last step and is flagged.
   - The paper fees are USDT fees + DOGE fees × the fill price.
   - A close's cost is its half-spread against the mid at its submission, plus its fee. The day's close and a §7
     close are reported apart.
   - **A §7 close inside the day (the author's decision of 2026-10-24, WO-HB3b).** This covers a halt or drawdown
     market order before 23:59:50.
     - It is part of C: its notional joins the day's close notional in P_post − f·(F + C) and f*_post.
     - F stays the limit fills, so P_pre − f·F and f*_pre are unchanged.
     - The bot-day is flagged (`s7_close_in_C`) and the close is also listed apart (`s7_close_*`).
     - The decision was fixed before any analysis of 2026-10-23. Its code is commit 59bfa59, committed
       2026-10-24T09:44:04Z.
     - Day 2026-10-22 has no §7 close, so none of its numbers depends on this.
7. **§7 afterwards uses the realized path of every step of the day**, as the bot's own §7 check does.
   - The notional at the cap is 1,000 DOGE × the day's first mid.
   - Net equity is W − f × (the notional of the limit fills with ts ≤ the step's ts).
8. **The shadow replay runs on the whole snapshot file**, to its end; the paper window ends at 23:59:50. The PIT
   compares the paper's P_pre/100, MaxDD/100 and mean |q| with the shadow's `Final PnL`, `Max DD` and `Mean |q|`:
   §9's PIT set.
   - Ties count ½.
   - The 5–95 % range is numpy's percentile (linear interpolation).
   - The other Q-A quantities appear beside them, because §9 Q-A asks for them "in the paper run and in the shadow
     replay":
     - the near-cap share against the shadow's `Near Cap Fraction`, with a PIT labelled exploratory;
     - q before the close against the shadow's `Terminal q`. The shadow never closes, so its q is at the end of the
       file. This row is descriptive, with no PIT, and shows the share of shadow paths at q ≤ −8.
   - The analysis refuses a shadow replay unless its record names that bot's own snapshots file (SHA-256), the same
     `--params`, the pipeline's 2 × 20 paths and the episode file's own SHA-256.
9. **Q-D.**
   - Downtime is the step gaps over 30 s, including the stretch from 00:00 to the first step and from the last step
     to 24:00. Uptime = 1 − downtime / 24 h.
   - Quoting-paused minutes are the union, inside the window, of the §7 pause intervals: pause → resume per trigger,
     and a drawdown pause or a halt until its flag clears.
   - New orders are every submit (quotes and closes). The busiest rolling 10 s is counted over (t − 10 s, t], as the
     bot's rate limiter counts it.
   - Websocket reconnects are the feed's `ws_closed` events; the re-subscriptions and the `ws_error` retries are
     listed beside them.
   - The bot's own `day_end` counters are shown beside the recomputed counts, as a check.
   - A day with more than 1 h of downtime is flagged in every table (`downtime_flag`, PROTOCOL §8).
   - **The RUN_LOG entries.** The table lists every entry written from 2026-10-22T00:00Z on. An entry about a day is
     written after it, so there is no upper bound. Each entry is listed under every analysed day whose date appears
     in its stamp or its text. An entry that names one bot only is listed for that bot; any other is listed for
     both. The CSV and `results.json` keep the full text.
10. **Exploratory** (labelled so; not in §9):
    - **At-price trade fills.** For each limit fill, take the public trades (`trades_*.csv`, taker side) logged after
      its order's post and in the 0.5 s up to the fill, on the side opposite to ours (taker sells for our bid, taker
      buys for our ask).
      - The fill is `at_price` if none of them printed strictly through the order price but one printed exactly at
        it.
      - It is `through` if one printed strictly through.
      - It is `no_trade` if none reached the price: a book cross at a clock tick, or a trigger outside the window.
      - Prices are compared in whole ticks.
      - PROTOCOL §3 describes trade fills as "strictly through". The paper engine applies that test, but it compares
        the Decimal order price with the float trade price exactly. An at-tick print whose float lies below a bid
        (or above an ask) is therefore "through" for the engine and can trigger the fill.
      - The `at_price` fills are split into `at_price_engine_through` (such a print exists) and
        `at_price_not_engine_through` (none does). In the second case the fill came from something else, e.g. a
        book cross at a clock tick.
    - **Q-B by fill class** (WO-HB3b, table `qb_fill_classes`; fixed on 2026-10-24 before any analysis of
      2026-10-23, code commit 59bfa59).
      - For each class of the at-price method, Q-B is repeated with its own definitions: count, mean δ_post,
        mean δ_fill, and per horizon the mean m(h), R(h) and the dropped markouts.
      - The same is done for all limit fills (it equals Q-B) and for the through and no_trade fills together.
      - For those two aggregates the table gives the window's fills over E.
      - The through + no_trade ratio is the author's sensitivity: it drops every at-price fill. Only the
        `at_price_engine_through` part of its difference to the all-fills ratio can come from the engine's
        Decimal-against-float comparison, and that part is an upper bound. The `at_price_not_engine_through` fills
        came from something else.
      - A hypothesis to report, not to tune on: δ_fill is about +½ tick for at-price fills and ≤ −½ tick for
        no_trade (book-cross) fills. A positive δ_fill elsewhere would reflect the depth stream (100 ms batches)
        lagging the trade stream.
    - **The P&L split** (WO-HB3d, table `pnl_split`; defined after the final analysis).
      - Over the trading window, P_pre = fill edge + inventory, in USDT for 100-DOGE orders.
      - Inventory = Σ over the window's steps i ≥ 1 of 100·q_{i−1}·(S_i − S_{i−1}).
      - Fill edge = Σ over the limit fills of s·a·(S − p), with S the mid of the first step whose wealth includes the
        fill. S is that step's mid, not `mid_at_fill` (δ_fill's book mid at the fill event).
      - **The convention: a fill enters the first step with ts ≥ its own.** A step marks W = cash + 100·q·S over
        every fill booked before it (`Ledger.mark`, before the step's quotes are placed), and its ts is read when its
        tick starts. The paper connector books fills outside the bot's tick, at the step clock's reading, and the step
        clock never goes back. So a fill's ts is never later than that step's.
      - A fill at a step's own ts is taken to have been booked before that step. This holds while the step clock
        advances between a step and a later fill. It fails only while ts is held (at a restart floor ahead of the wall
        clock): such a fill would be put on a step whose wealth excludes it, and the checks below flag the day. The
        table counts these fills (`fills_at_step_ts`). On 2026-10-22 and 2026-10-23 there are none, and no restart.
      - A day starts flat with zero cash (`Ledger.start_day`), so a fill before the day's first step enters step 0 and
        no term precedes it. The 23:59:50 close and a limit fill after the last window step are left out (they are in
        P_post).
      - Then W_i − W_{i−1} = 100·q_{i−1}·(S_i − S_{i−1}) + the edges of the fills step i adds, up to the bot's float
        rounding of W, with the logged q, the position of the step's wealth. On 2026-10-22 and 2026-10-23 every step
        row has W = cash + 100·q·S bit for bit, and no fill shares a step's ts.
      - A market fill inside the window (a §7 close) is a third part, so the parts sum to P_pre on any day. The two
        days have none.
      - The checks, each within 10⁻⁹ USDT: the parts' sum minus P_pre, and the largest |W_i − W_{i−1} − step i's
        parts| over the window. The table also counts the window steps whose 100·q differs from the cumulative fills
        (0 expected) and gives the fill edge per DOGE in ticks.
    - **The mobile-hotspot window**, 2026-10-22T22:47Z → 2026-10-23T05:39Z (RUN_LOG 2026-10-23T05:49:24Z). The clock
      checks and the step gaps are split into inside and outside it; a gap belongs to the part where it starts.
    - **Book and clock events.**
      - Book re-syncs (applied REST snapshots, clarification h) and the snapshots ahead of the diffs.
      - Snapshot errors.
      - Crossed or locked books, from the bot's `day_end` counters. Its `crossed_book` health rows stop at 100 a
        day; their count is shown beside it.
      - Re-anchors (`clock_jump`) and `clock_backward`.
      - The round-away quotes (clarification a: steps and orders).

The data description uses the frozen `load_single_day_calibration_arrays` on each bot's snapshot file and on the
2026-10-01 file. σ comes from `cvar_mm.gym.calibration._sigma_from_day` with 600 s blocks. The share of 1-tick spreads
is over rows.

## 3. The shadow replay (Snellius; never a full day on the laptop)

Snellius runs only on compute nodes, and Claude Code has no access, so the author runs these commands by hand.
`run` refuses unless:
- `configs/final_pipeline.env` is exported (FORMULA_SKEW_TICKS_MAX = 5, EVALUATION_FILL_SEEDS = "42 43",
  EVALUATION_ROLLOUTS = 20);
- for B0, `--gamma-file` holds exactly 0.8908584807525282 (SHA-256 34cbb018…);
- for B3, the policy is 05ee4c97… and the normalizer d2be858e….

`--params` takes `hb_calibration.json` (σ, A, κ) or `formula_train_params.json` (σ_train, A_train, κ_train).
Outputs are never overwritten. The out-dir may not lie inside the repository or the day file's folder, and may hold
nothing but shadow-replay outputs and their `SHA256SUMS`.

**Before the replay.**
1. Bring the clone to this branch: `git fetch origin && git checkout feat/hummingbot-analysis`, once the WO-HB3
   commits are pushed. The new modules and `hummingbot_adapter/config/hb_calibration.json` come with it.
2. Copy each finalized day folder unchanged from the laptop to `<HOME>/hb_forward_data/<bot>/<date>/`.
   For example, from the laptop: `scp -r <HB_DATA>/hb_b0/2026-10-22 <your Snellius login>:<HOME>/hb_forward_data/hb_b0/`.
   Only the snapshots file is needed, but the whole folder keeps its `SHA256SUMS`.
3. Verify the copy there: `cd <HOME>/hb_forward_data/hb_b0/2026-10-22 && sha256sum -c SHA256SUMS`.
4. Repeat steps 2–3 for hb_b3, and for 2026-10-23 once it is finalized.

`--shadow-dir` needs a shadow replay of every bot and day the analysis covers.

```
# on a compute node, e.g. srun --partition=rome --ntasks=1 --cpus-per-task=4 --time=08:00:00 --pty bash
source <VENV>/bin/activate
export PYTHONNOUSERSITE=1; unset PYTHONPATH
cd <the clone, checked out at feat/hummingbot-analysis>
source configs/final_pipeline.env
R=<PROJECT_DIR>
OUT=<HOME>/hb_forward_analysis

# Jan 7 2025 parity: the replay of the pipeline's own day, with the Jan 1-6 calibration
python -m hummingbot_adapter.shadow_replay run --model b0 --day-file <DATA_DIR>/binance_book_snapshot_25_2025-01-07_DOGEUSDT.csv --date 2025-01-07 --params $R/results/formula_train_params.json --gamma-file $R/results/b0_gamma_fixed.txt --out-dir $OUT/jan7_parity
python -m hummingbot_adapter.shadow_replay run --model b3 --day-file <DATA_DIR>/binance_book_snapshot_25_2025-01-07_DOGEUSDT.csv --date 2025-01-07 --params $R/results/formula_train_params.json --model-zip $R/models/ppo_b3_doge.zip --vecnorm $R/models/vecnorm_b3.pkl --out-dir $OUT/jan7_parity
python -m hummingbot_adapter.shadow_replay compare --ours $OUT/jan7_parity/shadow_b0_2025-01-07_episodes.csv --reference $R/results/b0_test_episodes.csv --day 2025-01-07
python -m hummingbot_adapter.shadow_replay compare --ours $OUT/jan7_parity/shadow_b3_2025-01-07_episodes.csv --reference $R/results/eval_parts/retrospective_holdout/b3/episodes_day_2025-01-07.csv --day 2025-01-07

# a prospective day, per bot, on that bot's own snapshots file, with the pre-registered calibration
python -m hummingbot_adapter.shadow_replay run --model b0 --day-file <HOME>/hb_forward_data/hb_b0/2026-10-22/snapshots_2026-10-22.csv.gz --date 2026-10-22 --params hummingbot_adapter/config/hb_calibration.json --gamma-file $R/results/b0_gamma_fixed.txt --out-dir $OUT/shadow
python -m hummingbot_adapter.shadow_replay run --model b3 --day-file <HOME>/hb_forward_data/hb_b3/2026-10-22/snapshots_2026-10-22.csv.gz --date 2026-10-22 --params hummingbot_adapter/config/hb_calibration.json --model-zip $R/models/ppo_b3_doge.zip --vecnorm $R/models/vecnorm_b3.pkl --out-dir $OUT/shadow
# and the same two lines for 2026-10-23, once its folders are finalized, copied and verified
```

**The parity's reference files.** They are derived from `scripts/pipeline/common.py`: `seeded_result_name` for B0,
and the B3 eval-part directory through `eval_part_dir` (`eval_episode_path`). `shadow_replay.reference_episode_paths`
returns the same paths. Check them with `ls -l`:

- B0: `<PROJECT_DIR>/results/b0_test_episodes.csv` (all 23 test days; `compare` selects
  Day == 2025-01-07, 40 rows).
- B3: `<PROJECT_DIR>/results/eval_parts/retrospective_holdout/b3/episodes_day_2025-01-07.csv`
  (40 rows).

**The parity check.**
- `compare` matches the rows on (Fill Seed, Rollout).
- It requires the same columns, in the same order, and the same keys.
- It compares every numeric column: Sharpe, Sortino, Max DD, P&L-to-MAP, Final PnL, Mean |q|, Terminal q,
  Mean spread, the two snapshot ratios and Near Cap Fraction.
- It exits 0 when every difference is within 10⁻⁹ relative or 10⁻¹² absolute. Otherwise it exits 1 and prints the
  rows.

**The episode file.** The shadow replay writes the pipeline's episode header, column for column:
`Day,Fill Seed,Rollout,Sharpe,Sortino,Max DD,P&L-to-MAP,Final PnL,Mean |q|,Terminal q,Mean spread,Sharpe Snapshot (supplementary),Sortino Snapshot (supplementary),Near Cap Fraction`.
The B3 file has the same header: `evaluate_formula_rl_per_day` builds its rows with `stats_dict_to_frame`, as
`run_b0_fair.py` does.

**The JSON record.** Each run's JSON records:
- the git commit and the SHA-256 of the day file, the params and the gamma file or the policy and normalizer;
- the parameters used;
- the numpy, pandas, torch, SB3 and gymnasium versions;
- the host, the start and end UTC and the wall time.

The out-dir's `SHA256SUMS` is rebuilt after every run. For the PIT, copy the out-dir back to the laptop unchanged and
pass it as `--shadow-dir`.
