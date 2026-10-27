# Running the forward test (B0 and B3 on binance_paper_trade)

PROTOCOL.md is the rule; this file is how to operate the two bots. Commands run in PowerShell in
`hummingbot_adapter\deploy`. All times are UTC.

## Layout

| Host path | In the container | What |
|---|---|---|
| repository root | `/repo` (read-only, `PYTHONPATH`) | the adapter and the frozen `cvar_mm/` code |
| `deploy\scripts` | `/home/hummingbot/scripts` (read-only) | `hb_forward_bot.py` (imports `hummingbot_adapter.hb_forward_bot`) and an `__init__.py`, which stops the repository's own `scripts/` package from shadowing this folder |
| `deploy\conf\scripts` | `/home/hummingbot/conf/scripts` (read-only) | the bot configs: `conf_hb_b0.yml`, `conf_hb_b3.yml` (production) and the `_smoke` variants |
| `HB_ARTIFACTS` | `/artifacts` (read-only) | `hb_calibration.json`, `b3_policy.npz` + `.json`, `b0_gamma_fixed.txt` |
| `HB_DATA` | `/hb_data` | the run outputs: `<bot>\<date>\` (six CSV files per day, gzipped with `SHA256SUMS` at the day's end), `<bot>\state.json`, `<bot>\SHA256SUMS` |
| `HB_DATA\<bot>\hummingbot\{conf,data,logs}` | `/home/hummingbot/{conf,data,logs}` | Hummingbot's own files: `conf_client.yml` (copied from `deploy\conf\<bot>`; Hummingbot rewrites the copy at each start), the password check, its sqlite and its log `logs_conf_<bot>.log` |

A smoke run uses `HB_DATA\smoke\` for all of these. The bot refuses to start if its mode and `HB_RUN_SUBDIR` do not
match, if a production setting differs from PROTOCOL.md, or if an input hash does not match: the strategy then
fails to load and the container restarts, with the error only in Hummingbot's log
(`HB_DATA\<bot>\hummingbot\logs\logs_conf_<bot>.log`), not in `docker logs`. If restoring the state fails (a state
of other settings, a finalized day, a damaged file), the bot writes `start_refused.txt` beside the state, never
changes `state.json`, and stays inert until it is fixed and restarted; `status.ps1` prints the refusal. Each start
first copies the state to `state.prev.json`.

**The calibration.** Put the published `hummingbot_adapter/config/hb_calibration.json` in `HB_ARTIFACTS`:
`verify_inputs` (`bot_config.py`) refuses to start unless the copy in `HB_ARTIFACTS` and the committed one both have
SHA-256 `ebbe6247…` (`CALIBRATION_SHA256`). The forward-test run of 2026-10-22 and 2026-10-23 used the file with
`d71af506…` (`CALIBRATION_SHA256_RUN`); the published copy differs from it only in two local paths, so σ, A and κ
are the same.

## Once

1. Copy `.env.example` to `.env` and set `HBOT_PASSWORD` (any string; the first start stores a check of it, later
   starts must use the same one). `.env` is git-ignored: never commit it.
2. Keep the machine awake for the whole run, on mains power. In the WO-HB2 trial a 4.7 h Windows sleep stopped the
   containers (no data, no quotes) and froze the Docker VM's monotonic clock; the bot now re-anchors on the wall
   clock after such a gap and logs it as a `clock_jump` deviation, but the hours asleep are lost. Run, as
   administrator: `powercfg /change standby-timeout-ac 0`, `powercfg /change hibernate-timeout-ac 0`, and set the
   lid-close action to "Do nothing" (Control Panel > Power Options > Choose what closing the lid does). Also
   resynchronize the clock once before the warm-up: `w32tm /resync /force`.
3. `.\prepare.ps1` (production) or `.\prepare.ps1 -Smoke` (smoke) creates the folders and copies `conf_client.yml`.

## Start, status, stop, restart

- Start both: `docker compose up -d`. One bot: `docker compose up -d hb_b3`.
- Status: `.\status.ps1` (smoke: `.\status.ps1 -Smoke`). It prints, per bot: the container's state and uptime,
  seconds since the last step, q, the day's gross P&L and drawdown, paused/halted, trigger counts, rate-cap skips,
  round-away counts, and the free disk on C:.
- Stop both: `docker compose stop`. The bot cancels its quotes, saves its state and closes its files (graceful
  stop, up to 60 s). Do not use `docker kill`: a kill loses up to one second of state, which the restart then
  treats as a gap.
- Restart one bot: `docker restart hb_b0`. Within the same UTC day the bot resumes with the same S0, rolling window,
  ledger, DOGE reference and paper balances; the first step after the gap carries the gap as its Δt (PROTOCOL
  §5.4). If it was down across 23:59:50 UTC, it closes the carried position at market at its first step and logs a
  deviation.
- Every incident, restart, halt and manual action goes into `hummingbot_adapter\RUN_LOG.md` (PROTOCOL §8).
- Step clock (clarification i): ts_us = the wall clock at the anchor + CLOCK_MONOTONIC_RAW since it (the Docker VM's
  CLOCK_MONOTONIC is NTP-slewed and ran 0.01-0.03 % fast on 2026-10-21). Clarification j: the RAW time since the
  anchor is divided by 1 + c·1e-6, with c = `CLOCK_RAW_RATE_PPM` in `execution.py` (RAW's rate against Binance, a
  decimal string applied exactly); the anchor is the uncorrected wall clock. Each clock check also logs
  `raw_offset_s`, the uncorrected clock's offset, from which c is re-estimated.
- No quote while halted or paused (clarification m): every quote post checks the §7 state when it goes out, so a
  halt or pause raised inside a re-quote's cancel (the paper connector runs the cancel's events inside the call)
  posts nothing afterwards (an orders row `skip`, reason `not_quoting`); the closes (halt, drawdown, daily) go out
  as before.
- Day boundary (clarification k): the 23:59:50 close and the day switch follow the wall clock; the new day is
  anchored on the wall clock with no floor, so a lead or lag of the step clock is dropped at the switch (logged in the
  `clock_anchor` health row as `lead_dropped_s`), not carried; ts stays inside the UTC day of the files it is written
  to (held at 23:59:59.999999 while the step clock is ahead of the wall clock's 00:00, or a carried close runs), and
  events the paper connector fires inside a tick share that tick's ts. The daily order cap counts in the wall
  clock's UTC day.
- Clock re-anchors (clarification g, forward only since i): when the wall clock is more than 2 s ahead of the step
  clock (a sleeping host stops the Docker VM's clocks), the bot re-anchors on the wall clock. When the step clock is
  more than 2 s ahead of the wall clock, it never moves ts back: it logs a `clock_backward` incident and leaves it to
  the §7 offset check against Binance. Both write a ready RUN_LOG.md line (UTC time, bot, what happened, minutes
  affected) to `HB_DATA\<bot>\run_log_incidents.md`; the repository is
  read-only in the containers, so append them from the repository root with
  `python -m hummingbot_adapter.run_log --data-root <HB_DATA>` (standard library only;
  `--dry-run` shows what would be appended; each line is appended once) and commit RUN_LOG.md ([documentation]).
  `status.ps1` prints the re-anchor and backward counts, the latest clock offset (with the uncorrected RAW and the
  wall clock's offsets), and the step-clock and raw
  monotonic rates against Binance over at least an hour (a diagnostic, empty until a check an hour old exists).
  Each check uses the request with the smallest round trip of 5, discarding round trips above 1 s.
- Book re-sync (clarification h): `hb_forward_bot.py` makes Hummingbot re-snapshot the Binance book from REST every
  60 s instead of every hour (Hummingbot 2.17 does not re-snapshot after a websocket reconnect); the bot refuses to
  start unless the paper connector's data source sees 60 s. Each applied snapshot is a `book_snapshot` row in the
  health log (update id, and the book's last applied diff id when the bot sees it), counted in `status.ps1`; failed
  REST re-syncs are counted as `snapshot_errors`. Caveat (WO-HB2 review): Hummingbot 2.17's
  `restore_from_snapshot_and_diffs` replays its whole 32-diff window onto the snapshot, older diffs included, so a
  level touched in the last few seconds can briefly come back stale after a re-sync; `snapshots_ahead_of_diffs`
  counts the snapshots seen with no newer diff applied yet (the exposed case). Crossed or locked tops of book are
  counted too (no step is taken on those), but Hummingbot already trims crossed levels after each diff, so a count
  of 0 is no evidence that the book is consistent.

## Clear a halt

A halt (PROTOCOL §7) cancels all orders, closes the position at market and posts nothing until the flag is cleared
by hand; a restart does not clear it. To clear it, after reading the bot's `health_<date>.csv`:

```powershell
docker stop hb_b3
docker compose run --rm --no-deps --entrypoint python hb_b3 -m hummingbot_adapter.clear_halt /hb_data/hb_b3/state.json --by "<name>" --note "<why>"
docker start hb_b3
```

(smoke: `/hb_data/smoke/hb_b3/state.json`). The tool refuses while the state is less than 10 s old, and records the
clearance in `halt_clearances.csv` beside the state. Log the clearance in `RUN_LOG.md`.

## Smoke run (WO-HB2 step 4)

Set `HB_RUN_SUBDIR=smoke/`, `HB_B0_CONF=conf_hb_b0_smoke.yml` and `HB_B3_CONF=conf_hb_b3_smoke.yml` in `.env`, run
`.\prepare.ps1 -Smoke`, then `docker compose up -d`. For a one-off config, set it for one command:
`$env:HB_B3_CONF = "conf_hb_b3_smoke_qcap.yml"; docker compose up -d hb_b3; Remove-Item Env:HB_B3_CONF`.
Before production: stop both, archive `HB_DATA\smoke`, and restore the production values in `.env`.
