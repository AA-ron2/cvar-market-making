# Market data

The market data is **not** in this repository. It is licensed third-party data and may not be redistributed.

## What the pipeline expects

| Item | Value |
|---|---|
| Data | Order-book snapshots in CSV, one file per UTC day, with a header row |
| Content | The top 25 bid and ask levels (price and amount); one row each time any of those levels changes |
| Venue and instrument | Binance Spot, DOGEUSDT |
| Columns | `timestamp` (the exchange time, integer microseconds since the Unix epoch, read as exact text) and `local_timestamp`; then the level columns. The loader and the features read `asks[0].price` and `bids[0].price` (the mid, the spread and the calibration) and `bids[i].amount` and `asks[i].amount` for i = 0…4 (the order-book imbalance); the other level columns, `asks[i].price`, `asks[i].amount`, `bids[i].price` and `bids[i].amount` up to i = 24, are loaded but not used. The header runs `exchange`, `symbol`, `timestamp`, `local_timestamp`, then the four columns of level 0, of level 1 and so on (the layout `hummingbot_adapter/telemetry.py` writes) |
| Period | 2025-01-01 to 2025-01-29, one UTC day per file (29 files) |
| File name | `binance_book_snapshot_25_YYYY-MM-DD_DOGEUSDT.csv` (decompressed) |
| Clock | `timestamp`, the exchange time in µs, is the clock; `local_timestamp` is diagnostic only |
| Location | this folder by default; set `DATA_DIR` to use another folder |

The default location is this `data/` folder (`cvar_mm/gym/experiment_config.py`); the `DATA_DIR` environment variable
overrides it. The pipeline on Snellius requires `DATA_DIR` to be exported (`scripts/pipeline/README.md`).

## Checking your copy

`results/final/profiling/formula_manifest.csv` lists the 29 files the thesis used, with their dates, their role in the
split (Jan 1–4 candidate training, Jan 5–6 validation, Jan 7–29 test) and the SHA-256 of each file. A copy with other
bytes (for example a different download or decompression of the same day) is a different input.

## Known data-quality facts

- Exchange-timestamp inversions: Jan 9 (737 µs), Jan 10 (817 µs) and Jan 23 (21,042 µs). The loader handles them with a
  stable sort of whole rows (F-01 in [docs/design_decisions.md](../docs/design_decisions.md)); all 29 days load.
- The tests that need these files skip when `DATA_DIR` is unset; every other test runs without data.
