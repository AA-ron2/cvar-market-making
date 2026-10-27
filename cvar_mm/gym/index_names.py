# Parts adapted from mbt-gym (Copyright (c) 2018, Joseph Jerome; BSD-3-Clause); see THIRD_PARTY_NOTICES.md.
"""
Fixed column indices for the flat state vector.

Layout:  [cash, inventory, time, midprice]
         idx 0     1        2       3

Convention follows mbt-gym (Jerome et al., 2023, §3):
    state[0] = cash  (w)
    state[1] = inventory  (q)
    state[2] = time  (t, in seconds from session start)
    state[3] = asset midprice  (S)

Action layout for limit-order market making:
    action[0] = bid depth  (δ_bid = S − p_bid,  ≥ 0)
    action[1] = ask depth  (δ_ask = p_ask − S,  ≥ 0)
"""

# ── state indices ─────────────────────────────────────────────
CASH_INDEX = 0
INVENTORY_INDEX = 1
TIME_INDEX = 2
ASSET_PRICE_INDEX = 3

# ── action indices ────────────────────────────────────────────
BID_INDEX = 0
ASK_INDEX = 1
