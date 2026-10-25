# Beyond Reward Shaping: CVaR-Constrained Reinforcement Learning for Avellaneda–Stoikov Market Making

MSc thesis in Computational Science, University of Amsterdam, 2026.

> **Status:** release in preparation. Version 1.0.0, with the code, tests, documentation and result tables, is planned for 27 October 2026.

## The question

A market maker quotes a bid and an ask around the mid-price. It earns the spread but carries inventory risk. In the Avellaneda–Stoikov (2008) model, a risk-aversion parameter γ sets these quotes. Here a reinforcement-learning agent (PPO) chooses γ and a quote skew at every decision step, while the Avellaneda–Stoikov formula always builds the quotes.

The thesis asks whether a **constraint** on tail drawdown risk works better than a **penalty** on drawdowns in the reward.

| Model | Description |
|---|---|
| B0 | Analytical Avellaneda–Stoikov with a fixed γ tuned on the training days |
| B1 | Unconstrained PPO choosing γ and the skew |
| B2 | PPO with a drawdown penalty in the reward (penalty-based reward design) |
| B3 | PPO-Lagrangian with a constraint on the CVaR₀.₂ of the daily maximum drawdown |
| B1⁺ | Control: B1 fine-tuned like B2 and B3, but without a risk term |

## Data and evaluation

- Binance Spot DOGEUSDT order-book snapshots (Tardis.dev `book_snapshot_25`), 1–29 January 2025, replayed with a best-price-reach execution proxy.
- Training and model selection: 1–6 January. Retrospective out-of-sample test: 7–29 January (23 days).
- The trading day is the unit of inference. The primary test is a within-day randomization test with a Holm correction.
- The data is licensed and not included in this repository.

## Main findings

- **No evidence that the CVaR constraint changes tail drawdown risk.** The CVaR₀.₂ of the daily maximum drawdown is 0.163 USDT for B3, 0.165 for B1 and 0.157 for B2. Holm-adjusted p = 0.496 (B3 vs B1) and 0.335 (B3 vs B2). B3 did not reach its constraint threshold within its 10 fine-tuning iterations.
- **The drawdown penalty (B2) helps a little.** Against B1 it has about 8 % higher Sharpe and 12 % less inventory (secondary tests, Holm-adjusted p ≤ 0.016).
- **The analytical baseline B0 beats every PPO variant.** It is best on all 23 days for Sharpe, Sortino, maximum drawdown and P&L-to-MAP, and its CVaR₀.₂ is 0.0505 USDT, about a third of theirs. Its policy lies inside the PPO action space, so PPO did not find a policy that was available to it.
- **Paper-trading forward test.** B0 and B3 ran on Hummingbot 2.17.0 against live Binance Spot DOGE-USDT data on 22–23 October 2026, for two days without halts or restarts. Paper trading delivered about a fifth (21–24 %) of the fills that the simulator's execution proxy expected.

## License

BSD 3-Clause; see [LICENSE](LICENSE).
