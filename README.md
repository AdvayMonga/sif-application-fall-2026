# SIF Application, Fall 2026

604,578 Polymarket wallets. Telling traders with a real edge apart from traders who got lucky.

**Report:** `deliverables/SIF_Application_Fall_2026.pdf` (built by `analysis/ml_page.py`, not committed)

## What is here

1. **A grouping of the wallets by how they trade**, using no profit data. It separates machines, reward farms, wallets that went bust, and ordinary traders.
2. **A measurement of which groups actually make money.** Machines are the only group with a real edge (+0.85¢ per dollar wagered, $20.3M profit). Farms sit at zero because their trades carry no risk.
3. **`sifeval`**, a tool that takes any track record and reports how much of it is skill and how much is luck.

## The tool

```bash
python -m sifeval eval examples/sample_trades.csv            # a file of trades: price, size in dollars, won (0/1)
python -m sifeval returns examples/sample_portfolio_values.csv   # daily account values, or a "return" column
python -m sifeval wallet 0x2728d99B... --data data.parquet   # any wallet in the dataset
python -m sifeval eval trades.csv --json                     # machine-readable output
```

Example:

```
TRACK RECORD  ·  2,072 trades  ·  $2,501,000 wagered  ·  P&L $+1,103,256
  realized edge         +44.11¢ per dollar   (+22.32¢ per share)
  luck-adjusted edge    +22.33¢ per share   90% CI [+20.91¢, +23.31¢]
  P(edge > +0.5¢/share) 100.0%      P(edge < −0.5¢/share) 0.0%
  verdict               skilled: edge above +0.5¢/share with P > 0.95
  trades for 95% proof  0 more (≈7 total)
```

### How it separates skill from luck

It never models the market. It uses how a bet pays: buy a share at price `p` and you either gain `1-p` or lose `p`.

- If the price is a fair probability, profit per share swings by `p(1-p)` per bet, and by `p(1-p)/n` over a record. A bet at 50¢ swings eight times more than a bet at 97¢.
- The multiplier is measured, not assumed: on the 75,855 wallets that made exactly one bet, the real swing came to `0.87 · p(1-p)`.
- Subtracting that swing from the spread of results across all wallets leaves the spread of real skill. 91% of wallets sit within one cent per share of zero. That distribution ships as `sifeval/reference.json` (16 KB), so the tool runs without the dataset.

Combining the three gives the luck-adjusted edge, a 90% range, the chance the edge is real, and how much more trading would settle the question.

### What it reports

| input | returns |
|---|---|
| any record | raw result, luck-adjusted result, 90% range, chance the edge is real, record still needed, percentiles against the 124,064 wallets with 20+ trades |
| a file of trades | plus: prices fair or not, whether bigger bets did better, how much profit rests on one trade, first half against second half, chance of a large loss |
| daily account values | plus: Sharpe, first half against second half, longest losing run against chance, autocorrelation, drawdown odds, alpha and beta against a benchmark |

## Layout

| path | what |
|---|---|
| `sifeval/` | the tool. `core.py` scores a record, `features.py` turns trades or a dataset row into inputs, `returns.py` handles P&L series, `diagnostics.py` the extra checks, `reference.json` the population prior |
| `analysis/` | how everything was produced, in run order: `common.py` decodes the raw columns, `skill_luck.py` removes luck from the population, `build_reference.py` writes the tool's prior, `ml_features.py` builds 26 behaviour measures, `ml_active.py` groups the wallets, `cluster_eval.py` scores each group, `ml_labels.py` checks the dataset's own labels, `ml_skill.py` trains and checks the model, `ml_page.py` builds the report |
| `tests/` | pytest for the tool |
| `examples/` | sample inputs, both synthetic and labelled as such |
| `data/` | SIF Live's public dashboard snapshot, used by the report |

`data.parquet`, `analysis/out/` and `deliverables/` are not committed.

## Reproducing

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest tests          # the tool

cd analysis                               # the analysis, in order
../.venv/bin/python skill_luck.py         # remove luck from the population
../.venv/bin/python build_reference.py    # write sifeval/reference.json
../.venv/bin/python ml_features.py        # 26 behaviour measures per wallet
../.venv/bin/python ml_active.py          # group the wallets
../.venv/bin/python cluster_eval.py       # score each group
../.venv/bin/python ml_labels.py          # check the dataset's own labels
../.venv/bin/python ml_skill.py           # train and check the model
../.venv/bin/python ml_page.py            # build the report
```

Put `data.parquet` in the repo root first. Scripts write to `analysis/out/`.
