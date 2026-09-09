# SIF Application — Fall 2026

Exploratory analysis of a 604,578-wallet Polymarket census (June 2024 – March 2025), plus **`trackrecord`**, a small tool that answers the question the analysis kept running into: *is this track record skill or luck?*

## What's here

| path | what |
|---|---|
| `analysis/` | the analyses behind the report: column decoding (`common.py`), skill-vs-luck deconvolution (`skill_luck.py`), crowd calibration (`crowd.py`), extreme-favorite edge (`edge.py`), toxic-fill model (`toxic.py`), ML skill-from-behavior pipeline (`ml_*.py`), and the memo/figure builders |
| `trackrecord/` | the eval harness: luck-adjusted edge with a confidence interval, probability the edge is real, trades needed for proof, and percentiles against the 124k active wallets |
| `tests/` | pytest for the harness |
| `examples/` | synthetic sample inputs: a 60-trade record and a 120-day portfolio-value series (both marked as examples) |

Data (`data.parquet`), generated outputs (`analysis/out/`) and report deliverables are not committed.

## The harness

```bash
python -m trackrecord eval examples/sample_trades.csv          # a CSV of trades: price, size (dollars), won (0/1) [, pnl]
python -m trackrecord wallet 0x2728d99B... --data data.parquet  # any wallet in the census
python -m trackrecord returns equity.csv                       # any P&L series: a "return" column, or a "value" column (portfolio equity)
python -m trackrecord eval trades.csv --json                    # machine-readable
```

Example output:

```
TRACK RECORD  ·  2,072 trades  ·  $2,501,000 wagered  ·  P&L $+1,103,256
  realized edge         +44.11¢ per dollar   (+22.32¢ per share)
  luck-adjusted edge    +22.33¢ per share   90% CI [+21.00¢, +23.50¢]
  P(edge > +0.5¢/share) 100.0%      P(edge < −0.5¢/share) 0.0%
  verdict               skilled: edge above +0.5¢/share with P > 0.95
  trades for 95% proof  0 more (≈7 total)
  vs 124,064 active wallets:  edge/$ 99th pct  ·  sizing dispersion 99th pct  ·  trades 99th pct
```

Returns mode grades a series of per-period returns with a plain t-test (flat prior): annualized return with a 95% band, Sharpe, P(true mean > 0), and how many more periods are needed for 95% significance at the current Sharpe.

How it works: every record is treated as *true edge + noise*, where the noise shrinks with the number of trades and depends on the prices traded (variance ≈ 0.87·p(1−p)/n, calibrated on 75k resolved single bets). The prior over true edge is the population's own distribution, recovered by nonparametric empirical Bayes from the census (`analysis/skill_luck.py`) and shipped as `trackrecord/reference.json`. The posterior gives the luck-adjusted edge, its interval, and how many more trades would be needed before the observed edge is distinguishable from luck.

## Reproducing the analyses

```bash
uv venv --python 3.12 .venv && uv pip install pandas pyarrow numpy scipy scikit-learn matplotlib umap-learn pytest
cd analysis
../.venv/bin/python skill_luck.py && ../.venv/bin/python crowd.py            # skill vs luck, crowd calibration
../.venv/bin/python edge.py && ../.venv/bin/python toxic.py                  # extreme-favorite edge, toxic fills
../.venv/bin/python ml_features.py && ../.venv/bin/python ml_active.py && ../.venv/bin/python ml_skill.py   # ML pipeline
../.venv/bin/python build_reference.py                                       # regenerate trackrecord/reference.json
```

Place `data.parquet` in the repo root first. Scripts write to `analysis/out/`.
