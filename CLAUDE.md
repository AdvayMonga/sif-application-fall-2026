# sif — SIF quant-club interview project

Dataset: `data.parquet` — 604,578 rows, one per Polymarket wallet (cross-sectional census, ~Jun 2024–Mar 2025; ΣPnL = 0).
Env: `.venv` (Python 3.12; pandas, pyarrow, numpy, scipy, scikit-learn, matplotlib). Run scripts from `analysis/` with `../.venv/bin/python`.

## Decoded columns (not documented anywhere else)
- `trader_volume` = **shares**; `mean_tx_value` = dollars. Price paid = `mean_tx_value / trader_volume`.
- `mean_delta`, `std_delta` = moments of |price − 0.5|; `E[p(1−p)] = 0.25 − (mean² + std²)`.
- `mean_time`, `std_time` = **ms of day** (linear mean of a circular variable).
- `trader_ppv` = pnl/shares clipped ±0.999; `trader_label` = pure threshold on ppv (awful < −0.0694 ≤ bad < 0 ≤ good < 0.0399 ≤ sharp) — total leakage.
- `std_*` null ⇔ `transaction_count == 1`. For those wallets pnl matches a resolution payoff 85% of the time (win → ppv = 1−p, lose → ppv = −p).

## Layout
- `sifeval/` — the tool. `core.py` scores a record against the population prior in `reference.json`, `features.py` turns trades or a dataset row into inputs, `returns.py` handles P&L series, `diagnostics.py` the extra checks, `cli.py` the report cards.
- `analysis/` — in run order: `common.py` (decode), `skill_luck.py` (NPMLE deconvolution), `build_reference.py` (writes `sifeval/reference.json`), `ml_features.py` (26 behaviour measures), `ml_active.py` (autoencoder + UMAP + HDBSCAN on n>=20 wallets), `cluster_eval.py` (score each group, wallet bootstrap), `ml_labels.py` (confident learning on the given label), `ml_skill.py` (denoised-skill vs raw-label AUC, attribution), `ml_page.py` (builds `out/memo_ml_artifact.html`).
- `tests/`, `examples/`, `data/` (SIF Live snapshot). All paths are repo-relative.
- `analysis/out/`, `deliverables/`, `data.parquet` are gitignored.

## Eval harness
`sifeval/` — `python -m sifeval eval <csv>` (price,size,won[,pnl]) or `wallet <address> --data data.parquet`. `core.py` = posterior against the census prior in `reference.json` (built by `analysis/build_reference.py`), `features.py` = trade list / census row → aggregates, `cli.py` = report card. Tests in `tests/`, example in `examples/`. Verdict thresholds: P(edge > +0.5¢/share) or P(edge < −0.5¢/share) > 0.95. `diagnostics.py` = extra fronts for trade lists (calibration, sizing value, concentration/bootstrap, split-half, drawdown/streak/MC at 5% bankroll). `returns.py` = returns mode (split-half Sharpe, streaks, autocorr, MC drawdowns, optional benchmark alpha/beta) (`sifeval returns <csv>`: t-test on per-period returns; used on SIF Live's public equity curve, snapshot in `deliverables/siflive_dashboard_2026-09-08.json`).

## Run order
Skill/crowd memo: `skill_luck.py` → `crowd.py` → `figures.py` → `memo.py`. Two-Exploits memo: `edge.py` → `figures_edge.py` → `memo_edge.py`. Run from `analysis/` with `../.venv/bin/python`.

## Report thesis (current)
Submission = the ML memo (`ml_page.py` → artifact f3f1c3d8…, PDF in `deliverables/SIF_Application_Fall_2026.pdf`): skill is predictable from behavior (AUC 0.90) only after luck is removed from the target; bet-size dispersion is the top tell; `sifeval` is the appendix. Earlier memos (skill/crowd, Two Exploits, One Bot) are kept as alternates. Dataset-only; external facts for context only.
