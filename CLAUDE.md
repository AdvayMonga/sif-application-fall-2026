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
- `analysis/common.py` — `load()` (derived cols `n`, `dom`, `pq`, `notional`), `singles()` (price/side/outcome recovery), `noise_constant()` (Var(ppv|n=1) = k·p(1−p)).
- `analysis/skill_luck.py` — NPMLE deconvolution of true skill θ (EM on a 241-pt grid; Gaussian or variance-matched t₄ noise, wallet-specific sd √(k·pq/n)); robustness over k and likelihood, splits by trade band/topic; tail diagnostic. Writes `out/skill_prior.csv`, `out/skill_posterior.parquet`, `out/robustness_*.csv`, `out/tail_diagnostic.csv`.
- `analysis/crowd.py` — realized win rate vs price paid on resolved single bets: full/quantile calibration tables with Wilson CIs, bootstrapped mid-range gap, by-topic, 50¢ trap, Prelec / Goldstein–Einhorn weighting fits with bootstrap CIs. Writes `out/calibration_*.csv`, `out/trap50.csv`, `out/weighting_fits.csv`, `out/crowd_summary.json`.
- `analysis/figures.py` — memo figures (matplotlib, dataviz-skill palette: blue accent / red emphasis / gray de-emphasis, ordinal blue ramp): funnel, recovered skill distribution, label-vs-truth stacked bars, calibration curve, 50¢ trap by topic. Writes `out/fig*.png`, `out/label_vs_truth_mid.csv`.
- `analysis/memo.py` — assembles the memo from `out/` tables + figures (every number pulled programmatically) into `out/memo.html`, then prints `out/memo.pdf` via headless Chrome. `out/pdfpreview/` holds page renders (pymupdf) for visual QA.
- `analysis/edge.py` — "Two Exploits" memo tables: farm footprint (99¢-only, maker-only zero-P&L, exact-zero, hedged pairs, same-second batches, $100-quest tickets) and the extreme-favorite edge (band table with Wilson CIs + 2026 fee, by hour/topic/size, repeat-wallet persistence, Monte Carlo, capacity). Writes `out/edge_*.csv/json/npy`.
- `analysis/figures_edge.py` — `out/fig_e1_bands.png`, `fig_e2_montecarlo.png`, `fig_e3_farm.png`.
- `analysis/memo_edge.py` — print memo: `out/memo_edge.html` → `out/memo_edge.pdf` (copy kept in `deliverables/`, gitignored).
- `analysis/memo_edge_web.py` — web-first memo with inline SVG charts (theme tokens, hover titles): `out/memo_edge_artifact.html` (copy in `deliverables/`), published as a Claude artifact.
- `analysis/toxic.py` — 'one bot' chain: 95¢+ band edge, CV loss model at ≥98¢ (AUC), risk quintiles, the sport/whole-share 10–15 pattern, loser mechanics (maker share, hours, price modes), clean-vs-raw sizing. Writes `out/toxic_*.csv/json`.
- `analysis/memo_toxic_web.py` — web memo for that chain (reuses the edge memo CSS): `out/memo_toxic_artifact.html`, published as a separate artifact.
- `analysis/ml_features.py` → `ml_embed.py`/`ml_cluster2.py` (all wallets; weak) → `ml_active.py` (n≥20: AE embedding, UMAP+HDBSCAN species, blind validation, probes) → `ml_skill.py` (denoised-skill vs raw-label vs profit AUC; attribution) → `ml_labels.py` (confident learning) → `ml_page.py` (OOF decile ledger + `out/memo_ml_artifact.html`). Outputs in `out/ml/`.
- `analysis/out/` — generated tables/figures/memo (not source).

## Eval harness
`trackrecord/` — `python -m trackrecord eval <csv>` (price,size,won[,pnl]) or `wallet <address> --data data.parquet`. `core.py` = posterior against the census prior in `reference.json` (built by `analysis/build_reference.py`), `features.py` = trade list / census row → aggregates, `cli.py` = report card. Tests in `tests/`, example in `examples/`. Verdict thresholds: P(edge > +0.5¢/share) or P(edge < −0.5¢/share) > 0.95. `diagnostics.py` = extra fronts for trade lists (calibration, sizing value, concentration/bootstrap, split-half, drawdown/streak/MC at 5% bankroll). `returns.py` = returns mode (split-half Sharpe, streaks, autocorr, MC drawdowns, optional benchmark alpha/beta) (`trackrecord returns <csv>`: t-test on per-period returns; used on SIF Live's public equity curve, snapshot in `deliverables/siflive_dashboard_2026-09-08.json`).

## Run order
Skill/crowd memo: `skill_luck.py` → `crowd.py` → `figures.py` → `memo.py`. Two-Exploits memo: `edge.py` → `figures_edge.py` → `memo_edge.py`. Run from `analysis/` with `../.venv/bin/python`.

## Report thesis (current)
Submission = the ML memo (`ml_page.py` → artifact f3f1c3d8…, PDF in `deliverables/SIF_Application_Fall_2026.pdf`): skill is predictable from behavior (AUC 0.90) only after luck is removed from the target; bet-size dispersion is the top tell; `trackrecord` is the appendix. Earlier memos (skill/crowd, Two Exploits, One Bot) are kept as alternates. Dataset-only; external facts for context only.
