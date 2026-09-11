"""Distill the population reference the eval harness needs (prior over true edge, noise constant, percentile tables) into sifeval/reference.json."""
import pathlib
import json, numpy as np, pandas as pd
from common import load, noise_constant
from skill_luck import GRID, run
OUT = str(pathlib.Path(__file__).resolve().parent.parent) + "/"
df = load(); k = noise_constant(df); d = df[df.n >= 2]
w, _ = run(d, k, "gauss")                                         # NPMLE prior over true edge (per share)
act = df[df.n >= 20]; q = np.linspace(0, 1, 101)
cv = (act.std_tx_value / act.mean_tx_value.clip(1e-6)).fillna(0).clip(0, 20)
ref = {"noise_constant": k, "grid": GRID.round(4).tolist(), "prior": [float(x) for x in w],
       "active_min_trades": 20, "n_active": int(len(act)),
       "pct": {"edge_per_share": np.quantile(act.trader_ppv, q).round(5).tolist(), "edge_per_dollar": np.quantile(act.trader_pnl / act.notional, q).round(5).tolist(),
               "ticket_cv": np.quantile(cv, q).round(4).tolist(), "trades": np.quantile(act.n, q).round(0).tolist(), "avg_price": np.quantile((act.notional / act.trader_volume).clip(0, 1), q).round(4).tolist()}}
json.dump(ref, open(OUT + "sifeval/reference.json", "w")); print("reference written; prior mass |edge|<1c:", round(sum(p for g, p in zip(GRID, w) if abs(g) < .01), 3))
