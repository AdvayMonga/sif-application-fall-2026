"""Run the evaluator on each discovered cluster, treating the cluster as one pooled record; CI by resampling wallets."""
import pathlib
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import load
from sifeval.core import posterior, load_reference
OUT = str(pathlib.Path(__file__).resolve().parent / "out/ml") + "/"
NAMES = {0: "machines A", 11: "machines B", 3: "farm A", 13: "farm B", 14: "farm C", 8: "bust A", 10: "bust B"}

df = load(); C = pd.read_parquet(OUT + "active_clusters.parquet")
A = df[df.n >= 20].reset_index(drop=True).assign(cluster=C.cluster.values)
A["pq"] = (0.25 - (A.mean_delta ** 2 + A.std_delta.fillna(0) ** 2)).clip(1e-4, 0.25)
ref = load_reference(); rng = np.random.default_rng(0)

rows = []
for cid, g in A.groupby("cluster"):
    shares, dollars, pnl = g.trader_volume.sum(), g.notional.sum(), g.trader_pnl.sum()
    eps = pnl / shares; n = int(g.n.sum()); pq = float(np.average(g.pq, weights=g.trader_volume))
    post = posterior(eps, pq, n, ref)                                    # pooled: treats every trade as independent
    idx = rng.integers(0, len(g), (2000, len(g)))                        # honest CI: resample WALLETS, not trades
    v, p_, d_ = g.trader_volume.values, g.trader_pnl.values, g.notional.values
    boot_share = p_[idx].sum(1) / v[idx].sum(1); boot_dollar = p_[idx].sum(1) / d_[idx].sum(1)
    rows.append(dict(cluster=cid, name=NAMES.get(cid, f"retail {cid}"), wallets=len(g), trades=n, notional_M=dollars / 1e6,
                     edge_per_dollar=100 * pnl / dollars, edge_per_share=100 * eps, pooled_post=100 * post["mean"],
                     ci_lo=100 * np.percentile(boot_share, 5), ci_hi=100 * np.percentile(boot_share, 95),
                     ci_lo_d=100 * np.percentile(boot_dollar, 5), ci_hi_d=100 * np.percentile(boot_dollar, 95),
                     p_positive=float((boot_share > 0).mean()), frac_profitable=float((g.trader_pnl > 0).mean()), median_trades=float(g.n.median())))
t = pd.DataFrame(rows).sort_values("edge_per_dollar", ascending=False)
t["verdict"] = np.where(t.p_positive > .95, "real positive edge", np.where(t.p_positive < .05, "real negative edge", "consistent with luck"))
t.to_csv(OUT + "cluster_eval.csv", index=False)
pd.set_option("display.width", 250); print(t[["name", "wallets", "trades", "notional_M", "edge_per_dollar", "ci_lo_d", "ci_hi_d", "p_positive", "frac_profitable", "verdict"]].round(3).to_string(index=False))

# grouped families
A["family"] = A.cluster.map(lambda c: NAMES.get(c, "retail").split()[0])
fam = []
for f, g in A.groupby("family"):
    idx = rng.integers(0, len(g), (2000, len(g))); p_, d_ = g.trader_pnl.values, g.notional.values
    boot = p_[idx].sum(1) / d_[idx].sum(1)
    fam.append(dict(family=f, wallets=len(g), notional_M=g.notional.sum() / 1e6, pnl_M=g.trader_pnl.sum() / 1e6, edge_per_dollar=100 * g.trader_pnl.sum() / g.notional.sum(),
                    ci_lo=100 * np.percentile(boot, 5), ci_hi=100 * np.percentile(boot, 95), p_positive=float((boot > 0).mean())))
F = pd.DataFrame(fam).sort_values("edge_per_dollar", ascending=False)
F["verdict"] = np.where(F.p_positive > .95, "real positive edge", np.where(F.p_positive < .05, "real negative edge", "consistent with luck"))
F.to_csv(OUT + "cluster_eval_family.csv", index=False); print(); print(F.round(3).to_string(index=False))
