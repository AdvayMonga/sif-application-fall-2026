"""Steps 3-6 redone on ACTIVE wallets (n>=20): AE embedding, HDBSCAN species, blind validation, ledger, probes (profit + deconvolved skill)."""
import numpy as np, pandas as pd, json
from common import load, singles
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
import umap
OUT = "/Users/advaymonga/Desktop/sif/sif-application-fall-2026/analysis/out/ml/"
df = load(); F = pd.read_parquet(OUT + "features.parquet"); act = (df.n >= 20).values
A = df[act].reset_index(drop=True); X = F[act].drop(columns=["trader", "int_shares_single"]).values.astype(np.float32); feats = list(F.drop(columns=["trader", "int_shares_single"]).columns)
Z = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=0).fit_transform(X).astype(np.float32)
rng = np.random.default_rng(0); noisy = Z + rng.normal(0, 0.3, Z.shape).astype(np.float32)
ae = MLPRegressor(hidden_layer_sizes=(64, 12, 64), activation="relu", max_iter=80, batch_size=512, random_state=0, early_stopping=True, n_iter_no_change=6).fit(noisy, Z)
enc = lambda M: np.maximum(np.maximum(M @ ae.coefs_[0] + ae.intercepts_[0], 0) @ ae.coefs_[1] + ae.intercepts_[1], 0)
E = enc(Z); print("active wallets", len(A), "AE R2", round(ae.score(noisy[:30000], Z[:30000]), 3))
U = umap.UMAP(n_neighbors=40, min_dist=0.05, random_state=0).fit_transform(E)
for mcs in [800, 1500, 3000]:
    lab = HDBSCAN(min_cluster_size=mcs, min_samples=40).fit(U).labels_; k = len(set(lab)) - (1 if -1 in lab else 0); print(f"mcs={mcs}: {k} clusters noise {(lab==-1).mean():.3f}")
    if 6 <= k <= 14: break
core = lab >= 0; lab = np.where(core, lab, KNeighborsClassifier(25).fit(U[core], lab[core]).predict(U)); A["cluster"] = lab
np.save(OUT + "active_umap.npy", U); np.save(OUT + "active_embedding.npy", E); pd.DataFrame({"trader": A.trader, "cluster": lab}).to_parquet(OUT + "active_clusters.parquet")
# evidence never seen by the embedding
s = singles(df); key = s.mean_time.astype(int).astype(str) + "|" + s.p.round(3).astype(str) + "|" + s.dom
A["ev_bot_clock"] = (A.std_time / 3.6e6).between(6.3, 7.6) & (A.mean_time / 3.6e6).between(10.5, 13.5)
A["ev_zero_pnl_maker"] = (A.price_levels_per_transaction == 0) & (A.trader_pnl.abs() <= 0.001 * A.notional)
post = pd.read_parquet("/Users/advaymonga/Desktop/sif/sif-application-fall-2026/analysis/out/skill_posterior.parquet").set_index("trader")
A["p_pos"] = post.p_positive.reindex(A.trader).values; A["ev_skilled"] = A.p_pos > 0.99; A["ev_anti"] = A.p_pos < 0.01
A["ev_bust"] = (A.trader_pnl < 0) & (-A.trader_pnl >= 0.98 * A.notional); A["profitable"] = A.trader_pnl > 0
ev = ["ev_bot_clock", "ev_zero_pnl_maker", "ev_skilled", "ev_anti", "ev_bust"]
g = A.groupby("cluster"); pd.set_option("display.width", 250)
tab = pd.DataFrame({"wallets": g.size(), "notional_M": g.notional.sum() / 1e6, "pnl_M": g.trader_pnl.sum() / 1e6, "c_per_dollar": 100 * g.trader_pnl.sum() / g.notional.sum(), "frac_profitable": g.profitable.mean(),
                    "median_n": g.n.median(), "avg_price": g.apply(lambda x: x.notional.sum() / x.trader_volume.sum()), "median_ticket": g.mean_tx_value.median(), "hour": g.apply(lambda x: x.mean_time.median() / 3.6e6),
                    "time_spread": g.apply(lambda x: x.std_time.median() / 3.6e6), "tx_day": g.transactions_per_day.median(), "maker_share": g.apply(lambda x: (x.price_levels_per_transaction == 0).mean()), "sport": g.topic_sport.mean(), "politics": g.topic_politics.mean()})
for e in ev: tab[e + "_lift"] = g[e].mean() / A[e].mean(); tab[e + "_share"] = g[e].sum() / A[e].sum()
tab.to_csv(OUT + "active_cluster_table.csv")
print(tab[["wallets", "notional_M", "pnl_M", "c_per_dollar", "frac_profitable", "median_n", "avg_price", "median_ticket", "hour", "time_spread", "tx_day", "maker_share", "sport", "politics"]].round(2).to_string())
print("\nLIFT:"); print(tab[[e + "_lift" for e in ev]].round(1).to_string()); print("\nSHARE captured:"); print(tab[[e + "_share" for e in ev]].round(2).to_string())
# probes on n>=50: profit sign, and deconvolved skill (p_pos>0.9 vs <0.1)
cv = StratifiedKFold(5, shuffle=True, random_state=0); res = {}
for tname, mask, y in [("profitable", (A.n >= 50).values, A.profitable.values), ("skilled(p>0.9) vs unskilled(p<0.1)", ((A.n >= 50) & ((A.p_pos > 0.9) | (A.p_pos < 0.1))).values, (A.p_pos > 0.9).values)]:
    for nm, Xm in [("embedding (linear)", E), ("raw (linear)", Z), ("raw (GBM)", Z)]:
        mdl = HistGradientBoostingClassifier(max_iter=200, random_state=0) if "GBM" in nm else make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        a = float(cross_val_score(mdl, Xm[mask], y[mask], cv=cv, scoring="roc_auc").mean()); res[f"{tname} | {nm}"] = a; print(f"probe {tname:38s} {nm:20s} AUC={a:.3f} n={mask.sum():,}")
json.dump(res, open(OUT + "active_probe.json", "w"))
