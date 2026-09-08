"""Steps 4 & 6: blind validation of clusters against evidence the embedding never saw; cluster ledger; linear probe."""
import numpy as np, pandas as pd, json
from common import load, singles
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
OUT = "/Users/advaymonga/Desktop/sif/sif-application-fall-2026/analysis/out/ml/"
df = load(); C = pd.read_parquet(OUT + "clusters.parquet"); df["cluster"] = C.cluster.values
E = np.load(OUT + "embedding16.npy"); F = pd.read_parquet(OUT + "features.parquet")
# independent fingerprints (none used by the embedding)
s = singles(df); key = s.mean_time.astype(int).astype(str) + "|" + s.p.round(3).astype(str) + "|" + s.dom
batch_ids = set(s[key.map(key.value_counts()) >= 3].trader)
df["ev_farm_batch"] = df.trader.isin(batch_ids)
df["ev_bot_clock"] = (df.std_time / 3.6e6).between(6.3, 7.6) & (df.mean_time / 3.6e6).between(10.5, 13.5) & (df.n >= 20)
df["ev_hedged_pair"] = (df.n == 2) & (df.trader_pnl == 0) & (df.std_delta == 0)
df["ev_zero_pnl_maker"] = (df.n >= 4) & (df.price_levels_per_transaction == 0) & (df.trader_pnl.abs() <= 0.001 * df.notional)
post = pd.read_parquet("/Users/advaymonga/Desktop/sif/sif-application-fall-2026/analysis/out/skill_posterior.parquet")
sk = set(post[(post.p_positive > 0.99) & (post.n >= 20)].trader); df["ev_skilled"] = df.trader.isin(sk)
df["ev_bust"] = (df.trader_pnl < 0) & (-df.trader_pnl >= 0.98 * df.notional)
df["profitable"] = df.trader_pnl > 0
ev = ["ev_farm_batch", "ev_bot_clock", "ev_hedged_pair", "ev_zero_pnl_maker", "ev_skilled", "ev_bust"]
g = df.groupby("cluster")
tab = pd.DataFrame({"wallets": g.size(), "share_of_wallets": g.size() / len(df), "notional_M": g.notional.sum() / 1e6, "pnl_M": g.trader_pnl.sum() / 1e6,
                    "c_per_dollar": 100 * g.trader_pnl.sum() / g.notional.sum(), "frac_profitable": g.profitable.mean(), "median_n": g.n.median(),
                    "avg_price": g.apply(lambda x: (x.notional.sum() / x.trader_volume.sum())), "median_ticket": g.mean_tx_value.median(),
                    "hour": g.apply(lambda x: (x.mean_time.median() / 3.6e6)), "time_spread": g.apply(lambda x: x.std_time.median() / 3.6e6), "sport": g.apply(lambda x: x.topic_sport.mean())})
for e in ev: tab[e + "_rate"] = g[e].mean(); tab[e + "_lift"] = g[e].mean() / df[e].mean(); tab[e + "_share"] = g[e].sum() / df[e].sum()
tab.to_csv(OUT + "cluster_table.csv"); pd.set_option("display.width", 250)
print(tab[["wallets", "notional_M", "pnl_M", "c_per_dollar", "frac_profitable", "median_n", "avg_price", "median_ticket", "hour", "time_spread", "sport"]].round(2).to_string())
print("\nLIFT of independent evidence by cluster (1 = population rate):"); print(tab[[e + "_lift" for e in ev]].round(1).to_string())
print("\nSHARE of each evidence type captured by cluster:"); print(tab[[e + "_share" for e in ev]].round(2).to_string())
# Step 6: linear probe — predict profitable on n>=50 wallets: embedding vs raw features vs both
m = (df.n >= 50).values; y = df.profitable.values[m]; Xf = F.drop(columns="trader").values[m]; Xe = E[m]
cv = StratifiedKFold(5, shuffle=True, random_state=0); res = {}
for nm, X in [("embedding16 (linear)", Xe), ("raw features (linear)", Xf), ("raw features (GBM)", Xf)]:
    mdl = HistGradientBoostingClassifier(max_iter=200, random_state=0) if "GBM" in nm else make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    res[nm] = float(cross_val_score(mdl, X, y, cv=cv, scoring="roc_auc").mean()); print(f"probe {nm:24s} AUC={res[nm]:.3f}  (n={m.sum():,})")
json.dump(res, open(OUT + "probe.json", "w"))
