"""'One bot' chain: losses at 98c+ are predictable and concentrated in a single sweep pattern. Tables for memo_toxic_web.py."""
import json, numpy as np, pandas as pd
from common import load, singles
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import binom
OUT = "/Users/advaymonga/Desktop/sif/sif-application-fall-2026/analysis/out/"

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

df = load(); s = singles(df); r = s[s.outcome != "open"].copy()
key = r.mean_time.astype(int).astype(str) + "|" + r.p.round(3).astype(str) + "|" + r.dom
r["batch"] = key.map(key.value_counts()) >= 3
r = r[(~r.batch) & (r.trader_pnl.abs() < 10000)].copy()
r["hr"] = r.mean_time / 3.6e6; r["hs"] = np.sin(2 * np.pi * r.hr / 24); r["hc"] = np.cos(2 * np.pi * r.hr / 24)
r["logusd"] = np.log10(r.mean_tx_value.clip(0.01)); r["int_shares"] = (np.abs(r.trader_volume - r.trader_volume.round()) < 0.005).astype(int)
r["round_usd"] = r.mean_tx_value.round(2).isin([1, 2, 5, 10, 20, 25, 50, 100, 200, 500, 1000]).astype(int); r["aggr"] = r.price_levels_per_transaction
r["sport"] = (r.dom == "sport").astype(int); r["politics"] = (r.dom == "politics").astype(int); r["shares"] = r.trader_volume
S = {}

# Step 1: the edge by band (95c+)
hi = r[r.p >= 0.95].copy(); hi["lost"] = 1 - hi.won
g = hi.groupby(pd.cut(hi.p, [.95, .98, .99, .995, 1.0], right=False), observed=True)
bands = pd.DataFrame({"n": g.size(), "implied": g.p.apply(lambda x: (1 - x).mean()), "realized": g.lost.mean(), "edge": 100 * g.trader_pnl.sum() / g.mean_tx_value.sum()})
bands["lo"], bands["hi"] = wilson(g.lost.sum(), g.size()); bands.to_csv(OUT + "toxic_bands.csv")

# Step 2: at >=98c, can a model predict WHICH bets lose?
h = r[r.p >= 0.98].copy(); h["lost"] = 1 - h.won
feats = ["p", "hs", "hc", "logusd", "int_shares", "round_usd", "aggr", "sport", "politics", "shares"]
X = h[feats].values; y = h.lost.values; oof = np.zeros(len(h))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
    oof[te] = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=8, l2_regularization=2.0, random_state=0).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
h["risk"] = oof; S["auc"] = float(roc_auc_score(y, oof)); S["n98"] = int(len(h)); S["losses98"] = int(y.sum()); S["implied98"] = float((1 - h.p).mean()); S["realized98"] = float(y.mean())
q = pd.qcut(h.risk.rank(method="first"), 5, labels=False)
quint = h.groupby(q).apply(lambda x: pd.Series({"n": len(x), "loss_rate": x.lost.mean(), "losses": x.lost.sum(), "implied": (1 - x.p).mean(), "edge": 100 * x.trader_pnl.sum() / x.mean_tx_value.sum(), "dollars": x.mean_tx_value.sum()}))
quint.to_csv(OUT + "toxic_quintiles.csv"); S["top_quintile_share_of_losses"] = float(quint.losses.iloc[-1] / quint.losses.sum())

# Step 3: what is the pattern? loss rate by share-count bucket, sport vs other, 97-99.5c
w = r[(r.p >= 0.97) & (r.p < 0.995)].copy(); w["lost"] = 1 - w.won
w["bucket"] = pd.cut(w.shares, [0, 5, 10, 15, 20, 50, 100, 1e9], labels=["<5", "5–10", "10–15", "15–20", "20–50", "50–100", "100+"])
pat = w.groupby(["sport", "bucket", "int_shares"], observed=True).agg(n=("lost", "size"), loss_rate=("lost", "mean"), losses=("lost", "sum")).reset_index()
pat.to_csv(OUT + "toxic_pattern.csv", index=False)
rule = (h.sport == 1) & (h.int_shares == 1) & h.shares.between(10, 15)
rr, others = h[rule], h[~rule]
S["rule"] = dict(n=int(len(rr)), losses=int(rr.lost.sum()), loss_rate=float(rr.lost.mean()), pnl=float(rr.trader_pnl.sum()), dollars=float(rr.mean_tx_value.sum()), implied=float((1 - rr.p).mean()))
S["others"] = dict(n=int(len(others)), losses=int(others.lost.sum()), loss_rate=float(others.lost.mean()), pnl=float(others.trader_pnl.sum()), dollars=float(others.mean_tx_value.sum()), implied=float((1 - others.p).mean()), edge=float(100 * others.trader_pnl.sum() / others.mean_tx_value.sum()))
S["rule_share_of_losses"] = float(rr.lost.sum() / h.lost.sum())
# Step 4: the mechanics of the losers — same-second clustering, maker-like, price points, hours
L = h[h.lost == 1]
sec = L.mean_time.astype(int); cl = sec.map(sec.value_counts())
allsec = h.mean_time.astype(int).value_counts()
S["losers"] = dict(n=int(len(L)), share_maker=float((L.aggr == 0).mean()), share_int_shares=float(L.int_shares.mean()), share_sport=float(L.sport.mean()),
                   share_same_second_as_another_loser=float((cl >= 2).mean()), share_same_second_any=float((sec.map(allsec) >= 2).mean()),
                   price_modes=L.p.round(3).value_counts().head(5).to_dict(), hour_hist=L.hr.astype(int).value_counts().sort_index().to_dict(), median_shares=float(L.shares.median()), dollars=float(L.mean_tx_value.sum()))
S["baseline_same_second"] = float((h.mean_time.astype(int).map(allsec) >= 2).mean())
# counterparty: who benefits when the loser's bid is hit? the taker selling at 98c holds the complement; realized gain = loser's stake
# Sizing: clean flow vs raw
for nm, loss, P in [("raw", S["realized98"], float(h.p.mean())), ("clean", S["others"]["loss_rate"], float(others.p.mean()))]:
    b = (1 - P) / P; N = 200
    S[f"mc_{nm}"] = dict(P=P, loss=loss, b=b, edge_per_bet=100 * ((1 - loss) * b - loss), p0=float(binom.pmf(0, N, loss)), p1=float(binom.pmf(1, N, loss)), p2plus=float(1 - binom.cdf(1, N, loss)),
                         mult25_0=(1 + .25 * b) ** N, mult25_1=(1 + .25 * b) ** (N - 1) * .75, mult25_2=(1 + .25 * b) ** (N - 2) * .75 ** 2)
json.dump(S, open(OUT + "toxic_summary.json", "w"), indent=1, default=float)
print(bands.round(4)); print(quint.round(4)); print(json.dumps(S, indent=1, default=float)[:3000])
