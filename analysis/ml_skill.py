"""Supervised skill model on active wallets: denoised target vs raw label (same wallets, same features), attribution, direction."""
import pathlib
import numpy as np, pandas as pd, json
from common import load
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance, partial_dependence
OUT = str(pathlib.Path(__file__).resolve().parent / "out/ml") + "/"
df = load(); F = pd.read_parquet(OUT + "features.parquet"); post = pd.read_parquet(str(pathlib.Path(__file__).resolve().parent / "out/skill_posterior.parquet")).set_index("trader")
df["p_pos"] = post.p_positive.reindex(df.trader).values
feats = [c for c in F.columns if c not in ("trader", "int_shares_single")]; X = F[feats].values.astype(np.float32)
cv = StratifiedKFold(5, shuffle=True, random_state=0); mk = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
res = {}
for nmin in [20, 50, 100]:
    base = (df.n >= nmin).values
    m1 = base & ((df.p_pos > 0.9) | (df.p_pos < 0.1)).values; y1 = (df.p_pos > 0.9).values[m1]                     # denoised target
    m2 = base & df.trader_label.isin(["sharp", "awful"]).values; y2 = (df.trader_label == "sharp").values[m2]         # raw label extremes
    m3 = base; y3 = (df.trader_pnl > 0).values[m3]                                                                     # profit sign
    for nm, m, y in [("denoised skill", m1, y1), ("raw label sharp-vs-awful", m2, y2), ("profit sign", m3, y3)]:
        p = cross_val_predict(mk(), X[m], y, cv=cv, method="predict_proba")[:, 1]; a = roc_auc_score(y, p); res[f"n>={nmin} | {nm}"] = dict(auc=float(a), n=int(m.sum()), pos_rate=float(y.mean()))
        print(f"n>={nmin:3d} {nm:26s} AUC={a:.3f} n={m.sum():,} pos={y.mean():.2f}")
# attribution on the main spec (n>=50, denoised)
m = (df.n >= 50).values & ((df.p_pos > 0.9) | (df.p_pos < 0.1)).values; y = (df.p_pos > 0.9).values[m]
mdl = mk().fit(X[m], y); pi = permutation_importance(mdl, X[m], y, n_repeats=5, random_state=0, scoring="roc_auc")
imp = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False); print("\npermutation importance (AUC drop):"); print(imp.head(12).round(4).to_string())
# direction: partial dependence at 10th vs 90th pct for top 8
rows = []
for f in imp.head(10).index:
    j = feats.index(f); lo, hi = np.percentile(X[m][:, j], [10, 90])
    pd_ = partial_dependence(mdl, X[m], [j], kind="average", grid_resolution=20)
    gx, gy = pd_["grid_values"][0], pd_["average"][0]
    rows.append(dict(feature=f, importance=float(imp[f]), pd_low=float(np.interp(lo, gx, gy)), pd_high=float(np.interp(hi, gx, gy)), x_low=float(lo), x_high=float(hi)))
attr = pd.DataFrame(rows); attr["direction"] = np.where(attr.pd_high > attr.pd_low, "more → more skilled", "more → less skilled"); print(); print(attr.round(3).to_string())
attr.to_csv(OUT + "skill_attribution.csv", index=False); json.dump(res, open(OUT + "skill_auc.json", "w"), indent=1)
# calibration-ish: predicted-skill deciles vs realized c/$ on held-out (cross_val_predict) for n>=50 wallets (all, not just extremes)
mall = (df.n >= 50).values; yall = (df.p_pos > 0.9).values[mall]
pall = cross_val_predict(mk(), X[m], y, cv=cv, method="predict_proba")[:, 1] if False else mk().fit(X[m], y).predict_proba(X[mall])[:, 1]   # in-sample for extremes, out for middle
d = df[mall].assign(score=pall); dec = pd.qcut(d.score.rank(method="first"), 10, labels=False)
led = d.groupby(dec).apply(lambda x: pd.Series({"wallets": len(x), "c_per_dollar": 100 * x.trader_pnl.sum() / x.notional.sum(), "frac_profitable": (x.trader_pnl > 0).mean(), "median_ppv_c": 100 * x.trader_ppv.median(), "notional_M": x.notional.sum() / 1e6}))
print("\nrealized outcomes by predicted-skill decile (n>=50 wallets):"); print(led.round(2).to_string()); led.to_csv(OUT + "skill_deciles.csv")
