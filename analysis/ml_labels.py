"""Step 5: confident learning on trader_label using behavior-only features; retrain on cleaned labels; compare what the model learns."""
import numpy as np, pandas as pd, json
from common import load
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.inspection import permutation_importance
OUT = "/Users/advaymonga/Desktop/sif/sif-application-fall-2026/analysis/out/ml/"
df = load(); F = pd.read_parquet(OUT + "features.parquet"); X = F.drop(columns="trader").values.astype(np.float32); feats = list(F.drop(columns="trader").columns)
y = df.trader_label.map({"awful": 0, "bad": 1, "good": 2, "sharp": 3}).values
oof = np.zeros((len(y), 4))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
    oof[te] = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0).fit(X[tr], y[tr]).predict_proba(X[te])
print("behavior-only model on given label: acc", round(accuracy_score(y, oof.argmax(1)), 3), "macro AUC", round(roc_auc_score(y, oof, multi_class="ovr"), 3))
# confident learning (Northcutt et al.): class thresholds = mean self-confidence per class; a label is suspect if another class exceeds its threshold and beats the given class
thr = np.array([oof[y == k, k].mean() for k in range(4)])
cand = (oof >= thr[None, :]); cand[np.arange(len(y)), y] = False
suspect = cand.any(1) & (oof.max(1) > oof[np.arange(len(y)), y]); alt = oof.argmax(1)
print(f"flagged as likely mislabeled: {suspect.sum():,} ({suspect.mean():.1%})")
by_n = pd.DataFrame({"n": df.n, "suspect": suspect, "label": df.trader_label}).groupby(pd.cut(df.n, [0, 1, 5, 20, 100, 1e9], labels=["1", "2-5", "6-20", "21-100", "100+"]), observed=True).suspect.mean()
print("flag rate by trade count:"); print(by_n.round(3).to_string())
by_lab = pd.Series(suspect).groupby(df.trader_label.values).mean(); print("flag rate by given label:"); print(by_lab.round(3).to_string())
# retrain on cleaned labels (drop suspects) and compare feature importances + accuracy on 'clean' high-n wallets
keep = ~suspect; hi = (df.n >= 50).values
def fit_imp(mask, name):
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0).fit(X[mask], y[mask])
    pi = permutation_importance(m, X[hi][:60000], y[hi][:60000], n_repeats=3, random_state=0, scoring="accuracy")
    imp = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False); print(f"\n{name}: top features:"); print(imp.head(8).round(4).to_string())
    return m, imp
m0, imp0 = fit_imp(np.ones(len(y), bool), "trained on given labels"); m1, imp1 = fit_imp(keep, "trained on cleaned labels")
tr_hi = hi & keep
print("\naccuracy on high-n wallets (n>=50), clean labels only:", round(accuracy_score(y[tr_hi], m0.predict(X[tr_hi])), 3), "(given) vs", round(accuracy_score(y[tr_hi], m1.predict(X[tr_hi])), 3), "(cleaned)")
pd.DataFrame({"given": imp0, "cleaned": imp1}).to_csv(OUT + "label_importances.csv")
pd.DataFrame({"trader": df.trader, "suspect": suspect, "given": df.trader_label, "suggested": pd.Series(alt).map({0: "awful", 1: "bad", 2: "good", 3: "sharp"})}).to_parquet(OUT + "label_issues.parquet")
json.dump({"acc_given": float(accuracy_score(y, oof.argmax(1))), "flag_rate": float(suspect.mean()), "flag_by_n": by_n.to_dict(), "flag_by_label": by_lab.to_dict()}, open(OUT + "labels_summary.json", "w"), default=str)
