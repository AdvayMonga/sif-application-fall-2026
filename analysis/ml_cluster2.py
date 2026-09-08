"""Step 3 (coarse): HDBSCAN on the UMAP plane of the 80k subsample, propagate to all wallets via kNN in embedding space."""
import numpy as np, pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import KNeighborsClassifier
OUT = "/Users/advaymonga/Desktop/sif/sif-application-fall-2026/analysis/out/ml/"
E = np.load(OUT + "embedding16.npy"); U = np.load(OUT + "umap_sub.npy"); sub = np.load(OUT + "umap_idx.npy"); F = pd.read_parquet(OUT + "features.parquet")
best = None
for mcs in [1500, 2500, 4000]:
    lab = HDBSCAN(min_cluster_size=mcs, min_samples=50, cluster_selection_method="eom").fit(U).labels_
    k = len(set(lab)) - (1 if -1 in lab else 0); print(f"min_cluster_size={mcs}: {k} clusters, noise {(lab==-1).mean():.3f}")
    if best is None or (6 <= k <= 14): best = (mcs, lab)
mcs, lab = best; print("using", mcs)
core = lab >= 0
knn_u = KNeighborsClassifier(25).fit(U[core], lab[core]); lab_sub = np.where(core, lab, knn_u.predict(U))       # fill noise points in-plane
knn_e = KNeighborsClassifier(15).fit(E[sub], lab_sub); allab = knn_e.predict(E)
pd.DataFrame({"trader": F.trader, "cluster": allab}).to_parquet(OUT + "clusters.parquet"); np.save(OUT + "umap_labels_sub.npy", lab_sub)
print(pd.Series(allab).value_counts().sort_index().to_string())
