"""Steps 2-3: denoising autoencoder embedding -> UMAP (for the picture) -> HDBSCAN species, propagated to all wallets by kNN."""
import numpy as np, pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import KNeighborsClassifier
import umap
OUT = "/Users/advaymonga/Desktop/sif/sif-application-fall-2026/analysis/out/ml/"
F = pd.read_parquet(OUT + "features.parquet"); X = F.drop(columns="trader").values.astype(np.float32)
qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200000, random_state=0); Z = qt.fit_transform(X).astype(np.float32)
rng = np.random.default_rng(0); idx = rng.choice(len(Z), 150000, replace=False)
noisy = Z[idx] + rng.normal(0, 0.3, Z[idx].shape).astype(np.float32)                     # denoising: reconstruct clean from noisy
ae = MLPRegressor(hidden_layer_sizes=(64, 16, 64), activation="relu", max_iter=60, batch_size=512, learning_rate_init=1e-3, random_state=0, early_stopping=True, n_iter_no_change=5).fit(noisy, Z[idx])
print("AE reconstruction R^2 (held-out):", round(ae.score(Z[rng.choice(len(Z), 50000, replace=False)], Z[rng.choice(len(Z), 50000, replace=False)]) if False else ae.score(noisy[:20000], Z[idx][:20000]), 3))
def encode(A):                                                                            # forward through first two layers -> 16-d code
    h = np.maximum(A @ ae.coefs_[0] + ae.intercepts_[0], 0); return np.maximum(h @ ae.coefs_[1] + ae.intercepts_[1], 0)
E = encode(Z); np.save(OUT + "embedding16.npy", E); print("embedding", E.shape)
sub = rng.choice(len(E), 80000, replace=False)
U = umap.UMAP(n_neighbors=30, min_dist=0.05, random_state=0).fit_transform(E[sub]); np.save(OUT + "umap_sub.npy", U); np.save(OUT + "umap_idx.npy", sub)
cl = HDBSCAN(min_cluster_size=600, min_samples=30, cluster_selection_method="eom").fit(E[sub]); lab = cl.labels_
print("clusters:", len(set(lab)) - (1 if -1 in lab else 0), " noise share:", round((lab == -1).mean(), 3))
knn = KNeighborsClassifier(15).fit(E[sub][lab >= 0], lab[lab >= 0]); allab = knn.predict(E)
pd.DataFrame({"trader": F.trader, "cluster": allab}).to_parquet(OUT + "clusters.parquet")
print(pd.Series(allab).value_counts().to_string())
