"""Nonparametric deconvolution of true skill (Kiefer-Wolfowitz NPMLE via EM) + robustness."""
import sys, numpy as np, pandas as pd
from scipy.stats import norm, t as student_t
from common import load, noise_constant

OUT = "/Users/advaymonga/Desktop/sif/sif-application-fall-2026/analysis/out/"
GRID = np.linspace(-1.0, 1.0, 401)   # covers the full ppv support (clipped at ±0.999)
SHARP, AWFUL = 0.0399, -0.0694


def npmle(y, s, lik="gauss", df_t=4, iters=500, tol=1e-6):
    """Return prior weights on GRID and per-wallet posterior over GRID."""
    if lik != "gauss":  # variance-matched t scale so both likelihoods share the same noise variance
        s = s * np.sqrt((df_t - 2) / df_t)
    z = (y[:, None] - GRID[None, :]) / s[:, None]
    L = (norm.pdf(z) if lik == "gauss" else student_t.pdf(z, df_t)) / s[:, None]
    L = L.astype(np.float32) + 1e-30
    w = np.full(len(GRID), 1 / len(GRID), dtype=np.float32)
    for _ in range(iters):
        post = L * w
        post /= post.sum(1, keepdims=True)
        w_new = post.mean(0)
        if np.abs(w_new - w).max() < tol:
            w = w_new; break
        w = w_new
    post = L * w; post /= post.sum(1, keepdims=True)
    return w, post


def summarize(w, post, labels):
    m = lambda lo, hi: float(w[(GRID > lo) & (GRID <= hi)].sum())
    p_pos = post[:, GRID > 0].sum(1)
    return dict(
        zero_edge=m(-0.01, 0.01), positive=m(0, 1), sharp=m(SHARP, 1), awful=m(-1, AWFUL),
        label_sharp=float((labels == "sharp").mean()), label_awful=float((labels == "awful").mean()),
        sharp_precision=float(post[labels == "sharp"][:, GRID > SHARP].sum(1).mean()),
        awful_precision=float(post[labels == "awful"][:, GRID < AWFUL].sum(1).mean()),
        skilled_95=int((p_pos > .95).sum()), skilled_99=int((p_pos > .99).sum()), n=len(labels),
    )


def tail_diagnostic(d, k):
    """Excess kurtosis of standardized ppv by trade-count band (Gaussian = 0; t4 = inf; two-point outcome < 0)."""
    z = d.trader_ppv / np.sqrt(k * d.pq / d.n)
    bands = pd.cut(d.n, [0, 1, 5, 20, 100, 1e9], labels=["1", "2-5", "6-20", "21-100", "100+"])
    g = z.groupby(bands, observed=True)
    return pd.DataFrame({"n_wallets": g.size(), "sd_z": g.std(), "excess_kurtosis": g.apply(pd.Series.kurt),
                         "share_|z|>4": g.apply(lambda x: (x.abs() > 4).mean())})


def run(d, k, lik):
    s = np.sqrt(k * d.pq.values / d.n.values + 1e-6)
    return npmle(d.trader_ppv.values, s, lik)


if __name__ == "__main__":
    df = load(); k0 = noise_constant(df); print(f"noise constant k = {k0:.3f}")
    d = df[df.n >= 2]
    tk = tail_diagnostic(df, k0); print(tk.round(3).to_string()); tk.to_csv(OUT + "tail_diagnostic.csv")

    # 1. main spec + robustness over noise scale and likelihood
    rows = []
    for k in [0.7 * k0, k0, 1.0]:
        for lik in ["gauss", "t4"]:
            w, post = run(d, k, "gauss" if lik == "gauss" else "t")
            rows.append(dict(k=round(k, 3), lik=lik, **summarize(w, post, d.trader_label.values)))
            if abs(k - k0) < 1e-9 and lik == "gauss":  # main spec: save prior + per-wallet posteriors
                pd.DataFrame({"theta": GRID, "w": w}).to_csv(OUT + "skill_prior.csv", index=False)
                pd.DataFrame({"trader": d.trader.values, "post_mean": (post * GRID).sum(1),
                              "p_positive": post[:, GRID > 0].sum(1), "p_sharp": post[:, GRID > SHARP].sum(1),
                              "label": d.trader_label.values, "n": d.n.values, "ppv": d.trader_ppv.values}
                             ).to_parquet(OUT + "skill_posterior.parquet")
            print(rows[-1])
    pd.DataFrame(rows).to_csv(OUT + "robustness_noise_lik.csv", index=False)

    # 2. splits: is the skill mass just machines-vs-everyone?
    rows = []
    bands = pd.cut(d.n, [1, 5, 20, 100, 1e9], labels=["2-5", "6-20", "21-100", "100+"])
    for name, mask in [("band " + str(b), bands == b) for b in bands.cat.categories] + \
                      [("topic " + t, d.dom == t) for t in ["sport", "politics", "economy, business and finance", "arts, culture, entertainment and media"]]:
        sub = d[mask.values]
        w, post = run(sub, k0, "gauss")
        rows.append(dict(split=name, **summarize(w, post, sub.trader_label.values))); print(rows[-1])
    pd.DataFrame(rows).to_csv(OUT + "robustness_splits.csv", index=False)
