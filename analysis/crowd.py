"""How wrong is the crowd: realized win rate vs price paid on resolved single bets, with CIs and weighting-function fits."""
import numpy as np, pandas as pd
from scipy.optimize import minimize, minimize_scalar
from common import load, singles

OUT = "/Users/advaymonga/Desktop/sif/sif-application-fall-2026/analysis/out/"
rng = np.random.default_rng(0)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def calib_table(r, bins):
    g = r.groupby(pd.cut(r.p, bins, include_lowest=True), observed=True)
    t = g.agg(n=("won", "size"), price=("p", "mean"), win=("won", "mean"), wins=("won", "sum"), dollars=("mean_tx_value", "sum"))
    lo, hi = wilson(t.wins, t.n); t["win_lo"], t["win_hi"] = lo, hi
    t["gap_pts"] = 100 * (t.win - t.price)
    return t


# --- probability-weighting fits: true prob q implied by price p if buyers apply w(q)=p ---
def prelec_inv(p, a): return np.exp(-(-np.log(p)) ** (1 / a))
_q = np.linspace(1e-4, 1 - 1e-4, 4000)
def ge_inv(p, gm, dl):
    wq = dl * _q**gm / (dl * _q**gm + (1 - _q) ** gm); return np.interp(p, wq, _q)


def fit(r):
    bins = np.quantile(r.p, np.linspace(0, 1, 21)); t = calib_table(r, bins)
    a = minimize_scalar(lambda a: np.sum(t.n * (t.win - prelec_inv(t.price, a)) ** 2), bounds=(0.2, 3), method="bounded").x
    ge = minimize(lambda x: np.sum(t.n * (t.win - ge_inv(t.price, *x)) ** 2), [0.8, 1.5], bounds=[(0.2, 3), (0.1, 5)]).x
    return a, ge[0], ge[1]


if __name__ == "__main__":
    df = load(); s = singles(df); res = s[s.outcome != "open"]
    print(f"single-trade wallets {len(s):,}; resolved {len(res):,} ({len(res)/len(s):.1%})")

    # 1. full-range calibration (fixed bins) incl. the sure-thing region
    full = calib_table(res, [0, .02, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .98, 1.0])
    full.to_csv(OUT + "calibration_full.csv"); print(full.round(3).to_string())

    # 2. mid-range gap with bootstrap CI, count- and dollar-weighted
    mid = res[(res.p > 0.2) & (res.p < 0.9)]
    gap = lambda d: 100 * (d.won.mean() - d.p.mean())
    gap_d = lambda d: 100 * (np.average(d.won, weights=d.mean_tx_value) - np.average(d.p, weights=d.mean_tx_value))
    bs = np.array([[gap(b), gap_d(b)] for b in (mid.sample(len(mid), replace=True, random_state=i) for i in range(1000))])
    summary = {"mid_n": len(mid), "gap_pts": gap(mid), "gap_ci": np.percentile(bs[:, 0], [2.5, 97.5]).tolist(),
               "gap_dollar_pts": gap_d(mid), "gap_dollar_ci": np.percentile(bs[:, 1], [2.5, 97.5]).tolist()}
    print(summary)

    # 3. by topic
    bt = res.groupby("dom").apply(lambda d: pd.Series({"n": len(d), "mid_n": ((d.p > .2) & (d.p < .9)).sum(),
                                                       "mid_gap_pts": gap(d[(d.p > .2) & (d.p < .9)]) if ((d.p > .2) & (d.p < .9)).sum() > 100 else np.nan}))
    bt = bt.sort_values("n", ascending=False); bt.to_csv(OUT + "calibration_by_topic.csv"); print(bt.round(2).head(8).to_string())

    # 4. the 50c trap: exact 0.50 vs neighbours, by topic
    at50 = res[(res.p >= 0.495) & (res.p <= 0.505)]; nb = res[((res.p >= 0.44) & (res.p < 0.495)) | ((res.p > 0.505) & (res.p <= 0.56))]
    rows = []
    for name, d in [("p=0.50", at50), ("neighbours 0.44-0.56", nb)]:
        lo, hi = wilson(d.won.sum(), len(d)); rows.append(dict(group=name, n=len(d), win=d.won.mean(), lo=lo, hi=hi, avg_price=d.p.mean()))
    for t_, d in at50.groupby("dom"):
        if len(d) >= 50: lo, hi = wilson(d.won.sum(), len(d)); rows.append(dict(group="p=0.50 " + t_, n=len(d), win=d.won.mean(), lo=lo, hi=hi, avg_price=0.5))
    trap = pd.DataFrame(rows); trap.to_csv(OUT + "trap50.csv", index=False); print(trap.round(3).to_string())
    # is it the favorite side or longshot side? at exactly 0.5 side is ambiguous; check exact price distribution
    print("exact p==0.5:", (res.p.round(4) == 0.5).sum(), " won:", res[res.p.round(4) == 0.5].won.mean().round(3))

    # 5. weighting-function fits with bootstrap CIs (0.02<p<0.95, excludes post-outcome scoops)
    r = res[(res.p > 0.02) & (res.p < 0.95)]
    a, g_, d_ = fit(r)
    B = np.array([fit(r.sample(len(r), replace=True, random_state=i)) for i in range(200)])
    ci = np.percentile(B, [2.5, 97.5], axis=0)
    fits = pd.DataFrame({"param": ["prelec_alpha", "ge_gamma", "ge_delta"], "est": [a, g_, d_], "lo": ci[0], "hi": ci[1]})
    fits.to_csv(OUT + "weighting_fits.csv", index=False); print(fits.round(3).to_string())
    pd.Series(summary).to_json(OUT + "crowd_summary.json")
    calib_table(r, np.quantile(r.p, np.linspace(0, 1, 21))).to_csv(OUT + "calibration_quantile.csv")
