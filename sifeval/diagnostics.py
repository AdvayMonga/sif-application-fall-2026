"""Extra fronts for a bet-level record: calibration vs price, sizing value, P&L concentration, split-half persistence, drawdown risk."""
import math, numpy as np, pandas as pd

def _prep(df):
    d = df.copy()
    if "time" in d: d = d.sort_values("time")
    d["shares"] = d["size"] / d["price"]
    if "pnl" not in d: d["pnl"] = np.where(d["won"].astype(bool), d["shares"] * (1 - d["price"]), -d["size"])
    return d.reset_index(drop=True)

def calibration(d):
    """Brier of the price as a forecast of the bettor's own wins vs. the bettor's realized rate; by price bucket."""
    if "won" not in d: return None
    y = d["won"].astype(float).values; p = d["price"].values
    band = pd.cut(d["price"], [0, .2, .4, .6, .8, .95, 1.0], include_lowest=True).astype(str)
    tab = pd.DataFrame({"band": band, "won": y, "p": p}).groupby("band", observed=True).agg(bets=("won", "size"), price=("p", "mean"), win_rate=("won", "mean"))
    tab["gap_pts"] = 100 * (tab.win_rate - tab.price)
    return {"brier_price": float(np.mean((p - y) ** 2)), "brier_constant": float(np.mean((y.mean() - y) ** 2)), "win_minus_price_pts": float(100 * (y.mean() - p.mean())), "by_price": tab.round(3).reset_index().to_dict("records")}

def sizing_value(d):
    """Size-weighted vs equal-weighted return per dollar: positive gap means bigger bets were better bets."""
    r = d["pnl"] / d["size"]; ew = float(r.mean()); sw = float(np.average(r, weights=d["size"]))
    return {"equal_weighted_return": ew, "size_weighted_return": sw, "sizing_gain_c_per_dollar": 100 * (sw - ew)}

def concentration(d, n_boot=2000, seed=0):
    """How much of the P&L is one lucky trade? Bootstrap CI on total P&L by resampling trades."""
    pnl = d["pnl"].values; tot = pnl.sum(); rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(pnl, len(pnl), replace=True).sum() for _ in range(n_boot)])
    top = np.sort(pnl)[::-1]
    return {"total_pnl": float(tot), "best_trade_share": float(top[0] / tot) if tot > 0 else None, "pnl_without_top3": float(tot - top[:3].sum()),
            "pnl_ci90": (float(np.percentile(boots, 5)), float(np.percentile(boots, 95))), "p_total_positive_bootstrap": float((boots > 0).mean())}

def persistence(d):
    """First half vs second half (in time order if a time column exists, else row order)."""
    h = len(d) // 2
    if h < 5: return None
    f = lambda x: float(x["pnl"].sum() / x["size"].sum())
    return {"first_half_return": f(d.iloc[:h]), "second_half_return": f(d.iloc[h:]), "n_half": h}

def drawdown_risk(d, horizon=None, n_sim=5000, seed=0, bankroll_frac=0.05):
    """Max drawdown of cumulative P&L (as a fraction of total staked), longest losing streak vs expectation, and Monte Carlo P(50% drawdown) resampling this record's own trade returns at its own sizes."""
    pnl = d["pnl"].values; stake = d["size"].values; cum = np.cumsum(pnl); dd = (cum - np.maximum.accumulate(np.r_[0, cum])[1:]); mdd = float(dd.min() / stake.sum()) if stake.sum() else 0.0
    wins = pnl > 0; streak = cur = 0
    for w in wins: cur = 0 if w else cur + 1; streak = max(streak, cur)
    p_lose = float(1 - wins.mean()); exp_streak = float(math.log(len(pnl)) / -math.log(p_lose)) if 0 < p_lose < 1 else float("nan")
    rng = np.random.default_rng(seed); ret = pnl / stake; rel = stake / stake.mean(); horizon = horizon or len(pnl)   # each resampled bet risks bankroll_frac x (its size relative to the average bet)
    idx = rng.integers(0, len(ret), (n_sim, horizon)); w = np.cumprod(1 + ret[idx] * rel[idx] * bankroll_frac, axis=1)
    dd = (w / np.maximum.accumulate(w, axis=1)).min(1)
    return {"max_drawdown_of_stake": mdd, "longest_losing_streak": int(streak), "expected_longest_streak": exp_streak, "bankroll_frac": bankroll_frac,
            "p_20pct_drawdown_next_horizon": float((dd < 0.8).mean()), "p_50pct_drawdown_next_horizon": float((dd < 0.5).mean())}

def extra_fronts(df):
    d = _prep(df)
    return {"calibration": calibration(d), "sizing": sizing_value(d), "concentration": concentration(d), "persistence": persistence(d), "risk": drawdown_risk(d)}
