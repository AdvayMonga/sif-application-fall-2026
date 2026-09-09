"""Returns mode: grade a series of per-period returns (a live equity book, a strategy's daily P&L) — is the mean return distinguishable from luck yet?"""
import math, numpy as np, pandas as pd
from scipy import stats

def evaluate_returns(r, periods_per_year=252, target_sharpe=None):
    """r: array of per-period returns (fractions). Returns mean, Sharpe, t-stat, P(true mean > 0), and periods needed for 95% significance."""
    r = np.asarray(pd.Series(r).dropna(), float); n = len(r); mu = r.mean(); sd = r.std(ddof=1) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan"); t = mu / se if n > 1 and se > 0 else float("nan")
    p_pos = float(stats.t.cdf(t, n - 1)) if n > 1 and not math.isnan(t) else float("nan")            # flat prior on the mean
    sharpe = mu / sd * math.sqrt(periods_per_year) if n > 1 and sd > 0 else float("nan")
    need = math.inf if mu <= 0 else int(math.ceil((1.645 * sd / mu) ** 2))                          # periods for t = 1.645 at the current mean/sd
    # what a 95% band around the CURRENT estimate looks like in annual terms
    lo, hi = (mu - 1.96 * se) * periods_per_year, (mu + 1.96 * se) * periods_per_year
    verdict = ("skilled: mean return positive with P > 0.95" if p_pos > 0.95 else "losing: mean return negative with P > 0.95" if p_pos < 0.05 else "unproven: return is consistent with luck")
    return {"n_periods": n, "mean_per_period": mu, "sd_per_period": sd, "annualized_return": mu * periods_per_year, "annualized_return_ci95": (lo, hi), "sharpe": sharpe,
            "t_stat": t, "p_positive": p_pos, "periods_needed_for_95pct": need, "max_drawdown": float(_mdd(r)), "verdict": verdict}

def _mdd(r):
    w = np.cumprod(1 + r); return (w / np.maximum.accumulate(w) - 1).min() if len(w) else 0.0

def card_returns(rep, label="period"):
    a = rep; need = a["periods_needed_for_95pct"]; more = "n/a (mean not positive)" if math.isinf(need) else f"{max(need - a['n_periods'], 0):,} more (≈{need:,} total)"
    return "\n".join([f"RETURN SERIES  ·  {a['n_periods']:,} {label}s  ·  annualized {a['annualized_return']:+.1%}  ·  Sharpe {a['sharpe']:.2f}  ·  max drawdown {a['max_drawdown']:.1%}",
                      f"  95% band on annual return   [{a['annualized_return_ci95'][0]:+.1%}, {a['annualized_return_ci95'][1]:+.1%}]",
                      f"  t-stat / P(true mean > 0)   {a['t_stat']:.2f} / {a['p_positive']:.1%}", f"  verdict                     {a['verdict']}", f"  {label}s for 95% proof        {more}"])
