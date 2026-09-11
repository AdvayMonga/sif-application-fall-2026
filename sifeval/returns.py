"""Returns mode: grade a series of per-period returns (a live equity book, a strategy's daily P&L) — is the mean return distinguishable from luck yet?"""
import math, numpy as np, pandas as pd
from scipy import stats

def evaluate_returns(r, periods_per_year=252, benchmark=None):
    """r: array of per-period returns (fractions). Returns mean, Sharpe, t-stat, P(true mean > 0), and periods needed for 95% significance."""
    r = np.asarray(pd.Series(r).dropna(), float); n = len(r); mu = r.mean(); sd = r.std(ddof=1) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan"); t = mu / se if n > 1 and se > 0 else float("nan")
    p_pos = float(stats.t.cdf(t, n - 1)) if n > 1 and not math.isnan(t) else float("nan")            # flat prior on the mean
    sharpe = mu / sd * math.sqrt(periods_per_year) if n > 1 and sd > 0 else float("nan")
    need = math.inf if mu <= 0 else int(math.ceil((1.645 * sd / mu) ** 2))                          # periods for t = 1.645 at the current mean/sd
    # what a 95% band around the CURRENT estimate looks like in annual terms
    lo, hi = (mu - 1.96 * se) * periods_per_year, (mu + 1.96 * se) * periods_per_year
    verdict = ("skilled: mean return positive with P > 0.95" if p_pos > 0.95 else "losing: mean return negative with P > 0.95" if p_pos < 0.05 else "unproven: return is consistent with luck")
    extra = {}
    if n >= 10:
        h = n // 2; extra["first_half_sharpe"] = float(r[:h].mean() / r[:h].std(ddof=1) * math.sqrt(periods_per_year)) if r[:h].std(ddof=1) > 0 else float("nan")
        extra["second_half_sharpe"] = float(r[h:].mean() / r[h:].std(ddof=1) * math.sqrt(periods_per_year)) if r[h:].std(ddof=1) > 0 else float("nan")
        extra["autocorr_lag1"] = float(np.corrcoef(r[:-1], r[1:])[0, 1]); neg = r < 0; streak = cur = 0
        for x in neg: cur = cur + 1 if x else 0; streak = max(streak, cur)
        extra["longest_losing_streak"] = int(streak); extra["expected_longest_streak"] = float(math.log(n) / -math.log(neg.mean())) if 0 < neg.mean() < 1 else float("nan")
        rng = np.random.default_rng(0); sim = np.cumprod(1 + r[rng.integers(0, n, (5000, periods_per_year))], axis=1)
        extra["p_50pct_drawdown_next_year"] = float(((sim / np.maximum.accumulate(sim, axis=1)).min(1) < 0.5).mean()); extra["p_20pct_drawdown_next_year"] = float(((sim / np.maximum.accumulate(sim, axis=1)).min(1) < 0.8).mean())
    if benchmark is not None:
        b = np.asarray(pd.Series(benchmark).dropna(), float)[-n:]
        if len(b) == n and b.std() > 0:
            beta = float(np.cov(r, b)[0, 1] / b.var(ddof=1)); alpha = float((r - beta * b).mean() * periods_per_year); resid = r - beta * b
            extra["beta"] = beta; extra["alpha_annualized"] = alpha; extra["alpha_t_stat"] = float(resid.mean() / (resid.std(ddof=1) / math.sqrt(n))) if resid.std(ddof=1) > 0 else float("nan")
    return {"n_periods": n, **extra, "mean_per_period": mu, "sd_per_period": sd, "annualized_return": mu * periods_per_year, "annualized_return_ci95": (lo, hi), "sharpe": sharpe,
            "t_stat": t, "p_positive": p_pos, "periods_needed_for_95pct": need, "max_drawdown": float(_mdd(r)), "verdict": verdict}

def _mdd(r):
    w = np.cumprod(1 + r); return (w / np.maximum.accumulate(w) - 1).min() if len(w) else 0.0

def card_returns(rep, label="period"):
    a = rep; need = a["periods_needed_for_95pct"]; more = "n/a (mean not positive)" if math.isinf(need) else f"{max(need - a['n_periods'], 0):,} more (≈{need:,} total)"
    ex = [l for l in [
        f"  persistence                 first-half Sharpe {a['first_half_sharpe']:.2f} / second-half {a['second_half_sharpe']:.2f}" if "first_half_sharpe" in a else None,
        f"  losing streak               longest {a['longest_losing_streak']} {label}s (expected ≈{a['expected_longest_streak']:.0f} by chance)   ·   autocorr(1) {a['autocorr_lag1']:+.2f}" if "longest_losing_streak" in a else None,
        f"  drawdown risk, next year    P(−20%) {a['p_20pct_drawdown_next_year']:.0%}   P(−50%) {a['p_50pct_drawdown_next_year']:.1%}  (resampling this record's own days)" if "p_50pct_drawdown_next_year" in a else None,
        f"  vs benchmark                beta {a['beta']:.2f}   alpha {a['alpha_annualized']:+.1%}/yr (t = {a['alpha_t_stat']:.2f})" if "beta" in a else None] if l]
    return "\n".join([f"RETURN SERIES  ·  {a['n_periods']:,} {label}s  ·  annualized {a['annualized_return']:+.1%}  ·  Sharpe {a['sharpe']:.2f}  ·  max drawdown {a['max_drawdown']:.1%}",
                      f"  95% band on annual return   [{a['annualized_return_ci95'][0]:+.1%}, {a['annualized_return_ci95'][1]:+.1%}]",
                      f"  t-stat / P(true mean > 0)   {a['t_stat']:.2f} / {a['p_positive']:.1%}", f"  verdict                     {a['verdict']}", f"  {label}s for 95% proof        {more}"] + ex)
