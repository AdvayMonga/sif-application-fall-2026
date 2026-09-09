"""Empirical-Bayes evaluation of a track record against the 604k-wallet Polymarket population."""
import json, math, numpy as np
from pathlib import Path

_REF = None
def load_reference():
    global _REF
    if _REF is None:
        _REF = json.load(open(Path(__file__).with_name("reference.json"))); p = np.asarray(_REF["prior"]); _REF["prior"] = (p / p.sum()).tolist()
    return _REF

def posterior(edge_per_share, pq, n, ref=None):
    """Posterior over true edge per share given observed edge, price regime E[p(1-p)] and trade count."""
    ref = ref or load_reference(); grid = np.asarray(ref["grid"]); prior = np.asarray(ref["prior"])
    sd = math.sqrt(ref["noise_constant"] * max(pq, 1e-4) / max(n, 1))
    lik = np.exp(-0.5 * ((edge_per_share - grid) / sd) ** 2); post = lik * prior; post /= post.sum()
    cdf = np.cumsum(post); q = lambda a: float(np.interp(a, cdf, grid))
    return {"mean": float((post * grid).sum()), "ci90": (q(0.05), q(0.95)), "p_positive": float(post[grid > 0].sum()),
            "p_meaningful_positive": float(post[grid > 0.005].sum()), "p_meaningful_negative": float(post[grid < -0.005].sum()), "noise_sd": sd}

def percentile(value, table):
    """Percentile (0-100) of value within a 101-point quantile table."""
    return float(np.interp(value, np.asarray(table), np.linspace(0, 100, len(table))))

def trades_needed(edge_per_share, pq, ref=None, z=1.645):
    """Trades required for the observed edge to be 95% significant at the current pace (inf if edge <= 0)."""
    ref = ref or load_reference()
    if edge_per_share <= 0: return math.inf
    return int(math.ceil((z * math.sqrt(ref["noise_constant"] * max(pq, 1e-4)) / edge_per_share) ** 2))

def evaluate(agg, ref=None):
    """agg: dict from features.from_trades / from_wallet_row. Returns the report card as a dict."""
    ref = ref or load_reference(); post = posterior(agg["edge_per_share"], agg["pq"], agg["n"], ref)
    need = trades_needed(agg["edge_per_share"], agg["pq"], ref)
    verdict = ("skilled: edge above +0.5¢/share with P > 0.95" if post["p_meaningful_positive"] > 0.95 else
               "losing: edge below −0.5¢/share with P > 0.95" if post["p_meaningful_negative"] > 0.95 else
               "unproven: record is consistent with luck")
    return {"inputs": agg, "posterior": post, "verdict": verdict, "trades_needed_for_95pct": need,
            "percentiles_vs_active_wallets": {"edge_per_dollar": percentile(agg["edge_per_dollar"], ref["pct"]["edge_per_dollar"]),
                                              "edge_per_share": percentile(agg["edge_per_share"], ref["pct"]["edge_per_share"]),
                                              "sizing_dispersion": percentile(agg["ticket_cv"], ref["pct"]["ticket_cv"]),
                                              "trades": percentile(agg["n"], ref["pct"]["trades"]), "avg_price": percentile(agg["avg_price"], ref["pct"]["avg_price"])},
            "reference": {"n_active_wallets": ref["n_active"], "active_min_trades": ref["active_min_trades"]}}
