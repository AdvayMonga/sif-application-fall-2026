import math, numpy as np, pandas as pd
from sifeval import evaluate, from_trades, load_reference
from sifeval.core import posterior, trades_needed

def test_zero_edge_is_unproven():
    rep = evaluate({"n": 10, "dollars": 100, "shares": 200, "pnl": 0, "edge_per_share": 0, "edge_per_dollar": 0, "pq": 0.2, "ticket_cv": 0.5, "avg_price": 0.5})
    assert "unproven" in rep["verdict"] and 0.01 < rep["posterior"]["p_positive"] < 0.5

def test_long_positive_record_is_skilled():
    rep = evaluate({"n": 2000, "dollars": 1e5, "shares": 2e5, "pnl": 10000, "edge_per_share": 0.05, "edge_per_dollar": 0.10, "pq": 0.2, "ticket_cv": 2.0, "avg_price": 0.5})
    assert rep["posterior"]["p_positive"] > 0.95 and "skilled" in rep["verdict"]

def test_shrinkage_pulls_small_samples_toward_prior():
    small = posterior(0.10, 0.2, 3); big = posterior(0.10, 0.2, 300)
    assert small["mean"] < big["mean"] <= 0.11

def test_trades_needed_monotone():
    assert trades_needed(0.01, 0.2) > trades_needed(0.05, 0.2) and math.isinf(trades_needed(-0.01, 0.2))

def test_from_trades_resolution_payoff():
    t = pd.DataFrame({"price": [0.5, 0.5], "size": [10, 10], "won": [1, 0]})
    a = from_trades(t); assert a["shares"] == 40 and abs(a["pnl"] - 0.0) < 1e-9 and a["ticket_cv"] == 0.0

def test_reference_prior_is_a_distribution():
    ref = load_reference(); assert abs(sum(ref["prior"]) - 1) < 1e-6 and len(ref["grid"]) == len(ref["prior"])

def test_returns_mode_needs_more_data_for_modest_sharpe():
    from sifeval.returns import evaluate_returns
    rng = np.random.default_rng(0); r = rng.normal(0.0004, 0.01, 60)          # ~Sharpe 0.6, 60 days
    rep = evaluate_returns(r); assert "unproven" in rep["verdict"] and rep["periods_needed_for_95pct"] > 60

def test_returns_mode_recognizes_strong_record():
    from sifeval.returns import evaluate_returns
    rng = np.random.default_rng(1); r = rng.normal(0.002, 0.01, 500)
    rep = evaluate_returns(r); assert rep["p_positive"] > 0.95 and "skilled" in rep["verdict"]

def test_extra_fronts_run_and_flag_concentration():
    from sifeval.diagnostics import extra_fronts
    t = pd.DataFrame({"price": [0.5]*20, "size": [10]*19 + [1000], "won": [0]*10 + [1]*10})
    x = extra_fronts(t); assert x["concentration"]["best_trade_share"] > 0.9 and x["sizing"]["sizing_gain_c_per_dollar"] > 0 and x["risk"]["longest_losing_streak"] == 10

def test_returns_benchmark_beta():
    from sifeval.returns import evaluate_returns
    rng = np.random.default_rng(2); b = rng.normal(0.0004, 0.01, 300); r = 0.5 * b + rng.normal(0, 0.002, 300)
    rep = evaluate_returns(r, benchmark=b); assert 0.4 < rep["beta"] < 0.6 and "p_50pct_drawdown_next_year" in rep
