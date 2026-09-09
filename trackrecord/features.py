"""Turn a trade list or a dataset row into the aggregates the evaluator needs."""
import numpy as np, pandas as pd

def from_trades(df):
    """df columns: price (0-1, paid per share), size (dollars), won (0/1). Optional pnl (dollars per trade) overrides resolution payoff."""
    d = df.copy(); d["shares"] = d["size"] / d["price"]
    if "pnl" not in d: d["pnl"] = np.where(d["won"].astype(bool), d["shares"] * (1 - d["price"]), -d["size"])
    n = len(d); shares = d.shares.sum(); dollars = d["size"].sum(); pnl = d.pnl.sum()
    return {"n": int(n), "dollars": float(dollars), "shares": float(shares), "pnl": float(pnl), "edge_per_share": float(pnl / shares), "edge_per_dollar": float(pnl / dollars),
            "pq": float((d.price * (1 - d.price)).mean()), "ticket_cv": float(d["size"].std(ddof=0) / d["size"].mean()) if n > 1 else 0.0, "avg_price": float(dollars / shares), "win_rate": float(d["won"].mean()) if "won" in d else None}

def from_wallet_row(r):
    """r: one row of the SIF parquet (pandas Series)."""
    shares = float(r.trader_volume); dollars = float(r.mean_tx_value * r.transaction_count); pnl = float(r.trader_pnl)
    sd = 0.0 if pd.isna(r.std_delta) else float(r.std_delta)
    return {"n": int(r.transaction_count), "dollars": dollars, "shares": shares, "pnl": pnl, "edge_per_share": float(r.trader_ppv), "edge_per_dollar": pnl / dollars,
            "pq": float(max(0.25 - (r.mean_delta ** 2 + sd ** 2), 1e-4)), "ticket_cv": float(0.0 if pd.isna(r.std_tx_value) else r.std_tx_value / max(r.mean_tx_value, 1e-6)), "avg_price": float(min(dollars / shares, 1.0)), "win_rate": None}
