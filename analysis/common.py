"""Load the dataset and decode its undocumented columns."""
import pathlib, numpy as np, pandas as pd

DATA = str(pathlib.Path(__file__).resolve().parent.parent / "data.parquet")
TOPICS = None


def load():
    global TOPICS
    df = pd.read_parquet(DATA)
    TOPICS = [c for c in df if c.startswith("topic_")]
    df["n"] = df.transaction_count.astype(float)
    df["dom"] = df[TOPICS].idxmax(axis=1).str.replace("topic_", "")
    # E[p(1-p)] per wallet from the two moments of |p-0.5|; drives per-trade noise
    df["pq"] = (0.25 - (df.mean_delta**2 + df.std_delta.fillna(0) ** 2)).clip(1e-4, 0.25)
    df["notional"] = df.mean_tx_value * df.transaction_count  # dollars (volume is shares)
    return df


def singles(df):
    """Single-trade wallets with price paid, side and resolution recovered."""
    s = df[df.transaction_count == 1].copy()
    s["p"] = s.mean_tx_value / s.trader_volume  # dollars / shares
    win = (s.trader_ppv - (1 - s.p)).abs() < 0.0015
    lose = (s.trader_ppv + s.p).abs() < 0.0015
    s["outcome"] = np.select([win, lose], ["won", "lost"], "open")
    s["won"] = win.astype(int)
    return s


def noise_constant(df):
    """Var(ppv | n=1) = k * p(1-p): fit k on single-trade wallets."""
    o = df[df.n == 1]
    b = pd.cut(o.pq, [0, .01, .05, .1, .15, .2, .25])
    g = o.groupby(b, observed=True).agg(pq=("pq", "mean"), var=("trader_ppv", "var"), cnt=("pq", "size"))
    k, _ = np.polyfit(g.pq, g["var"], 1, w=np.sqrt(g.cnt))
    return float(k)
