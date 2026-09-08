"""Step 1: behavior-only wallet features (nothing derived from pnl/ppv/label). Writes out/ml/features.parquet."""
import numpy as np, pandas as pd
from common import load
OUT = "/Users/advaymonga/Desktop/sif/sif-application-fall-2026/analysis/out/ml/"
df = load()
tp = [c for c in df if c.startswith("topic_")]
F = pd.DataFrame(index=df.index)
F["log_n"] = np.log10(df.n); F["log_tx_day"] = np.log10(df.transactions_per_day); F["log_mkts_day"] = np.log10(df.markets_per_day.clip(1e-3))
F["burst"] = np.log10(df.transactions_per_day / df.markets_per_day.clip(1e-3))          # trades per market-day
F["log_ticket"] = np.log10(df.mean_tx_value.clip(1e-3)); F["ticket_cv"] = (df.std_tx_value / df.mean_tx_value.clip(1e-6)).fillna(0).clip(0, 20)
F["avg_price"] = (df.notional / df.trader_volume).clip(0, 1); F["mean_delta"] = df.mean_delta; F["std_delta"] = df.std_delta.fillna(0)
F["price_side"] = np.sign(F.avg_price - 0.5)                                            # favorites vs longshots
F["hour_sin"] = np.sin(2 * np.pi * df.mean_time / 86.4e6); F["hour_cos"] = np.cos(2 * np.pi * df.mean_time / 86.4e6)
F["time_spread_h"] = (df.std_time / 3.6e6).fillna(0); F["vw_time_gap_h"] = ((df.mean_time_vw - df.mean_time).abs() / 3.6e6).fillna(0).clip(0, 12)
F["aggr"] = np.log10(df.price_levels_per_transaction + 1e-4); F["aggr_vw"] = np.log10(df.price_levels_vw_per_transaction + 1e-4); F["maker_like"] = (df.price_levels_per_transaction == 0).astype(float)
P = df[tp].clip(1e-9, 1); F["topic_entropy"] = -(P * np.log(P)).sum(axis=1); F["topic_max"] = df.largest_transformers_topic_share; F["tag_agree"] = 1 - (df.largest_transformers_topic_share - df.largest_tags_topic_share).abs()
for c in ["sport", "politics", "economy, business and finance", "arts, culture, entertainment and media", "weather"]: F["t_" + c.split(",")[0]] = df["topic_" + c]
F["round_ticket"] = df.mean_tx_value.round(2).isin([1, 2, 5, 10, 20, 25, 50, 100, 200, 500, 1000]).astype(float)
F["int_shares_single"] = ((df.n == 1) & ((df.trader_volume - df.trader_volume.round()).abs() < 0.005)).astype(float)
F["trader"] = df.trader.values
F.to_parquet(OUT + "features.parquet"); print(F.shape); print(F.drop(columns="trader").describe().T[["mean", "std", "min", "max"]].round(3).to_string())
