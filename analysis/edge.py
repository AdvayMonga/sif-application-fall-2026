"""Part 1 farm footprint + Part 2 extreme-favorite edge: tables for the memo (dataset-only)."""
import json, numpy as np, pandas as pd
from common import load, singles

OUT = "/Users/advaymonga/Desktop/sif/sif-application-fall-2026/analysis/out/"
BANDS = [.90, .95, .98, .99, .995, 1.0]
RATE = {"sport": .03, "politics": .04, "economy, business and finance": .06, "arts, culture, entertainment and media": .05, "weather": .05, "conflict, war and peace": 0.0}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def band_table(d):
    g = d.groupby(pd.cut(d.p, BANDS, right=False), observed=True)
    t = pd.DataFrame({"n": g.size(), "dollars": g.mean_tx_value.sum(), "implied_loss": g.p.apply(lambda x: (1 - x).mean()),
                      "realized_loss": g.won.apply(lambda x: 1 - x.mean()), "pnl": g.trader_pnl.sum()})
    t["loss_lo"], t["loss_hi"] = wilson(t.n - g.won.sum(), t.n)
    t["edge_c_per_dollar"] = 100 * t.pnl / t.dollars
    # 2026 taker fee as cents per dollar spent: rate*(1-p), topic-weighted
    t["fee_c_per_dollar_2026"] = g.apply(lambda x: 100 * np.average(x.dom.map(RATE).fillna(.04) * (1 - x.p), weights=x.mean_tx_value))
    return t


if __name__ == "__main__":
    df = load(); s = singles(df); r = s[s.outcome != "open"].copy()
    r["hr"] = (r.mean_time // 3.6e6).astype(int)
    key = r.mean_time.astype(int).astype(str) + "|" + r.p.round(3).astype(str) + "|" + r.dom
    r["batch"] = key.map(key.value_counts()) >= 3
    clean = r[(r.p >= .90) & (~r.batch) & (r.trader_pnl.abs() < 10000)].copy()
    summary = {}

    # ---- Part 2: the edge ----
    t = band_table(clean); t.to_csv(OUT + "edge_bands.csv"); print(t.round(4).to_string())
    win = {"02-06 UTC (US games end)": clean.hr.between(2, 6), "08-14 UTC": clean.hr.between(8, 14), "15-23 UTC": clean.hr.between(15, 23)}
    core = clean[(clean.p >= .95) & (clean.p < .995)]
    def cell(x): return pd.Series({"n": len(x), "implied_loss": (1 - x.p).mean(), "realized_loss": 1 - x.won.mean(), "edge_c_per_dollar": 100 * x.trader_pnl.sum() / x.mean_tx_value.sum(), "dollars": x.mean_tx_value.sum()})
    by_hour = pd.DataFrame({k: cell(core[m]) for k, m in win.items()}).T; by_hour.to_csv(OUT + "edge_by_hour.csv")
    by_topic = core.groupby("dom").apply(cell).query("n>=150").sort_values("n", ascending=False); by_topic.to_csv(OUT + "edge_by_topic.csv")
    by_size = core.groupby(pd.cut(core.mean_tx_value, [0, 2, 10, 50, 200, 1000, 1e9]), observed=True).apply(cell); by_size.to_csv(OUT + "edge_by_size.csv")
    print(by_hour.round(4)); print(by_topic.round(4)); print(by_size.round(4))
    # 90-95c control (no edge) and 85-90 for contrast
    ctrl = r[(r.p >= .90) & (r.p < .95) & (~r.batch) & (r.trader_pnl.abs() < 10000)]
    summary["ctrl_90_95"] = dict(n=len(ctrl), implied=float((1 - ctrl.p).mean()), realized=float(1 - ctrl.won.mean()), edge=float(100 * ctrl.trader_pnl.sum() / ctrl.mean_tx_value.sum()))

    # repeat wallets living at 90-98.5c, not always-on
    df["avgp"] = df.notional / df.trader_volume
    ao = (df.std_time / 3.6e6).between(6.3, 7.6) & (df.mean_time / 3.6e6).between(10.5, 13.5)
    rep = df[(df.n >= 20) & ~ao & df.avgp.between(.90, .985)]
    hold = (rep.trader_pnl.abs() / ((1 - rep.avgp) * rep.trader_volume)) >= 0.5
    g = rep.groupby(pd.cut(rep.avgp, [.90, .93, .96, .975, .985]), observed=True)
    repeat = pd.DataFrame({"wallets": g.size(), "notional_M": g.notional.sum() / 1e6, "c_per_dollar": 100 * g.trader_pnl.sum() / g.notional.sum(), "frac_profitable": g.trader_pnl.apply(lambda x: (x > 0).mean())})
    repeat.to_csv(OUT + "edge_repeat_wallets.csv"); print(repeat.round(3))
    summary["holders"] = dict(n=int(hold.sum()), notional_M=float(rep[hold].notional.sum() / 1e6), c_per_dollar=float(100 * rep[hold].trader_pnl.sum() / rep[hold].notional.sum()), frac_profitable=float((rep[hold].trader_pnl > 0).mean()))

    # Monte Carlo
    rng = np.random.default_rng(0)
    P = core.p.mean(); L = 1 - core.won.mean(); Limp = (1 - core.p).mean(); b = (1 - P) / P
    Lhi = wilson(len(core) - core.won.sum(), len(core))[1]
    summary["mc"] = dict(P=float(P), realized_loss=float(L), loss_hi=float(Lhi), implied_loss=float(Limp), b=float(b), edge_per_bet_pct=float(100 * ((1 - L) * b - L)), kelly=float(max(0, (1 - L) - L / b)))
    rows = []; paths = {}
    for lname, loss in [("realized", L), ("realized upper CI", Lhi), ("if price were honest", Limp)]:
        for f in [0.10, 0.25, 0.50, 1.00]:
            wins = rng.random((20000, 200)) > loss
            w = np.cumprod(np.where(wins, 1 + f * b, 1 - f), axis=1)
            dd = (w / np.maximum.accumulate(w, axis=1)).min(1)
            rows.append(dict(loss_scenario=lname, loss=loss, f=f, median_final=float(np.median(w[:, -1])), p_loss=float((w[:, -1] < 1).mean()), p_dd50=float((dd < .5).mean()), p_dd90=float((dd < .1).mean())))
            if lname == "realized": paths[f] = np.percentile(w, [10, 50, 90], axis=0)
    mc = pd.DataFrame(rows); mc.to_csv(OUT + "edge_montecarlo.csv", index=False); print(mc.round(3))
    np.save(OUT + "edge_mc_paths.npy", np.array([paths[f] for f in [0.10, 0.25, 0.50, 1.00]]))
    cap = df[df.avgp.between(.95, .995)]
    summary["capacity"] = dict(wallets=int(len(cap)), notional_M=float(cap.notional.sum() / 1e6), singles_k=float(core.mean_tx_value.sum() / 1e3))
    summary["wash_pair"] = dict(shares=96576.17, price=0.90, loss=86918.55)

    # ---- Part 1: farm footprint ----
    pure99 = df[(df.mean_delta >= .49) & (df.avgp >= .9)]
    maker0 = df[(df.n >= 4) & (df.price_levels_per_transaction == 0) & (df.trader_pnl.abs() <= 0.001 * df.notional)]
    exact0 = df[(df.n >= 3) & (df.trader_pnl == 0)]
    hedged = df[(df.n == 2) & (df.trader_pnl == 0) & (df.std_delta == 0)]
    sec_key = s.mean_time.astype(int).astype(str) + "|" + s.p.round(3).astype(str) + "|" + s.dom
    sec_batch = s[sec_key.map(sec_key.value_counts()) >= 3]
    q100 = s[(s.p.round(3) == .999) & s.trader_volume.round(1).isin([100.5, 101.0])]
    summary["farm"] = dict(
        pure99_wallets=int(len(pure99)), pure99_notional_M=float(pure99.notional.sum() / 1e6), pure99_pnl=float(pure99.trader_pnl.sum()),
        maker0_wallets=int(len(maker0)), maker0_notional_M=float(maker0.notional.sum() / 1e6), maker0_tx_day=float(maker0.transactions_per_day.median()), maker0_mkts_day=float(maker0.markets_per_day.median()), maker0_share_ge95=float((maker0.avgp >= .95).mean()),
        exact0_wallets=int(len(exact0)), exact0_notional_M=float(exact0.notional.sum() / 1e6), hedged_pairs=int(len(hedged)),
        sec_batch_wallets=int(len(sec_batch)), sec_batch_share=float(len(sec_batch) / len(s)), q100_wallets=int(len(q100)), q100_pnl=float(q100.trader_pnl.sum()),
        resolved98_n=int(((r.p >= .98)).sum()), resolved98_lost=int((r[r.p >= .98].won == 0).sum()))
    json.dump(summary, open(OUT + "edge_summary.json", "w"), indent=1, default=float); print(json.dumps(summary, indent=1, default=float))
