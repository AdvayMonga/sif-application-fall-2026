"""CLI: `python -m trackrecord eval trades.csv` or `python -m trackrecord wallet 0x... --data data.parquet`."""
import argparse, json, math, sys, pandas as pd
from .core import evaluate
from .features import from_trades, from_wallet_row
from .returns import evaluate_returns, card_returns
from .diagnostics import extra_fronts

def card_extra(x):
    cal, sz, con, per, rk = x["calibration"], x["sizing"], x["concentration"], x["persistence"], x["risk"]; L = []
    if cal: L.append(f"  calibration           win rate − price {cal['win_minus_price_pts']:+.1f} pts   ·   Brier vs price {cal['brier_price']:.3f}  (constant {cal['brier_constant']:.3f})")
    L.append(f"  sizing value          size-weighted {100*sz['size_weighted_return']:+.1f}¢/$ vs equal-weighted {100*sz['equal_weighted_return']:+.1f}¢/$  →  sizing {'added' if sz['sizing_gain_c_per_dollar']>0 else 'cost'} {abs(sz['sizing_gain_c_per_dollar']):.1f}¢/$")
    bt = f"best trade = {con['best_trade_share']:.0%} of P&L   ·   " if con["best_trade_share"] is not None else ""
    L.append(f"  concentration         {bt}without top 3: ${con['pnl_without_top3']:+,.0f}   ·   bootstrap 90% CI on P&L [${con['pnl_ci90'][0]:+,.0f}, ${con['pnl_ci90'][1]:+,.0f}]")
    if per: L.append(f"  persistence           first half {100*per['first_half_return']:+.1f}¢/$ · second half {100*per['second_half_return']:+.1f}¢/$")
    L.append(f"  risk                  max drawdown {rk['max_drawdown_of_stake']:.1%} of stake   ·   longest losing streak {rk['longest_losing_streak']} (expected ≈{rk['expected_longest_streak']:.0f})   ·   P(−20%) {rk['p_20pct_drawdown_next_horizon']:.0%} / P(−50%) {rk['p_50pct_drawdown_next_horizon']:.1%} over the same number of bets at {rk['bankroll_frac']:.0%} of bankroll per average bet")
    return "\n".join(L)

def card(rep):
    a, p, q = rep["inputs"], rep["posterior"], rep["percentiles_vs_active_wallets"]; c = lambda x: f"{100*x:+.2f}¢"
    need = rep["trades_needed_for_95pct"]; more = "n/a (edge not positive)" if math.isinf(need) else f"{max(need - a['n'], 0):,} more (≈{need:,} total)"
    lines = [f"TRACK RECORD  ·  {a['n']:,} trades  ·  ${a['dollars']:,.0f} wagered  ·  P&L ${a['pnl']:+,.0f}",
             f"  realized edge         {c(a['edge_per_dollar'])} per dollar   ({c(a['edge_per_share'])} per share)",
             f"  luck-adjusted edge    {c(p['mean'])} per share   90% CI [{c(p['ci90'][0])}, {c(p['ci90'][1])}]",
             f"  P(edge > +0.5¢/share) {p['p_meaningful_positive']:.1%}      P(edge < −0.5¢/share) {p['p_meaningful_negative']:.1%}",
             f"  verdict               {rep['verdict']}",
             f"  trades for 95% proof  {more}",
             f"  vs {rep['reference']['n_active_wallets']:,} active wallets:  edge/$ {q['edge_per_dollar']:.0f}th pct  ·  sizing dispersion {q['sizing_dispersion']:.0f}th pct  ·  trades {q['trades']:.0f}th pct  ·  avg price {a['avg_price']:.2f} ({q['avg_price']:.0f}th pct)",
             f"  sizing                " + ("varies bets with conviction (top 30% of population)" if q["sizing_dispersion"] >= 70 else "flat sizing — the population's coin-flipper signature" if q["sizing_dispersion"] <= 30 else "typical dispersion")]
    return "\n".join(lines)

def main(argv=None):
    ap = argparse.ArgumentParser(prog="trackrecord"); sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval", help="evaluate a CSV of trades (price,size,won[,pnl])"); e.add_argument("csv"); e.add_argument("--json", action="store_true")
    w = sub.add_parser("wallet", help="evaluate a wallet address from the SIF parquet"); w.add_argument("address"); w.add_argument("--data", default="data.parquet"); w.add_argument("--json", action="store_true")
    r = sub.add_parser("returns", help="evaluate a series of per-period returns (CSV with a 'return' column, or portfolio 'value' column)"); r.add_argument("csv"); r.add_argument("--periods-per-year", type=int, default=252); r.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "returns":
        d = pd.read_csv(a.csv); ret = d["return"] if "return" in d else d["value"].pct_change().dropna()
        bench = (d["benchmark"] if "benchmark" in d else d["benchmark_value"].pct_change().dropna() if "benchmark_value" in d else None)
        rep = evaluate_returns(ret, a.periods_per_year, bench); print(json.dumps(rep, indent=1, default=str) if a.json else card_returns(rep, "day" if a.periods_per_year == 252 else "period")); return
    if a.cmd == "eval":
        tdf = pd.read_csv(a.csv); agg = from_trades(tdf); rep = evaluate(agg); rep["fronts"] = extra_fronts(tdf)
        print(json.dumps(rep, indent=1, default=str) if a.json else card(rep) + "\n" + card_extra(rep["fronts"])); return
    else:
        df = pd.read_parquet(a.data); row = df[df.trader.str.lower() == a.address.lower()]
        if row.empty: sys.exit(f"wallet {a.address} not in {a.data}")
        agg = from_wallet_row(row.iloc[0])
    rep = evaluate(agg); print(json.dumps(rep, indent=1, default=str) if a.json else card(rep))

if __name__ == "__main__": main()
