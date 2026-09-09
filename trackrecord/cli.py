"""CLI: `python -m trackrecord eval trades.csv` or `python -m trackrecord wallet 0x... --data data.parquet`."""
import argparse, json, math, sys, pandas as pd
from .core import evaluate
from .features import from_trades, from_wallet_row

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
    a = ap.parse_args(argv)
    if a.cmd == "eval": agg = from_trades(pd.read_csv(a.csv))
    else:
        df = pd.read_parquet(a.data); row = df[df.trader.str.lower() == a.address.lower()]
        if row.empty: sys.exit(f"wallet {a.address} not in {a.data}")
        agg = from_wallet_row(row.iloc[0])
    rep = evaluate(agg); print(json.dumps(rep, indent=1, default=str) if a.json else card(rep))

if __name__ == "__main__": main()
