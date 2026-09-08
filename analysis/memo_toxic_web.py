"""Web memo: 'the risk at 98c+ is one bot'. Reuses the edge memo's CSS; inline SVG figures from out/toxic_*."""
import json, re, pandas as pd, numpy as np
OUT = "/Users/advaymonga/Desktop/sif/sif-application-fall-2026/analysis/out/"
S = json.load(open(OUT + "toxic_summary.json")); B = pd.read_csv(OUT + "toxic_bands.csv", index_col=0); Q = pd.read_csv(OUT + "toxic_quintiles.csv", index_col=0); P = pd.read_csv(OUT + "toxic_pattern.csv")
css = re.search(r"<style>.*?</style>", open(OUT + "memo_edge_artifact.html").read(), re.S).group(0)
pct = lambda x, d=1: f"{100*x:.{d}f}%"
R, O, L, MR, MC = S["rule"], S["others"], S["losers"], S["mc_raw"], S["mc_clean"]

def bars(vals, labels, ylab, title, fmt=lambda v: f"{v:g}", ymax=None, colors=None, W=680, H=280, L_=56, T=22, Bm=40):
    ymax = ymax or max(vals) * 1.18; n = len(vals); bw = min(84, (W - L_ - 16) / n * 0.7)
    ys = lambda v: T + (H - T - Bm) * (1 - v / ymax); xs = lambda i: L_ + (W - L_ - 16) * (i + 0.5) / n; base = ys(0)
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{title}">']
    step = ymax / 4
    for k in range(1, 5):
        v = k * step; o.append(f'<line x1="{L_}" x2="{W-16}" y1="{ys(v):.1f}" y2="{ys(v):.1f}" class="grid"/><text x="{L_-8}" y="{ys(v)+4:.1f}" class="tick" text-anchor="end">{fmt(v)}</text>')
    for i, (v, lab) in enumerate(zip(vals, labels)):
        x = xs(i); c = (colors[i] if colors else "bar")
        o.append(f'<g class="mark"><title>{lab}: {fmt(v)}</title><rect x="{x-bw/2:.1f}" y="{ys(v):.1f}" width="{bw:.1f}" height="{max(base-ys(v),0):.1f}" rx="3" class="{c}"/><text x="{x:.1f}" y="{ys(v)-7:.1f}" class="lbl strong" text-anchor="middle">{fmt(v)}</text></g>')
        o.append(f'<text x="{x:.1f}" y="{base+18}" class="tick" text-anchor="middle">{lab}</text>')
    o.append(f'<line x1="{L_}" x2="{W-16}" y1="{base}" y2="{base}" class="axis"/></svg>'); return "".join(o)

fig1 = bars(Q.losses.tolist(), ["lowest risk", "2", "3", "4", "highest risk"], "losses", "Losses by model risk quintile", fmt=lambda v: f"{v:.0f}", ymax=90)
sp = P[(P.sport == 1) & (P.int_shares == 1)].set_index("bucket").reindex(["<5", "5–10", "10–15", "15–20", "20–50", "50–100", "100+"])
ot = P[~((P.sport == 1) & (P.int_shares == 1))].groupby("bucket").apply(lambda x: x.losses.sum() / x.n.sum()).reindex(sp.index)
fig2 = bars((100 * sp.loss_rate.fillna(0)).tolist(), list(sp.index), "loss rate", "Sport bets with whole-share sizes, 97–99.5¢: loss rate by size", fmt=lambda v: f"{v:.1f}%", ymax=35)
hh = [L["hour_hist"].get(str(h), 0) for h in range(24)]
fig3 = bars(hh, [f"{h:02d}" for h in range(24)], "losers", "When the 77 losing fills happened (UTC hour)", fmt=lambda v: f"{v:.0f}", ymax=16, W=680, H=240)
band_rows = "".join(f"<tr><td>{n}</td><td>{int(r.n):,}</td><td>{pct(r.implied)}</td><td>{pct(r.realized,2)}</td><td>{pct(r.lo,2)}–{pct(r.hi,2)}</td><td>{r.edge:+.2f}</td></tr>" for n, (_, r) in zip(["95–98¢", "98–99¢", "99–99.5¢", "99.5–100¢"], B.iterrows()))
quint_rows = "".join(f"<tr><td>{n}</td><td>{int(r.n):,}</td><td>{int(r.losses)}</td><td>{pct(r.loss_rate,2)}</td><td>{pct(r.implied,2)}</td><td>{r.edge:+.2f}</td></tr>" for n, (_, r) in zip(["lowest", "2", "3", "4", "highest"], Q.iterrows()))

page = f"""<title>SIF Application, Fall 2026 (v2)</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
{css}
<div class="wrap">
<h1>SIF Application, Fall 2026</h1>
<p class="sub">Advay Monga · 604,578 Polymarket wallets, June 2024 – March 2025. Every number comes from the provided file alone.</p>

<h2>1. Key figures</h2>
<h3>Figure 1 — A model knows which 98¢+ bets will lose before they resolve</h3>
{fig1}
<div class="cap">{S['n98']:,} resolved bets at 98¢ and above, {S['losses98']} of which lost. Bets are sorted into five equal groups by a cross-validated model's predicted risk. The top group holds {pct(S['top_quintile_share_of_losses'],0)} of all losses (AUC {S['auc']:.2f}).</div>
<h3>Figure 2 — The losses are one pattern: sport, whole-share lots of 10–15</h3>
{fig2}
<div class="cap">Sport bets at 97–99.5¢ sized in whole shares, grouped by lot size. The 10–15 share lot loses {pct(R['loss_rate'],0)} of the time; every other size loses about 1% or less.</div>
<h3>Figure 3 — They lose when games end</h3>
{fig3}
<div class="cap">Hour of day (UTC) of the {L['n']} losing fills. Peaks at 02–05 UTC (US games ending) and 17–23 UTC (European evening, US afternoon).</div>
<div class="kpi">
 <div><b>{pct(S['implied98'],2)}</b><span>loss rate the price implies at 98¢+</span></div>
 <div><b>{pct(S['realized98'],2)}</b><span>loss rate that actually happened</span></div>
 <div><b>{pct(O['loss_rate'],2)}</b><span>loss rate once the one pattern is removed ({int(O['n']):,} bets)</span></div>
 <div><b>{S['auc']:.2f}</b><span>AUC of the loss model, 5-fold cross-validated</span></div>
</div>

<h2>2. The conclusion</h2>
<p><strong>At 98¢ and above, the risk is not chance. It is one bot getting picked off.</strong> Remove its fills and a buyer of near-certain favorites loses {pct(O['loss_rate'],2)} of the time against the {pct(O['implied'],2)} the price implies — a seventh of the stated risk, worth {O['edge']:+.2f}¢ per dollar with almost no tail.</p>

<h2>3. How I got there</h2>
<h3>Step 1 — Establish the edge</h3>
<p>For 75,855 single-trade wallets the price paid and the outcome can be recovered. Line resolved bets up by price. Above 95¢, far fewer lose than the price says. The gap is the starting point, not the finding.</p>
<div class="tw"><table><thead><tr><th>price band</th><th>bets</th><th>implied loss</th><th>realized loss</th><th>95% CI</th><th>edge ¢/$</th></tr></thead><tbody>{band_rows}</tbody></table></div>
<h3>Step 2 — Ask whether the losses are random</h3>
<p>If the residual {pct(S['realized98'],2)} loss rate were pure chance, no model could tell which bets would be the losers. A gradient-boosted classifier on the bet's fingerprint — price, hour, ticket size, whole-share sizing, whether it crossed the spread, topic — predicts the losers with cross-validated AUC {S['auc']:.2f}. {pct(S['top_quintile_share_of_losses'],0)} of losses sit in the top risk quintile.</p>
<div class="tw"><table><thead><tr><th>risk quintile</th><th>bets</th><th>losses</th><th>loss rate</th><th>implied</th><th>edge ¢/$</th></tr></thead><tbody>{quint_rows}</tbody></table></div>
<h3>Step 3 — Find the pattern the model found</h3>
<p>Open the model up and one rule carries it: <strong>sport, an integer number of shares, 10 to 15 of them</strong>. {R['n']} bets match. {R['losses']} of them lost ({pct(R['loss_rate'],0)} against {pct(R['implied'])} implied) — {pct(S['rule_share_of_losses'],0)} of every loss at 98¢+. The other {O['n']:,} bets lost {O['losses']} times: {pct(O['loss_rate'],2)}.</p>
<h3>Step 4 — Explain the pattern</h3>
<ul>
<li><strong>They were all resting orders.</strong> {pct(L['share_maker'],0)} of the losing fills never crossed the spread. These are bids sitting on the book, not buyers pressing a button.</li>
<li><strong>Same size, same prices, same hours.</strong> Median {L['median_shares']:.0f} shares, priced at 98.4–99.0¢, {pct(L['share_sport'],0)} in sport, clustered in the hours games end (figure 3).</li>
<li><strong>Someone is sweeping them.</strong> A bid resting at 98.6¢ on a live game is stale the moment the game turns. A faster taker sells into it, keeps the complement, and the resting bot eats the −100%. That is the entire loss rate.</li>
</ul>

<h2>4. What it means</h2>
<ul>
<li><strong>For a buyer:</strong> take the favorite by crossing the spread near resolution, not by resting a bid on a live market. Loss rate {pct(O['loss_rate'],2)}, edge {O['edge']:+.2f}¢/$. Over 200 bets at 25% of bankroll each: {pct(MC['p0'],0)} chance of zero losses (×{MC['mult25_0']:.2f}), {pct(MC['p1'],0)} of one (×{MC['mult25_1']:.2f}), {pct(MC['p2plus'],0)} of two or more (×{MC['mult25_2']:.2f} or worse).</li>
<li><strong>For a market maker:</strong> the whole risk at these prices is being stale on live sports. Pull or reprice bids at game-turn moments and the book earns the {pct(O['implied'],2)}-vs-{pct(O['loss_rate'],2)} spread with almost no tail.</li>
<li><strong>For the fund:</strong> "rare loss" and "random loss" are different things. Here the rare losses had a fingerprint, and the fingerprint was one counterparty's mistake.</li>
</ul>

<h2>5. Limitations</h2>
<ul>
<li>Outcomes come from single-trade wallets that held to resolution; 15% of single bets were still open and are excluded.</li>
<li>{S['losses98']} losses is a small count. The AUC and the rule are cross-validated, but the dollar figures are small (${R['dollars']:,.0f} matched the rule) — this is a mechanism finding, not a capacity finding.</li>
<li>Whether the same bot is still running, or was banned with the March 2026 wash-trading rules, is not knowable from this file.</li>
</ul>

<h2>Method</h2>
<ul class="small">
<li><code>trader_volume</code> is shares; <code>mean_tx_value</code> is dollars per trade; price = dollars ÷ shares. For single-trade wallets, 85% of <code>trader_pnl</code> equals a resolution payoff, which identifies the outcome.</li>
<li>Same-second identical bets (3+ wallets, same price and topic) and one $87k wash pair are excluded. Loss-rate intervals are Wilson 95%.</li>
<li>Model: HistGradientBoosting, 5-fold stratified CV, features = price, hour (sin/cos), log ticket, whole-share flag, round-dollar flag, spread-crossing depth, topic, share count. Generated by <code>analysis/toxic.py</code> and <code>memo_toxic_web.py</code>.</li>
</ul>
</div>"""
open(OUT + "memo_toxic_artifact.html", "w").write(page); print("written", len(page))
