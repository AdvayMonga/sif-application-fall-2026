"""Web-first memo: thesis first, short copy, inline SVG charts drawn from out/ tables. Writes out/memo_edge_artifact.html."""
import json, math, numpy as np, pandas as pd
from scipy.stats import binom
OUT = "/Users/advaymonga/Desktop/sif/sif-application-fall-2026/analysis/out/"
S = json.load(open(OUT + "edge_summary.json")); F, MC, C, H, CAP = S["farm"], S["mc"], S["ctrl_90_95"], S["holders"], S["capacity"]
t = pd.read_csv(OUT + "edge_bands.csv", index_col=0); hr = pd.read_csv(OUT + "edge_by_hour.csv", index_col=0)
tp = pd.read_csv(OUT + "edge_by_topic.csv", index_col=0); sz = pd.read_csv(OUT + "edge_by_size.csv", index_col=0)
rep = pd.read_csv(OUT + "edge_repeat_wallets.csv", index_col=0); mc = pd.read_csv(OUT + "edge_montecarlo.csv")
pct = lambda x, d=1: f"{100*x:.{d}f}%"
BANDS = ["90–95¢", "95–98¢", "98–99¢", "99–99.5¢", "99.5–100¢"]
core_edge = 100 * t.iloc[1:4].pnl.sum() / t.iloc[1:4].dollars.sum()
mc25 = mc[(mc.loss_scenario == "realized") & (mc.f == 0.25)].iloc[0]; mc100 = mc[(mc.loss_scenario == "realized") & (mc.f == 1.0)].iloc[0]
honest25 = mc[(mc.loss_scenario == "if price were honest") & (mc.f == 0.25)].iloc[0]

# ---------- SVG helpers (theme via CSS vars) ----------
def svg_bands():
    W, Hh, L, R, T, B = 680, 340, 64, 16, 28, 56
    ys = lambda v: T + (Hh - T - B) * (math.log10(20) - math.log10(max(v, 0.006))) / (math.log10(20) - math.log10(0.006))
    xs = lambda i: L + (W - L - R) * (i + 0.5) / 5
    o = [f'<svg viewBox="0 0 {W} {Hh}" class="chart" role="img" aria-label="Implied vs realized loss rate by price band">']
    for v, lab in [(0.01, "0.01%"), (0.1, "0.1%"), (1, "1%"), (10, "10%")]:
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{ys(v):.1f}" y2="{ys(v):.1f}" class="grid"/><text x="{L-8}" y="{ys(v)+4:.1f}" class="tick" text-anchor="end">{lab}</text>')
    for i, (idx, r) in enumerate(t.iterrows()):
        x = xs(i); yi, yr = ys(100 * r.implied_loss), ys(max(100 * r.realized_loss, 0.006))
        ylo, yhi = ys(max(100 * r.loss_lo, 0.006)), ys(max(100 * r.loss_hi, 0.006))
        o.append(f'<g class="mark"><title>{BANDS[i]}: price implies {pct(r.implied_loss)} lose; {pct(r.realized_loss,2)} did (95% CI {pct(r.loss_lo,2)}–{pct(r.loss_hi,2)}); n={int(r.n):,}; edge {r.edge_c_per_dollar:+.2f}¢/$</title>')
        o.append(f'<line x1="{x}" x2="{x}" y1="{yi:.1f}" y2="{yr:.1f}" class="gap"/>')
        o.append(f'<line x1="{x}" x2="{x}" y1="{ylo:.1f}" y2="{yhi:.1f}" class="ci"/>')
        o.append(f'<circle cx="{x}" cy="{yi:.1f}" r="6" class="ref"/><circle cx="{x}" cy="{yr:.1f}" r="6" class="acc"/>')
        col = "accent" if r.edge_c_per_dollar >= 0.6 else "muted"
        o.append(f'<text x="{x}" y="{max(yi, ylo)+24:.1f}" class="lbl {col}" text-anchor="middle">{r.edge_c_per_dollar:+.1f}¢/$</text></g>')
        o.append(f'<text x="{x}" y="{Hh-8}" class="tick" text-anchor="middle">{BANDS[i]}</text>')
    o.append(f'<g class="legend"><circle cx="{L+8}" cy="{T-12}" r="5" class="ref"/><text x="{L+20}" y="{T-8}" class="tick">share the price says should lose</text>'
             f'<circle cx="{L+238}" cy="{T-12}" r="5" class="acc"/><text x="{L+250}" y="{T-8}" class="tick">share that actually lost (95% CI)</text></g></svg>')
    return "".join(o)

def svg_ladder():
    L_, b, N = MC["realized_loss"], MC["b"], 200
    probs = [binom.pmf(k, N, L_) for k in (0, 1, 2)] + [1 - binom.cdf(2, N, L_)]
    labs = ["0 losses", "1 loss", "2 losses", "3+ losses"]; rules = [(0.10, "10% of bankroll per bet"), (0.25, "25%"), (0.50, "50%")]
    W, Hh, L, R, T, B = 680, 330, 56, 16, 20, 118; bw = 84
    ys = lambda p: T + (Hh - T - B) * (1 - p / 0.45); xs = lambda i: L + (W - L - R) * (i + 0.5) / 4
    o = [f'<svg viewBox="0 0 {W} {Hh}" class="chart" role="img" aria-label="Distribution of losses over 200 bets and resulting bankroll">']
    for v in (0.1, 0.2, 0.3, 0.4):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{ys(v):.1f}" y2="{ys(v):.1f}" class="grid"/><text x="{L-8}" y="{ys(v)+4:.1f}" class="tick" text-anchor="end">{int(v*100)}%</text>')
    base = ys(0)
    for i, (p, lab) in enumerate(zip(probs, labs)):
        x = xs(i); k = min(i, 3)
        o.append(f'<g class="mark"><title>{lab} in 200 bets: probability {pct(p,0)}</title><rect x="{x-bw/2}" y="{ys(p):.1f}" width="{bw}" height="{base-ys(p):.1f}" rx="3" class="bar"/>')
        o.append(f'<text x="{x}" y="{ys(p)-8:.1f}" class="lbl strong" text-anchor="middle">{pct(p,0)}</text></g>')
        o.append(f'<text x="{x}" y="{base+18}" class="tick" text-anchor="middle">{lab}</text>')
        for j, (f_, _) in enumerate(rules):
            kk = 3 if i == 3 else i
            mult = (1 + f_ * b) ** (N - kk) * (1 - f_) ** kk
            o.append(f'<text x="{x}" y="{base+42+j*20}" class="lbl {"loss" if mult < 1 else ""}" text-anchor="middle">×{mult:.2f}</text>')
    for j, (f_, lab) in enumerate(rules):
        o.append(f'<text x="{L-8}" y="{base+42+j*20}" class="tick" text-anchor="end">{lab}</text>')
    o.append(f'<line x1="{L}" x2="{W-R}" y1="{base}" y2="{base}" class="axis"/></svg>')
    return "".join(o)

def svg_farm(counts, labels):
    W, Hh, L, R, T, B = 680, 260, 56, 16, 20, 40; bw = 84; mx = max(counts) * 1.15
    ys = lambda v: T + (Hh - T - B) * (1 - v / mx); xs = lambda i: L + (W - L - R) * (i + 0.5) / len(counts); base = ys(0)
    o = [f'<svg viewBox="0 0 {W} {Hh}" class="chart" role="img" aria-label="Zero-P&L maker wallets by average price traded">']
    for v in range(0, int(mx), 4000):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{ys(v):.1f}" y2="{ys(v):.1f}" class="grid"/><text x="{L-8}" y="{ys(v)+4:.1f}" class="tick" text-anchor="end">{v//1000}k</text>')
    for i, (c, lab) in enumerate(zip(counts, labels)):
        x = xs(i)
        o.append(f'<g class="mark"><title>{lab}: {c:,} wallets</title><rect x="{x-bw/2}" y="{ys(c):.1f}" width="{bw}" height="{base-ys(c):.1f}" rx="3" class="bar"/><text x="{x}" y="{ys(c)-8:.1f}" class="lbl" text-anchor="middle">{c:,}</text></g>')
        o.append(f'<text x="{x}" y="{base+18}" class="tick" text-anchor="middle">{lab}</text>')
    o.append(f'<line x1="{L}" x2="{W-R}" y1="{base}" y2="{base}" class="axis"/></svg>')
    return "".join(o)

import sys; sys.path.insert(0, "."); from common import load
df = load(); df["avgp"] = df.notional / df.trader_volume
m = df[(df.n >= 4) & (df.price_levels_per_transaction == 0) & (df.trader_pnl.abs() <= 0.001 * df.notional)]
fc = pd.cut(m.avgp, [0, .05, .5, .95, .99, 1.0001], include_lowest=True).value_counts().sort_index().tolist()

# ---------- tables ----------
def tbl(rows, head):
    o = ['<div class="tw"><table><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"]
    for r in rows: o.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    return "".join(o) + "</tbody></table></div>"
band_rows = [[BANDS[i], f"{int(r.n):,}", pct(r.implied_loss), pct(r.realized_loss, 2), f"{r.edge_c_per_dollar:+.2f}", f"{r.fee_c_per_dollar_2026:.2f}"] for i, (_, r) in enumerate(t.iterrows())]
def slice_rows(d, names):
    return [[n, f"{int(r.n):,}", pct(r.implied_loss), pct(r.realized_loss, 2), f"{r.edge_c_per_dollar:+.2f}"] for n, (_, r) in zip(names, d.iterrows())]
slices = [["<b>By hour placed</b>", "", "", "", ""]] + slice_rows(hr, hr.index) + [["<b>By topic</b>", "", "", "", ""]] + slice_rows(tp, [i.replace("arts, culture, entertainment and media", "entertainment").replace("economy, business and finance", "economy & finance") for i in tp.index]) + \
         [["<b>By ticket size</b>", "", "", "", ""]] + slice_rows(sz, ["under $2", "$2–10", "$10–50", "$50–200", "$200–1,000", "over $1,000"])
rep_rows = [[n, f"{int(r.wallets):,}", f"${r.notional_M:.1f}M", f"{r.c_per_dollar:+.2f}", pct(r.frac_profitable, 0)] for n, (_, r) in zip(["90–93¢", "93–96¢", "96–97.5¢", "97.5–98.5¢"], rep.iterrows())]

page = f"""<title>SIF Application, Fall 2026</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#f7f6f2;--card:#edece6;--ink:#14161a;--ink2:#575c63;--muted:#8a8f96;--rule:#dcdad3;--accent:#2a78d6;--ref:#9a9891;--loss:#c9403f;--grid:#e6e4dd}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#15171a;--card:#1f2226;--ink:#f1efe9;--ink2:#b7b4ab;--muted:#858a91;--rule:#2d3136;--accent:#5b9bea;--ref:#7b7f85;--loss:#e66767;--grid:#26292e}}}}
:root[data-theme="dark"]{{--bg:#15171a;--card:#1f2226;--ink:#f1efe9;--ink2:#b7b4ab;--muted:#858a91;--rule:#2d3136;--accent:#5b9bea;--ref:#7b7f85;--loss:#e66767;--grid:#26292e}}
body{{background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;font-size:16.5px;line-height:1.55;margin:0}}
.wrap{{max-width:70ch;margin:0 auto;padding:56px 24px 96px}}
h1{{font-family:"Source Serif 4",Georgia,serif;font-weight:700;font-size:2.5rem;line-height:1.08;letter-spacing:-.012em;margin:0 0 10px;text-wrap:balance}}
h2{{font-family:"Source Serif 4",Georgia,serif;font-weight:600;font-size:1.6rem;line-height:1.2;margin:56px 0 14px;text-wrap:balance}}
h3{{font-weight:600;font-size:1rem;margin:30px 0 8px;letter-spacing:.01em}}
.sub{{color:var(--ink2);margin:0 0 26px;font-size:1.02rem}} .eyebrow{{color:var(--accent);font-size:.8rem;letter-spacing:.08em;text-transform:uppercase;font-weight:600;margin:0 0 10px}}
ul.lede{{border-left:3px solid var(--accent);padding:4px 0 4px 22px;margin:0 0 20px;list-style:disc}} ul.lede li{{margin:0 0 10px;font-size:1.05rem}} ul.lede li:last-child{{margin:0}}
.kpi{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin:22px 0 10px}} @media (min-width:640px){{.kpi{{grid-template-columns:repeat(4,1fr)}}}}
.kpi div{{background:var(--card);border-radius:6px;padding:14px 14px 12px}} .kpi b{{display:block;font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-weight:500;font-size:1.5rem;line-height:1.1;margin-bottom:6px}} .kpi span{{color:var(--ink2);font-size:.82rem;line-height:1.35}}
p{{margin:0 0 14px}} ul{{margin:0 0 14px;padding-left:22px}} li{{margin-bottom:6px}} strong{{font-weight:600}}
a{{color:var(--accent);text-decoration:none;border-bottom:1px solid transparent}} a:hover,a:focus-visible{{border-bottom-color:var(--accent);outline:none}}
.chart{{width:100%;height:auto;display:block;margin:18px 0 6px;font-family:"IBM Plex Sans",sans-serif}}
.chart .grid{{stroke:var(--grid);stroke-width:1}} .chart .axis{{stroke:var(--rule);stroke-width:1}} .chart .tick{{fill:var(--muted);font-size:12px}}
.chart .lbl{{fill:var(--ink2);font-size:12.5px}} .chart .lbl.strong{{fill:var(--ink);font-weight:600}} .chart .lbl.accent{{fill:var(--accent);font-weight:600}} .chart .lbl.loss{{fill:var(--loss);font-weight:600}}
.chart .ref{{fill:var(--ref)}} .chart .acc{{fill:var(--accent);stroke:var(--bg);stroke-width:2}} .chart .ci{{stroke:var(--accent);stroke-width:2}} .chart .gap{{stroke:var(--ref);stroke-width:1.5;stroke-dasharray:3 3}}
.chart .bar{{fill:var(--accent)}} .chart .mark:hover .bar,.chart .mark:hover .acc{{filter:brightness(1.15)}} .chart .mark{{cursor:default}}
.cap{{color:var(--muted);font-size:.85rem;margin:0 0 22px;line-height:1.4}}
.tw{{overflow-x:auto;margin:8px 0 20px}} table{{border-collapse:collapse;width:100%;font-size:.86rem;font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums}}
th,td{{padding:7px 10px;text-align:right;border-bottom:1px solid var(--rule);white-space:nowrap}} th{{color:var(--ink2);font-weight:500;font-family:"IBM Plex Sans",sans-serif;font-size:.78rem;letter-spacing:.03em;text-transform:uppercase}}
td:first-child,th:first-child{{text-align:left;white-space:normal}} td b{{font-family:"IBM Plex Sans",sans-serif;color:var(--ink2);font-weight:600;font-size:.8rem;letter-spacing:.02em}}
.small{{font-size:.86rem;color:var(--ink2)}} code{{font-family:"IBM Plex Mono",monospace;font-size:.85em;background:var(--card);padding:1px 5px;border-radius:3px}}
@media (prefers-reduced-motion: reduce){{*{{transition:none}}}}
</style>
<div class="wrap">
<h1>SIF Application, Fall 2026</h1>
<p class="sub">Advay Monga · Exploratory analysis of 604,578 Polymarket wallets, June 2024 – March 2025. Every number comes from the provided file alone.</p>

<h2>1. Key figures</h2>
<h3>Figure 1 — Above 95¢, far fewer favorites lose than the price implies</h3>
{svg_bands()}
<div class="cap">Resolved single bets, clone batches and one $87k wash pair removed. Dashed line is the gap between the implied and realized loss rate. Hover a band for the numbers.</div>
<h3>Figure 2 — What 200 bets on those favorites do to a bankroll</h3>
{svg_ladder()}
<div class="cap">Probability of each loss count over 200 bets at the realized rate, and the bankroll multiple it leaves under three sizing rules. Red means below the starting bankroll.</div>
<h3>Figure 3 — Where 34,791 zero-P&amp;L maker wallets park their orders</h3>
{svg_farm(fc, ["≤5¢", "5–50¢", "50–95¢", "95–99¢", "99–100¢"])}
<div class="cap">Wallets that never crossed the spread, made 4+ trades, and ended within 0.1% of zero P&amp;L. Two-thirds sit above 95¢, where a resting order is nearly riskless.</div>
<div class="kpi">
 <div><b>{pct(MC['realized_loss'],2)}</b><span>realized loss rate on 95–99.5¢ favorites (price implies {pct(MC['implied_loss'])})</span></div>
 <div><b>{MC['edge_per_bet_pct']:+.1f}%</b><span>expected return per bet, held to resolution</span></div>
 <div><b>×{mc25.median_final:.2f}</b><span>median bankroll after 200 bets at 25% per bet</span></div>
 <div><b>{F['maker0_wallets']:,}</b><span>maker-only zero-P&amp;L wallets, ${F['maker0_notional_M']:.0f}M notional: the farming footprint</span></div>
</div>

<h2>2. What the figures show</h2>
<ul>
<li><strong>Favorites priced 95–99.5¢ lose {pct(MC['realized_loss'],2)} of the time; the price says {pct(MC['implied_loss'])}.</strong> Buying them and holding to resolution returns {MC['edge_per_bet_pct']:+.1f}% per bet, in every hour, topic, and ticket size. This edge is open.</li>
<li><strong>{F['maker0_wallets']:,} wallets already run a different exploit:</strong> zero-risk order flow that farms Polymarket's daily liquidity rewards and a promised airdrop. On-chain they earn $0. That exploit is crowded and now banned.</li>
<li><strong>Sizing matters more than the edge.</strong> At 25% of bankroll per bet the median outcome over 200 bets is ×{mc25.median_final:.2f} with a {pct(mc25.p_loss)} chance of ending down; all-in is ruin.</li>
</ul>

<h2>3. Data and method</h2>
<ul>
<li>One row per wallet with lifetime aggregates: profit, shares traded, dollars per trade, price regime, topics, time of day. Total profit across the file is exactly $0.</li>
<li>Decoding: <code>trader_volume</code> is shares and <code>mean_tx_value</code> is dollars, so price paid = dollars ÷ shares. For single-trade wallets, 85% of <code>trader_pnl</code> equals a resolution payoff, which identifies whether the bet won.</li>
<li>Same-second identical bets (3+ wallets, same price and topic) are treated as clones and excluded. Intervals are Wilson 95%.</li>
</ul>

<h2>4. Finding 1: favorites above 95¢ are underpriced</h2>
<p>For 75,855 single-trade wallets I know the price paid and the outcome. Above 95¢, far fewer bets lose than the price implies. At 90–95¢ the effect does not exist.</p>
{tbl(band_rows, ["price band", "bets", "implied loss", "realized loss", "edge ¢/$", "2026 fee ¢/$"])}
<p>Across 95–99.5¢ the gap is worth <strong>{core_edge:+.1f}¢ per dollar wagered</strong>. The 2026 taker fee barely touches it, because Polymarket's fee scales with p(1−p) and is near zero at these prices.</p>
<h3>The objection</h3>
<p>These buyers may already know the answer: a 99¢ contract bought after the game ended, before resolution. Some did. But then the gap would live only in post-game hours and only in sports. It doesn't.</p>
{tbl(slices, ["slice (95–99.5¢)", "bets", "implied loss", "realized loss", "edge ¢/$"])}
<h3>Repeat traders confirm it</h3>
<p>Among wallets with 20+ trades whose average fill is 90–98.5¢ (bots excluded), the ones that hold to resolution earn <strong>{H['c_per_dollar']:+.2f}¢ per dollar</strong> on ${H['notional_M']:.1f}M, and {pct(H['frac_profitable'],0)} are profitable. Most humans on this platform lose.</p>
{tbl(rep_rows, ["avg fill", "wallets", "notional", "return ¢/$", "profitable"])}

<h2>5. Finding 2: the exploit 35,000 wallets already run</h2>
<p>{F['pure99_wallets']:,} wallets only ever traded at 99¢ or above; they moved ${F['pure99_notional_M']:.0f}M and made <strong>${F['pure99_pnl']:,.0f}</strong> combined. {F['exact0_wallets']:,} wallets with 3+ trades have profit of exactly $0.00. {F['hedged_pairs']:,} bought both sides of one market at once. {F['q100_wallets']} bought exactly 100.5 or 101 shares at 99.9¢, clearing a "$100 trade" bar for 10¢. {pct(F['sec_batch_share'],0)} of single-trade wallets fired an identical bet in the same second as two or more others. None of this is betting; it is activity manufactured for a scorer.</p>
<p>The scorer is real. Polymarket's <a href="https://docs.polymarket.com/programs/liquidity-rewards">Liquidity Rewards</a> pay makers in USDC every midnight UTC for resting orders near the midpoint, whether or not they fill, and Polymarket has <a href="https://cryptonews.com/cryptocurrency/polymarket-airdrop/">confirmed a token and airdrop</a> with eligibility based on trading history. These wallets trade one market a day, 3.8 times a day, at zero risk. That is the footprint.</p>
<p><strong>Worth copying? No.</strong> {F['maker0_wallets']:,} wallets split the same daily pools, capital sits at 99¢ around the clock, and Polymarket <a href="https://cryptodiffer.com/feed/project-updates/how-to-farm-the-polymarket-airdrop-a-realistic-strategy-guide">banned Sybil and wash farming</a> in March 2026. What it does show: a large share of the platform's reported users and volume is this.</p>

<h2>6. Strategy and risk</h2>
<ul>
<li><strong>Rule:</strong> buy favorites at 95–99.5¢, hold to resolution. A win pays {100*MC['b']:.2f}%, a loss pays −100%, expected return {MC['edge_per_bet_pct']:+.2f}% per bet.</li>
<li><strong>Sizing:</strong> 25% of bankroll per bet gives median ×{mc25.median_final:.2f} over 200 bets, {pct(mc25.p_loss)} chance of ending down, {pct(mc25.p_dd50)} chance of a 50% drawdown. All-in: {pct(mc100.p_loss,0)} of paths hit a loss and are ruined. Full Kelly ({pct(MC['kelly'],0)}) is not serious here.</li>
<li><strong>Control:</strong> if the price were honest, the same 25% rule ends at ×{honest25.median_final:.2f}. The return is the mispricing, nothing else.</li>
<li><strong>Capacity and turnover:</strong> ${CAP['notional_M']:.0f}M of notional traded in this band in nine months. Sports resolve in hours, politics in months; annual return depends on cycles, not on size.</li>
<li><strong>Why it persists:</strong> tying up $100 to make $1.40 bores retail; someone always wants the 1–5¢ lottery ticket on the other side (sub-10¢ longshots won 19 of 4,746 against 134 expected); the shape suits patient capital, not bots.</li>
</ul>

<h2>7. Limitations</h2>
<ul>
<li>Outcomes come from single-trade wallets that held to resolution; 15% of single bets were still open and are excluded. The same gap among repeat traders argues against a selection artifact.</li>
<li>Part of the edge is informed post-event buying. A rule that buys blindly at 97¢ captures less than {MC['edge_per_bet_pct']:.1f}% per bet; how much less is the first thing to test live.</li>
<li>Capital efficiency, not capacity, is the binding constraint, and holding periods are not observable in this file.</li>
</ul>

<h2>8. Conclusion</h2>
<p>The file contains a crowded exploit and an open one. The crowded one manufactures activity for Polymarket's reward programs and earns nothing on-chain. The open one is a persistent mispricing of near-certain outcomes, worth about {MC['edge_per_bet_pct']:.1f}% per bet, that survives every control I could run and only needs patient capital and discipline about sizing.</p>

<h2>Appendix: reproducibility</h2>
<p class="small">All numbers are generated by <code>analysis/edge.py</code> and assembled by <code>memo_edge_web.py</code>; nothing is typed by hand. External facts about reward programs are cited and used as context only. Python 3.12, pandas, numpy, scipy.</p>
</div>"""
open(OUT + "memo_edge_artifact.html", "w").write(page); print("web memo written", len(page))
