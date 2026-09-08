"""Memo: the exploit others run (reward farming) and the one that isn't taken (extreme-favorite mispricing). HTML -> PDF."""
import base64, json, subprocess, pandas as pd
OUT = "/Users/advaymonga/Desktop/sif/sif-application-fall-2026/analysis/out/"; CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
pct = lambda x, d=1: f"{100*x:.{d}f}%"
img = lambda f: f'<img src="data:image/png;base64,{base64.b64encode(open(OUT+f,"rb").read()).decode()}">'
def table(df, fmt=None, index=True): return df.to_html(index=index, float_format=(fmt or (lambda x: f"{x:,.2f}")), border=0, classes="t")
S = json.load(open(OUT + "edge_summary.json")); F = S["farm"]; MC = S["mc"]; C = S["ctrl_90_95"]; H = S["holders"]; CAP = S["capacity"]
t = pd.read_csv(OUT + "edge_bands.csv", index_col=0); hr = pd.read_csv(OUT + "edge_by_hour.csv", index_col=0)
tp = pd.read_csv(OUT + "edge_by_topic.csv", index_col=0); sz = pd.read_csv(OUT + "edge_by_size.csv", index_col=0)
rep = pd.read_csv(OUT + "edge_repeat_wallets.csv", index_col=0); mc = pd.read_csv(OUT + "edge_montecarlo.csv")
core = t.iloc[1:4]; core_edge = 100 * core.pnl.sum() / core.dollars.sum()
bt = t.copy(); bt.index = ["90–95¢", "95–98¢", "98–99¢", "99–99.5¢", "99.5–100¢"]
bt = pd.DataFrame({"bets": bt.n.map("{:,}".format), "dollars": bt.dollars.map("${:,.0f}".format), "implied loss": bt.implied_loss.map(pct), "realized loss": bt.realized_loss.map(lambda x: pct(x, 2)),
                   "95% CI": [f"{pct(a,2)}–{pct(b,2)}" for a, b in zip(bt.loss_lo, bt.loss_hi)], "edge ¢/$": bt.edge_c_per_dollar.map("{:+.2f}".format), "2026 fee ¢/$": bt.fee_c_per_dollar_2026.map("{:.2f}".format)})
def ctl(d, name):
    d = d.copy(); d.index.name = name
    return pd.DataFrame({"bets": d.n.astype(int).map("{:,}".format), "implied loss": d.implied_loss.map(pct), "realized loss": d.realized_loss.map(lambda x: pct(x, 2)), "edge ¢/$": d.edge_c_per_dollar.map("{:+.2f}".format)}, index=d.index)
sz.index = ["under $2", "$2–10", "$10–50", "$50–200", "$200–1,000", "over $1,000"]
rep.index = ["90–93¢", "93–96¢", "96–97.5¢", "97.5–98.5¢"]; rep2 = pd.DataFrame({"wallets": rep.wallets.astype(int).map("{:,}".format), "notional": rep.notional_M.map("${:.1f}M".format), "return ¢/$": rep.c_per_dollar.map("{:+.2f}".format), "share profitable": rep.frac_profitable.map(pct)})
mc2 = mc[mc.loss_scenario != "realized upper CI"].copy(); mc2["scenario"] = mc2.loss_scenario.map({"realized": "realized loss rate", "if price were honest": "if the price were honest"})
mc2 = pd.DataFrame({"scenario": mc2.scenario, "bankroll per bet": mc2.f.map(lambda x: f"{x:.0%}"), "median after 200 bets": mc2.median_final.map("×{:.2f}".format), "P(end below start)": mc2.p_loss.map(pct), "P(50% drawdown)": mc2.p_dd50.map(pct), "P(90% drawdown)": mc2.p_dd90.map(pct)})

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Two Exploits</title>
<style>
 body{{font-family:-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;color:#0b0b0b;max-width:820px;margin:40px auto;line-height:1.5;font-size:11.5pt}}
 h1{{font-size:24pt;margin:0 0 4px}} h2{{font-size:16pt;margin:30px 0 8px;page-break-after:avoid}} h3{{font-size:12.5pt;margin:20px 0 4px;page-break-after:avoid}}
 .sub{{color:#52514e;margin-bottom:22px}} .kpi{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}}
 .kpi div{{background:#f4f4f1;border-radius:8px;padding:12px}} .kpi b{{display:block;font-size:19pt;font-weight:600;white-space:nowrap}} .kpi span{{color:#52514e;font-size:9.5pt}}
 img{{max-width:100%;display:block;margin:12px auto;page-break-inside:avoid}} .cap{{color:#52514e;font-size:9.5pt;margin:-4px 0 16px}}
 table.t{{border-collapse:collapse;font-size:9.5pt;margin:8px 0 16px;width:100%}} table.t th,table.t td{{padding:4px 8px;text-align:right;border-bottom:1px solid #e1e0d9}}
 table.t th{{color:#52514e;font-weight:600}} table.t td:first-child,table.t th:first-child{{text-align:left}} tr{{page-break-inside:avoid}}
 ul{{margin-top:4px;padding-left:22px}} li{{margin-bottom:5px}} .small{{font-size:9.5pt;color:#52514e}} .pb{{page-break-before:always}} a{{color:#2a78d6;text-decoration:none}}
 @media print{{body{{margin:0;max-width:none;padding:0 14mm}}}}
</style></head><body>
<h1>Two Exploits in One Dataset</h1>
<div class="sub">The one 35,000 wallets are already running on Polymarket, and the one nobody has taken<br>Advay Monga · SIF application · September 2026</div>
<div class="kpi">
 <div><b>{F['maker0_wallets']:,}</b><span>wallets with the reward-farming footprint: maker-only, zero P&amp;L, ${F['maker0_notional_M']:.0f}M notional</span></div>
 <div><b>$0</b><span>what those wallets earn from trading. Their pay is off-chain.</span></div>
 <div><b>{pct(MC['realized_loss'],2)}</b><span>realized loss rate on 95–99.5¢ favorites, vs {pct(MC['implied_loss'],1)} implied by price</span></div>
 <div><b>+{MC['edge_per_bet_pct']:.1f}%</b><span>expected return per bet from buying them and holding to resolution</span></div>
</div>

<h2>Setup</h2>
<p>The file is one row per Polymarket wallet, roughly June 2024 to March 2025: lifetime profit, volume, trade count, price regime, topics, time of day. Total profit across all 604,578 rows is exactly $0, so every dollar one wallet made, another lost. I decoded the undocumented columns first (volume is in shares, so dollars ÷ shares is the price paid; for single-trade wallets 85% of profits match a resolution payoff to the cent, so I know whether each bet won). Details are in the appendix.</p>
<p>I went looking for exploits in the quant sense: a place where money is systematically flowing in one direction and someone could stand on the receiving end. I found two. One is already crowded. One is not.</p>

<h2>1. The exploit everyone is running: farming the platform</h2>
<p>A large population of wallets trades in a way that makes no sense as betting. {F['pure99_wallets']:,} wallets only ever traded at 99¢ or above and moved ${F['pure99_notional_M']:.0f}M of notional for a combined <b>${F['pure99_pnl']:,.0f}</b>. {F['exact0_wallets']:,} wallets with three or more trades have profit of exactly $0.00. {F['hedged_pairs']:,} two-trade wallets bought both sides of the same market at once. {F['q100_wallets']} wallets bought exactly 100.5 or 101 shares at 99.9¢, clearing a "$100 trade" bar for a 10¢ profit each. And {pct(F['sec_batch_share'],0)} of all single-trade wallets fired an identical bet in the <i>same second</i> as two or more other wallets.</p>
{img("fig_e3_farm.png")}
<div class="cap">Wallets that never crossed the spread, made ≥4 trades, and ended within 0.1% of zero P&amp;L. Two-thirds live above 95¢, where a resting order is nearly riskless.</div>
<p>They are not losing money and they are not making money on trades. They are making money elsewhere. Polymarket pays makers daily through its <a href="https://docs.polymarket.com/programs/liquidity-rewards">Liquidity Rewards</a> program: rest limit orders near the midpoint, get scored every minute, receive USDC at midnight UTC. Orders earn whether or not they fill. Polymarket has also <a href="https://cryptonews.com/cryptocurrency/polymarket-airdrop/">confirmed a token and an airdrop</a> (5–10% of supply, no date), with eligibility based on trading history. The {F['maker0_wallets']:,} maker-only zero-P&amp;L wallets, trading about one market a day at 3.8 trades a day, are the footprint of exactly that: manufacture the activity the reward program scores, take no market risk.</p>
<p>Is it an exploit worth copying? It pays real dollars, but it is crowded ({F['maker0_wallets']:,} wallets competing for the same daily pools, which are split by share of score), it requires capital parked at 99¢ around the clock, and Polymarket <a href="https://cryptodiffer.com/feed/project-updates/how-to-farm-the-polymarket-airdrop-a-realistic-strategy-guide">banned Sybil and wash farming</a> in March 2026 with wallet bans. The dataset can size it but cannot see the payouts. I would not build a strategy on it. What it does tell you is that a large share of the platform's reported "users" and "volume" is this.</p>

<h2>2. The exploit nobody is taking: favorites above 95¢ are underpriced</h2>
<p>For the 75,855 single-trade wallets I know the price paid and the outcome. That lets me check, for every price level, how often bets actually lost versus how often the price said they should. I removed the same-second clone batches and one $87k wash pair, and kept resolved bets at 90¢ and above.</p>
{img("fig_e1_bands.png")}
<div class="cap">Implied loss rate is 1 − price. Realized is the share of resolved bets that lost. Labels are the realized edge in cents per dollar wagered.</div>
{table(bt)}
<p>At 90–95¢ there is nothing: {pct(C['implied'])} implied, {pct(C['realized'])} realized. From 95¢ up, buyers lose a fraction of what the price implies, and the gap is worth <b>{core_edge:+.1f}¢ per dollar</b> across 95–99.5¢. Fees introduced in 2026 take almost none of it, because Polymarket's fee formula scales with p(1−p) and is nearly zero at these prices (last column).</p>

<h3>Does it hold up?</h3>
<p>The obvious objection is that these buyers already know the answer: they bought a 99¢ contract after the game ended and before the market resolved. Some of that is happening, and it is part of the edge. But the pattern is not confined to it. Realized losses beat implied losses in every time window, including hours when no US game is finishing:</p>
{table(ctl(hr, "bets placed at (95–99.5¢)"))}
<p>In every topic with enough data:</p>
{table(ctl(tp, "topic (95–99.5¢)"))}
<p>And at every ticket size, including bets over $200 where a "$1 first-trade quest" cannot be the explanation:</p>
{table(ctl(sz, "ticket size (95–99.5¢)"))}
<p>Repeat traders who live in this regime confirm it. Among wallets with 20+ trades whose average fill is 90–98.5¢ (excluding always-on bots), the ones that hold to resolution earn <b>{H['c_per_dollar']:+.2f}¢ per dollar</b> on ${H['notional_M']:.1f}M and {pct(H['frac_profitable'],0)} of them are profitable, in a market where most humans lose:</p>
{table(rep2)}

<h3>As a strategy</h3>
<p>Pooling 95–99.5¢: the price implies a {pct(MC['implied_loss'])} loss rate; the realized rate is {pct(MC['realized_loss'],2)} (upper 95% bound {pct(MC['loss_hi'],2)}). A winning bet pays {100*MC['b']:.2f}%, a losing bet pays −100%, so the expected return per bet is <b>{MC['edge_per_bet_pct']:+.2f}%</b>. Full Kelly is {pct(MC['kelly'],0)} of bankroll per bet, which is far too aggressive given how rare and total the losses are. Simulating 200 bets:</p>
{img("fig_e2_montecarlo.png")}
{table(mc2, index=False)}
<p>Losses are rare but total, so the whole question is how many you take. At 25% of bankroll per bet the median outcome is roughly a doubling over 200 bets with a 2–3% chance of being down and a 2% chance of a 50% drawdown. At 100% per bet, one loss is ruin, and about two-thirds of paths hit one. If the price were honest, the same rule loses money — the return is entirely the mispricing.</p>
<p>Capacity is not the constraint: ${CAP['notional_M']:.0f}M of notional traded in this price band in nine months across {CAP['wallets']:,} wallets. Turnover is: sports markets resolve in hours to days, politics in weeks to months, so the annualized return depends on how many cycles you can run. Two hundred cycles a year is realistic in sports alone.</p>

<h3>Why it exists and why it persists</h3>
<ul>
<li><b>Capital lockup.</b> Tying up $100 to make $1.40 is unattractive to retail and, in the fee-free era, to market makers who could earn more elsewhere. The people who do buy at 97¢ are mostly people who know something.</li>
<li><b>Lottery demand on the other side.</b> Someone has to be selling these favorites, i.e. buying the 1–5¢ longshot. Sub-10¢ longshots in this data won 19 of 4,746 resolved bets against 134 expected. That demand is structural, so the favorite stays slightly cheap.</li>
<li><b>It does not scale for a bot.</b> The edge is 1–2% per bet with a fat left tail. It suits patient capital that can hold hundreds of small positions to resolution, which is a fund-shaped opportunity, not a high-frequency one.</li>
</ul>

<h3>Caveats</h3>
<ul>
<li>Outcomes come from single-trade wallets who held to resolution; 15% of single bets were still open at the snapshot and are excluded. If unresolved bets were disproportionately the eventual losers, the realized rate is understated, though the same gap appears among repeat traders.</li>
<li>Part of the edge is informed post-event buying. A strategy that buys blindly at 97¢ will capture less than {MC['edge_per_bet_pct']:.1f}% per bet; how much less is the first thing to test with live data.</li>
<li>Capital efficiency, not capacity, is the binding constraint, and this memo cannot measure holding periods.</li>
</ul>

<h2 class="pb">Appendix</h2>
<h3>A. How the columns were decoded</h3>
<ul>
<li><b>trader_volume</b> is shares; <b>mean_tx_value</b> is dollars per trade. Price paid = dollars ÷ shares (exact for single-trade wallets; the odd modal bet sizes like $1.03 and $100.40 become $1 and $100 once converted).</li>
<li><b>mean_delta / std_delta</b> are the mean and sd of |price − 0.5|. <b>mean_time / std_time</b> are milliseconds into the UTC day.</li>
<li><b>trader_label</b> is a pure threshold on profit per share and is not used here.</li>
<li>For single-trade wallets, 85% of <b>trader_pnl</b> equals a resolution payoff (win → (1−p) per share, lose → −p), which identifies the outcome. Same-second identical bets (≥3 wallets, same price and topic) are treated as clones and excluded from edge estimates.</li>
</ul>
<h3>B. Reproducibility</h3>
<p class="small">All numbers are produced by <code>analysis/common.py</code>, <code>edge.py</code>, <code>figures_edge.py</code> and assembled by <code>memo_edge.py</code>; nothing is typed by hand. Python 3.12, pandas, numpy, matplotlib. External facts (reward program, airdrop, 2026 rules) are cited inline and used only as context; every quantitative claim comes from the dataset.</p>
</body></html>"""
open(OUT + "memo_edge.html", "w").write(html)
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={OUT}memo_edge.pdf", f"file://{OUT}memo_edge.html"], capture_output=True, timeout=120)
print("memo written")

# ---- artifact version: same content, token-based theme, no document skeleton ----
inner = html.split("<body>")[1].split("</body>")[0]
inner = inner.replace('<div class="sub">', '<p class="sub">').replace("</div>\n<div class=\"kpi\">", "</p>\n<div class=\"kpi\">", 1)
ART = """<title>Two Exploits in One Dataset</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{--bg:#f7f6f2;--card:#eeede7;--ink:#14161a;--ink2:#5a5f66;--muted:#8a8f96;--rule:#dcdad3;--accent:#2a78d6;--loss:#c9403f;--plate:#fcfcfb}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15171a;--card:#1f2226;--ink:#f1efe9;--ink2:#b7b4ab;--muted:#858a91;--rule:#2d3136;--accent:#5b9bea;--loss:#e66767;--plate:#fcfcfb}}
:root[data-theme="dark"]{--bg:#15171a;--card:#1f2226;--ink:#f1efe9;--ink2:#b7b4ab;--muted:#858a91;--rule:#2d3136;--accent:#5b9bea;--loss:#e66767;--plate:#fcfcfb}
body{background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;font-size:16px;line-height:1.55;margin:0}
.wrap{max-width:68ch;margin:0 auto;padding:56px 24px 96px}
h1{font-family:"Source Serif 4",Georgia,"Times New Roman",serif;font-weight:700;font-size:2.4rem;line-height:1.1;letter-spacing:-.01em;margin:0 0 10px;text-wrap:balance}
h2{font-family:"Source Serif 4",Georgia,serif;font-weight:600;font-size:1.55rem;line-height:1.2;margin:48px 0 12px;text-wrap:balance}
h3{font-family:"IBM Plex Sans",sans-serif;font-weight:600;font-size:1.05rem;margin:30px 0 8px;color:var(--ink)}
.sub{color:var(--ink2);margin:0 0 28px;font-size:1rem}
.kpi{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin:20px 0 8px}
@media (min-width:640px){.kpi{grid-template-columns:repeat(4,1fr)}}
.kpi div{background:var(--card);border-radius:6px;padding:14px 14px 12px}
.kpi b{display:block;font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-weight:500;font-size:1.55rem;line-height:1.1;margin-bottom:6px;color:var(--ink)}
.kpi span{color:var(--ink2);font-size:.82rem;line-height:1.35}
p{margin:0 0 14px} ul{margin:0 0 14px;padding-left:22px} li{margin-bottom:6px}
a{color:var(--accent);text-decoration:none;border-bottom:1px solid transparent} a:hover{border-bottom-color:var(--accent)}
img{display:block;width:100%;max-width:100%;margin:18px 0 8px;background:var(--plate);border-radius:6px}
.cap{color:var(--muted);font-size:.85rem;margin:0 0 22px;line-height:1.4}
.tw{overflow-x:auto;margin:10px 0 20px}
table.t{border-collapse:collapse;width:100%;font-size:.86rem;font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums}
table.t th,table.t td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--rule);white-space:nowrap}
table.t th{color:var(--ink2);font-weight:500;font-family:"IBM Plex Sans",sans-serif;font-size:.8rem;letter-spacing:.02em;text-transform:uppercase}
table.t td:first-child,table.t th:first-child{text-align:left;white-space:normal}
.small{font-size:.85rem;color:var(--ink2)} code{font-family:"IBM Plex Mono",monospace;font-size:.85em;background:var(--card);padding:1px 5px;border-radius:3px}
.pb{page-break-before:auto}
</style>
<div class="wrap">""" + inner.replace('<table border="0" class="dataframe t">', '<div class="tw"><table border="0" class="dataframe t">').replace("</table>", "</table></div>") + "</div>"
open(OUT + "memo_edge_artifact.html", "w").write(ART)
print("artifact html written")
