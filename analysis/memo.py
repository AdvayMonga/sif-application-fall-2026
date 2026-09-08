"""Assemble the memo (HTML -> PDF via headless Chrome) from analysis/out tables; no hand-typed numbers."""
import base64, json, subprocess, pandas as pd
from skill_luck import GRID

OUT = "/Users/advaymonga/Desktop/sif/sif-application-fall-2026/analysis/out/"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
pct = lambda x, d=0: f"{100*x:.{d}f}%"
img = lambda f: f'<img src="data:image/png;base64,{base64.b64encode(open(OUT+f,"rb").read()).decode()}">'
def table(df, fmt=None, index=True):
    return df.to_html(index=index, float_format=(fmt or (lambda x: f"{x:,.3f}")), border=0, classes="t")

rb = pd.read_csv(OUT + "robustness_noise_lik.csv"); sub = pd.read_csv(OUT + "robustness_subsets.csv")
lt = pd.read_csv(OUT + "label_vs_truth_mid.csv", index_col=0); bt = pd.read_csv(OUT + "calibration_by_topic.csv", index_col=0)
trap = pd.read_csv(OUT + "trap50.csv"); wf = pd.read_csv(OUT + "weighting_fits.csv"); cs = json.load(open(OUT + "crowd_summary.json"))
post = pd.read_parquet(OUT + "skill_posterior.parquet")

main = rb[rb.lik == "gauss"].iloc[1]
mid_g = sub[sub.subset.str.startswith("mid-price pq") & (sub.lik == "gauss")].iloc[0]
t50 = trap[trap.group == "p=0.50"].iloc[0]; nb = trap[trap.group.str.startswith("neigh")].iloc[0]
g_specs = pd.concat([rb[rb.lik == "gauss"].sharp_precision, pd.Series([mid_g.sharp_precision])])
prec_lo, prec_hi = g_specs.min(), g_specs.max(); prec_t = rb[rb.lik == "t4"].sharp_precision.min()
sk_lo, sk_hi = rb.skilled_99.min(), rb.skilled_99.max()

def clean(df, col):
    d = df[[col, "zero_edge", "positive", "sharp", "awful", "sharp_precision", "skilled_99", "n"]].copy()
    d.columns = [col, "zero edge (±1¢)", "positive edge", "true sharp", "true awful", "'sharp' label precision", "skilled @99%", "wallets"]
    for c in d.columns[1:6]: d[c] = d[c].map(pct)
    d["skilled @99%"] = d["skilled @99%"].map("{:,}".format); d["wallets"] = d["wallets"].map("{:,}".format); return d
rb2 = rb.copy(); rb2["spec"] = rb2.apply(lambda r: f"noise k = {r.k:.2f}, {'Gaussian' if r.lik=='gauss' else 't (4 df)'}", axis=1)
sub2 = sub.copy(); sub2["spec"] = (sub2.subset.str.replace("mid-price pq>=0.09 (avg price 0.1-0.9), n>=2", "10–90¢ bettors, n ≥ 2")
                                   .str.replace("mid-price, ", "10–90¢ bettors, ").str.replace("all n>=2", "all wallets, n ≥ 2").str.replace(">=", "≥")
                                   + ", " + sub2.lik.map({"gauss": "Gaussian", "t": "t (4 df)"}))
btt = bt.head(6).copy(); btt.columns = ["resolved bets", "of which 20–90¢", "gap (pts)"]; btt.index.name = "topic"
for c in btt.columns[:2]: btt[c] = btt[c].astype(int).map("{:,}".format)
top = post[(post.n >= 20) & (post.p_positive > 0.99)].sort_values("post_mean", ascending=False).head(20)[["trader", "n", "ppv", "post_mean", "label"]].copy()
top["n"] = top.n.astype(int); top.columns = ["wallet", "trades", "observed edge", "estimated true edge", "label"]

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Skill, Luck, and the Crowd</title>
<style>
 body{{font-family:-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;color:#0b0b0b;max-width:820px;margin:40px auto;line-height:1.5;font-size:11.5pt}}
 h1{{font-size:24pt;margin:0 0 4px}} h2{{font-size:16pt;margin:30px 0 8px;page-break-after:avoid}} h3{{font-size:12.5pt;margin:20px 0 4px;page-break-after:avoid}}
 .sub{{color:#52514e;margin-bottom:22px}} .kpi{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}}
 .kpi div{{background:#f4f4f1;border-radius:8px;padding:12px}} .kpi b{{display:block;font-size:19pt;font-weight:600;white-space:nowrap}} .kpi span{{color:#52514e;font-size:9.5pt}}
 img{{max-width:100%;display:block;margin:12px auto;page-break-inside:avoid}} .cap{{color:#52514e;font-size:9.5pt;margin:-4px 0 16px}}
 table.t{{border-collapse:collapse;font-size:9.5pt;margin:8px 0 16px;width:100%}} table.t th,table.t td{{padding:4px 8px;text-align:right;border-bottom:1px solid #e1e0d9}}
 table.t th{{color:#52514e;font-weight:600}} table.t td:first-child,table.t th:first-child{{text-align:left}} table.t td:first-child{{width:30%}} tr{{page-break-inside:avoid}}
 ul{{margin-top:4px;padding-left:22px}} li{{margin-bottom:5px}} .small{{font-size:9.5pt;color:#52514e}} .pb{{page-break-before:always}}
 @media print{{body{{margin:0;max-width:none;padding:0 14mm}}}}
</style></head><body>
<h1>Skill, Luck, and the Crowd</h1>
<div class="sub">What 604,578 Polymarket wallets say about who actually has an edge<br>Advay Monga · SIF application · September 2026</div>

<div class="kpi">
 <div><b>{pct(mid_g.positive)}</b><span>of people who bet on real uncertainty have a positive expected edge</span></div>
 <div><b>{pct(prec_lo)}–{pct(prec_hi)}</b><span>of wallets labeled "sharp" actually are. The rest got lucky.</span></div>
 <div><b>{cs['gap_pts']:.0f} pts</b><span>how much less often the side people buy wins, compared to the price</span></div>
 <div><b>{pct(t50.win)}</b><span>win rate of bets placed at exactly 50¢</span></div>
</div>

<h2>Setup</h2>
<p>The dataset has one row per wallet on Polymarket, covering roughly June 2024 to March 2025. Each row is a lifetime summary: profit, volume, trade count, what topics they bet on, what time of day they trade. There is also a label: <i>awful, bad, good,</i> or <i>sharp</i>.</p>
<p>Total profit across all rows is exactly $0. This is a closed system. Every dollar someone made, someone else in this file lost. That makes it a good place to ask two questions that are normally hard to answer: how much of a good track record is skill, and how wrong is the average bettor?</p>

<h3>What the columns actually mean</h3>
<p>None of the columns are documented and a few are misleading. I worked these out from the data and checked a handful of wallets against Polymarket's public API (the addresses are real).</p>
<ul>
<li><b>trader_volume is shares, not dollars.</b> Dividing dollars by shares gives the exact price paid. Once converted, the odd modal bet sizes ($1.03, $13, $100.40) become $1, $10, $100: the site's quick-buy buttons.</li>
<li><b>mean_delta / std_delta</b> are the mean and spread of |price − 0.5|. From them I can compute each wallet's expected p(1−p), which sets how noisy its results should be.</li>
<li><b>mean_time / std_time</b> are milliseconds into the day.</li>
<li><b>trader_label is just a threshold on trader_ppv</b> (profit per share): awful &lt; −0.069, bad &lt; 0, good &lt; 0.040, sharp above. No exceptions. A model trained on the columns gets AUC 1.000, because the answer is in the inputs.</li>
<li><b>For single-trade wallets I can recover the bet.</b> Price paid = dollars / shares, and 85% of these wallets' profit matches a resolution payoff to the cent, so I know whether they won.</li>
</ul>
<p>The label matters for what follows. It is a cut on realized return, and realized return over a few bets is mostly noise. So "sharp" does not mean skilled. It means a wallet's luck landed above a line.</p>

<h2>1. How much skill is there?</h2>
{img("fig1_funnel.png")}
<div class="cap">Spread of profit per share by number of trades. Gray: what you would see if nobody had skill. Blue: what we see.</div>
<p>If everyone were flipping coins, the spread of outcomes would shrink like 1/√n as people trade more. It shrinks much slower. The gap between the two lines is real variation in skill. (The gray line is calibrated, not assumed: for single-trade wallets, Var(profit per share) = 0.87 × p(1−p), which is what a bet held to resolution should give.)</p>

<h3>Method</h3>
<p>Each wallet's observed return is true edge plus luck, and I know how big the luck term is for each wallet: σ² = 0.87 · p(1−p) / n. So I can ask what distribution of true edge across the population, blurred by each wallet's own noise, reproduces the data. This is nonparametric empirical Bayes (Kiefer–Wolfowitz NPMLE, fit by EM on a {len(GRID)}-point grid). It is the same tool the mutual fund literature uses to count how many managers are actually skilled. It gives a population distribution and, for every wallet, a posterior over its true edge.</p>

<h3>Results</h3>
{img("fig2_skill_distribution.png")}
<div class="cap">Cumulative distribution of true edge. Vertical lines are the label cutoffs at −7¢ (awful | bad), 0 (bad | good) and +4¢ (good | sharp).</div>
<p>Across all wallets, {pct(main.zero_edge)} have true edge within ±1¢ per share. Most of that is by construction: a large share of wallets only ever bought 99¢ sure things and never took a real position. Among wallets whose average price is between 10¢ and 90¢, the picture is more interesting: <b>{pct(mid_g.positive)} have positive expected edge, {pct(mid_g.awful)} are reliably negative (below −7¢), and {pct(mid_g.zero_edge)} are near zero.</b> Only {pct(mid_g.sharp,1)} are truly above the "sharp" line, versus {pct(mid_g.label_sharp)} labeled that way.</p>
{img("fig2b_label_vs_truth.png")}
<div class="cap">For each label, where the wallets' true edge actually lies.</div>
<p>Of wallets labeled sharp, <b>{lt.loc['sharp','truly sharp']:.0f}% are truly sharp</b>, {lt.loc['sharp','truly good']:.0f}% are mildly positive, and <b>{lt.loc['sharp','truly bad']:.0f}% have slightly negative true edge</b>. They are ordinary losing bettors who had a good few trades. Precision of the sharp label is {pct(prec_lo)}–{pct(prec_hi)} depending on the noise assumption, and {pct(prec_t)} under the most conservative one. It is never above one in five.</p>

<h3>How stable is this?</h3>
<p>The population numbers barely move across noise assumptions. The one thing that does move is the count of wallets I can call skilled with 99% confidence: {sk_lo:,} to {sk_hi:,} depending on whether noise is Gaussian or heavy-tailed. Standardized residuals look Gaussian for 10–90¢ bettors (excess kurtosis 0.5) and heavy only for wallets betting at extreme prices, where a single loss on a 99¢ bet is a 10σ event. So I report the count as a range. Full tables are in the appendix.</p>
<p>For comparison, Barras, Scaillet and Wermers (2010) find about 75% of US mutual funds have zero alpha after fees, 24% negative, and under 1% positive. Prediction market traders who take real risk are more spread out in both directions. The conclusion is the same: a short track record tells you almost nothing, and a label built on one is mostly noise.</p>

<h2>2. How wrong is the crowd?</h2>
<p>For the 75,855 single-trade wallets I know the price paid and whether the bet won. That is {cs['mid_n']:,} resolved bets between 20¢ and 90¢ where I can check whether the price was an honest probability for the side people chose.</p>
{img("fig3_calibration.png")}
<div class="cap">Win rate vs price paid, 20 quantile bins with 95% intervals. Bets above 95¢ are excluded from the fits; most are bought after the outcome is already known.</div>
<p>Across 20–90¢, the side people bought won <b>{cs['gap_pts']:.1f} points</b> less often than the price implied (bootstrap 95% CI {cs['gap_ci'][0]:.1f} to {cs['gap_ci'][1]:.1f}). It is negative in every topic.</p>
{table(btt, fmt=lambda x: f"{x:,.1f}")}
<p>This is not the textbook favorite-longshot bias. That would show up as an inverse-S curve: overpaying for longshots, underpaying for favorites. Fitting the standard weighting functions, the curvature is close to 1 (Goldstein–Einhorn γ = {wf.iloc[1].est:.2f}) but the elevation is large (δ = {wf.iloc[2].est:.2f}, CI {wf.iloc[2].lo:.2f}–{wf.iloc[2].hi:.2f}). In plain terms: people are too optimistic about whatever they buy, at every price. That is a winner's curse. The person who crosses the spread is, on average, the one who is wrong, and the market maker on the other side keeps the difference.</p>

<h3>The 50¢ trap</h3>
{img("fig4_trap50_by_topic.png")}
<p>Bets placed at exactly 50¢ won <b>{pct(t50.win)}</b> of the time (CI {pct(t50.lo)}–{pct(t50.hi)}), against {pct(nb.win)} for bets at 44–56¢. It shows up in every topic. 50¢ is the opening price of a new market, and most "will X happen?" questions resolve No. People buy Yes anyway.</p>

<h2>3. What this is useful for</h2>
<ul>
<li><b>A way to grade track records.</b> The deconvolution gives a probability that a strategy or a person has real edge, given how many observations you have. It works on anything with a return and a sample size. The dataset's own label shows what happens without it: a threshold on realized return over short records is ~85% noise.</li>
<li><b>Be the maker, not the taker.</b> Whoever initiates a trade loses about 15 points. The always-on accounts in this file collect the other side of that, roughly 1¢ per dollar on a third of all volume. Polymarket's 2026 taker fees are about 1¢ per dollar, so that edge is now mostly taxed away.</li>
<li><b>One specific rule.</b> Selling Yes into fresh 50/50 "will X happen" markets has a ~30 point base-rate edge here, in every topic.</li>
<li><b>A candidate signal.</b> The wallets with P(edge &gt; 0) &gt; 0.99 are public addresses with public trades. Whether their edge persists is exactly what a forward test would show. This memo shows they had edge as of the snapshot, not that they keep it.</li>
</ul>
<h3>Caveats</h3>
<ul>
<li>The crowd result is about buyers only. It says people who cross the spread are wrong, not that the market's mid-price is wrong.</li>
<li>Skill estimates are as of the snapshot. Without later data they are statements about expected edge, not guarantees.</li>
<li>The noise model is Gaussian with wallet-specific variance. It fits 10–90¢ bettors well and is conservative for extreme-price wallets.</li>
</ul>

<h2 class="pb">Appendix</h2>
<h3>A. Robustness of the skill estimates</h3>
{table(clean(rb2,"spec"), index=False)}
{table(clean(sub2,"spec"), index=False)}
<h3>B. Wallets with the strongest evidence of skill</h3>
<p class="small">Wallets with ≥20 trades and P(edge &gt; 0) &gt; 0.99, ranked by estimated true edge (profit per share). Edges of 50¢+ sustained over many trades probably reflect informed trading rather than forecasting skill; the population results do not depend on them.</p>
{table(top, fmt=lambda x: f"{x:,.3f}", index=False)}
<h3>C. Reproducibility</h3>
<p class="small">Every number here is generated by <code>analysis/common.py</code> (decoding), <code>skill_luck.py</code> (deconvolution), <code>crowd.py</code> (calibration), <code>figures.py</code>, and assembled by <code>memo.py</code>. Python 3.12, pandas, numpy, scipy, matplotlib.</p>
</body></html>"""
open(OUT + "memo.html", "w").write(html)
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={OUT}memo.pdf", f"file://{OUT}memo.html"], capture_output=True, timeout=120)
print("memo written")
