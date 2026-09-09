"""Build the ML memo page (v3): out-of-fold skill scores, SVG figures, published as its own artifact."""
import json, re, sys, numpy as np, pandas as pd
from common import load
sys.path.insert(0, "/Users/advaymonga/Desktop/sif/sif-application-fall-2026")
from trackrecord import evaluate, from_wallet_row, from_trades
from trackrecord.cli import card, card_extra
from trackrecord.diagnostics import extra_fronts
from trackrecord.returns import evaluate_returns, card_returns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
OUT = "/Users/advaymonga/Desktop/sif/sif-application-fall-2026/analysis/out/ml/"; ROOT = "/Users/advaymonga/Desktop/sif/sif-application-fall-2026/analysis/out/"
df = load(); F = pd.read_parquet(OUT + "features.parquet"); post = pd.read_parquet(ROOT + "skill_posterior.parquet").set_index("trader")
df["p_pos"] = post.p_positive.reindex(df.trader).values; feats = [c for c in F.columns if c not in ("trader", "int_shares_single")]; X = F[feats].values.astype(np.float32)
# out-of-fold skill scores for every n>=50 wallet: extremes get their held-out fold score, middle wallets get the fold-average
act = (df.n >= 50).values; ext = act & ((df.p_pos > 0.9) | (df.p_pos < 0.1)).values; mid = act & ~ext
y = (df.p_pos > 0.9).values; score = np.zeros(len(df)); cnt = np.zeros(len(df)); idx_ext = np.where(ext)[0]
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(idx_ext, y[idx_ext]):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=0).fit(X[idx_ext[tr]], y[idx_ext[tr]])
    score[idx_ext[te]] += m.predict_proba(X[idx_ext[te]])[:, 1]; cnt[idx_ext[te]] += 1
    score[mid] += m.predict_proba(X[mid])[:, 1]; cnt[mid] += 1
score = np.where(cnt > 0, score / np.maximum(cnt, 1), np.nan); d = df[act].assign(score=score[act])
dec = pd.qcut(d.score.rank(method="first"), 10, labels=False)
led = d.groupby(dec).apply(lambda x: pd.Series({"wallets": len(x), "c_per_dollar": 100 * x.trader_pnl.sum() / x.notional.sum(), "frac_profitable": (x.trader_pnl > 0).mean(), "notional_M": x.notional.sum() / 1e6, "pnl_M": x.trader_pnl.sum() / 1e6}))
led.to_csv(OUT + "skill_deciles_oof.csv"); print(led.round(2).to_string())
auc = json.load(open(OUT + "skill_auc.json")); attr = pd.read_csv(OUT + "skill_attribution.csv"); ct = pd.read_csv(OUT + "active_cluster_table.csv", index_col=0)
css = re.search(r"<style>.*?</style>", open(ROOT + "memo_edge_artifact.html").read(), re.S).group(0)
pct = lambda x, dd=1: f"{100*x:.{dd}f}%"

def grouped_auc():
    W, H, L, T, B = 680, 300, 56, 22, 44; groups = ["n ≥ 20", "n ≥ 50", "n ≥ 100"]; series = [("denoised skill", "bar"), ("profit sign", "ref"), ("raw label sharp vs awful", "ref2")]
    ys = lambda v: T + (H - T - B) * (1 - (v - 0.5) / 0.5); gx = lambda i: L + (W - L - 16) * (i + 0.5) / 3; bw = 46
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Cross-validated AUC by target">']
    for v in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]: o.append(f'<line x1="{L}" x2="{W-16}" y1="{ys(v):.1f}" y2="{ys(v):.1f}" class="grid"/><text x="{L-8}" y="{ys(v)+4:.1f}" class="tick" text-anchor="end">{v:.1f}</text>')
    for i, (g, nmin) in enumerate(zip(groups, [20, 50, 100])):
        for j, (nm, cls) in enumerate(series):
            key = f"n>={nmin} | " + {"denoised skill": "denoised skill", "profit sign": "profit sign", "raw label sharp vs awful": "raw label sharp-vs-awful"}[nm]; v = auc[key]["auc"]; x = gx(i) + (j - 1) * (bw + 6)
            o.append(f'<g class="mark"><title>{g}, target = {nm}: AUC {v:.3f} (n={auc[key]["n"]:,})</title><rect x="{x-bw/2:.1f}" y="{ys(v):.1f}" width="{bw}" height="{ys(0.5)-ys(v):.1f}" rx="3" class="{cls}"/><text x="{x:.1f}" y="{ys(v)-6:.1f}" class="lbl strong" text-anchor="middle">{v:.2f}</text></g>')
        o.append(f'<text x="{gx(i):.1f}" y="{H-14}" class="tick" text-anchor="middle">{g} trades</text>')
    o.append(f'<line x1="{L}" x2="{W-16}" y1="{ys(0.5)}" y2="{ys(0.5)}" class="axis"/>')
    for j, (nm, cls) in enumerate(series): o.append(f'<rect x="{L+j*215}" y="{T-14}" width="12" height="12" rx="2" class="{cls}"/><text x="{L+16+j*215}" y="{T-4}" class="tick">{nm}</text>')
    return "".join(o) + "</svg>"

def importance():
    a = attr.head(10); W, L, rowh, T = 680, 250, 24, 14; H = T + rowh * len(a) + 16; mx = a.importance.max()
    names = {"ticket_cv": "bet-size dispersion (std ÷ mean)", "avg_price": "average price traded", "std_delta": "spread of price regime", "log_n": "trade count", "burst": "trades per market-day", "time_spread_h": "spread of trading hours", "aggr": "book depth eaten per trade", "mean_delta": "distance from 50¢", "aggr_vw": "book depth eaten (volume-weighted)", "topic_entropy": "topic diversity", "log_ticket": "ticket size", "t_economy": "share in economics"}
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Feature importance">']
    for i, r in a.iterrows():
        y = T + i * rowh; w = (W - L - 200) * r.importance / mx; cls = "bar" if "more skilled" in r.direction else "loss"
        o.append(f'<g class="mark"><title>{names.get(r.feature, r.feature)}: AUC drop {r.importance:.3f}; {r.direction}</title><text x="{L-8}" y="{y+16}" class="lbl" text-anchor="end">{names.get(r.feature, r.feature)}</text><rect x="{L}" y="{y+4}" width="{w:.1f}" height="16" rx="3" class="{cls}"/><text x="{L+w+8:.1f}" y="{y+16}" class="tick">{r.importance:.3f} · {"higher = more skilled" if "more skilled" in r.direction else "higher = less skilled"}</text></g>')
    return "".join(o) + "</svg>"

def deciles():
    W, H, L, T, B = 680, 280, 56, 20, 40; v = led.c_per_dollar.values; ymin, ymax = min(v.min(), -10) * 1.15, max(v.max(), 2) * 1.4
    ys = lambda x: T + (H - T - B) * (ymax - x) / (ymax - ymin); xs = lambda i: L + (W - L - 16) * (i + 0.5) / 10; bw = 44; z = ys(0)
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Realized return by predicted-skill decile">']
    for g in [-10, -5, 0, 1]: o.append(f'<line x1="{L}" x2="{W-16}" y1="{ys(g):.1f}" y2="{ys(g):.1f}" class="{"axis" if g==0 else "grid"}"/><text x="{L-8}" y="{ys(g)+4:.1f}" class="tick" text-anchor="end">{g:+d}¢</text>')
    for i, r in led.iterrows():
        x = xs(i); top, bot = (ys(r.c_per_dollar), z) if r.c_per_dollar >= 0 else (z, ys(r.c_per_dollar))
        o.append(f'<g class="mark"><title>decile {i+1}: {r.c_per_dollar:+.2f}¢ per dollar; {pct(r.frac_profitable,0)} profitable; ${r.notional_M:.0f}M notional</title><rect x="{x-bw/2}" y="{top:.1f}" width="{bw}" height="{bot-top:.1f}" rx="3" class="{"bar" if r.c_per_dollar>=0 else "loss"}"/></g>')
        o.append(f'<text x="{x}" y="{H-14}" class="tick" text-anchor="middle">{i+1}</text>')
    o.append(f'<text x="{L}" y="{H-2}" class="tick">← predicted least skilled</text><text x="{W-16}" y="{H-2}" class="tick" text-anchor="end">predicted most skilled →</text>')
    return "".join(o) + "</svg>"

def umap_svg():
    U = np.load(OUT + "active_umap.npy"); C = pd.read_parquet(OUT + "active_clusters.parquet"); A = df[df.n >= 20].reset_index(drop=True)
    grp = {0: "machines", 11: "machines", 3: "farms", 13: "farms", 14: "farms", 8: "bust", 10: "bust"}; g = C.cluster.map(grp).fillna("retail")
    rng = np.random.default_rng(0); sub = rng.choice(len(U), 14000, replace=False)
    W, H = 680, 460; x0, x1 = np.percentile(U[:, 0], [0.5, 99.5]); y0, y1 = np.percentile(U[:, 1], [0.5, 99.5])
    cls = {"machines": "acc", "farms": "p-farm", "bust": "p-bust", "retail": "ref"}; o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="UMAP of active wallets">']
    for i in sub[np.argsort([g.iloc[i] == "retail" for i in sub])[::-1]]:
        px = 20 + (W - 40) * (U[i, 0] - x0) / (x1 - x0); py = 20 + (H - 60) * (1 - (U[i, 1] - y0) / (y1 - y0))
        if 0 < px < W and 0 < py < H - 30: o.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.7" class="{cls[g.iloc[i]]}"/>')
    for xpos, k, lab in [(24, "acc", "machines (+$20M)"), (190, "p-farm", "farms (zero-P&L makers)"), (400, "p-bust", "bust (lost everything)"), (580, "ref", "everyone else")]:
        o.append(f'<circle cx="{xpos}" cy="{H-14}" r="5" class="{k}"/><text x="{xpos+10}" y="{H-10}" class="tick">{lab}</text>')
    return "".join(o) + "</svg>"

named = {0: "machines A", 11: "machines B", 3: "farm A", 13: "farm B", 14: "farm C", 8: "bust A", 10: "bust B"}
rows = "".join(f"<tr><td>{named.get(i, f'retail {i}')}</td><td>{int(r.wallets):,}</td><td>${r.notional_M:,.0f}M</td><td>{r.pnl_M:+.1f}</td><td>{r.c_per_dollar:+.2f}</td><td>{pct(r.frac_profitable,0)}</td><td>{int(r.median_n)}</td><td>{r.avg_price:.2f}</td><td>{pct(r.maker_share,0)}</td><td>{r.ev_zero_pnl_maker_lift:.1f}×</td><td>{r.ev_bust_lift:.1f}×</td></tr>" for i, r in ct.sort_values("pnl_M", ascending=False).iterrows())
a20, a50 = auc["n>=20 | denoised skill"]["auc"], auc["n>=50 | denoised skill"]["auc"]; r50, p50 = auc["n>=50 | raw label sharp-vs-awful"]["auc"], auc["n>=50 | profit sign"]["auc"]
top = led.iloc[-1]; lab = json.load(open(OUT + "labels_summary.json"))
def rc(addr, title):
    row = df[df.trader.str.lower() == addr.lower()].iloc[0]; return f'<p class="small" style="margin:14px 0 4px"><strong>{title}</strong></p><pre class="card">{card(evaluate(from_wallet_row(row)))}</pre>'
cards = rc("0x2728d99B2405a52db60160837E130B3ba3c1A83c", "The biggest winner in the file") + rc("0x99C538dB47a2cBc0A56EbF465309d678a6f7d406", "A one-trade wallet the dataset labels \"sharp\"") + \
        rc(df[(df.n >= 50) & (df.trader_label == "sharp")].sort_values("trader_pnl").iloc[len(df[(df.n >= 50) & (df.trader_label == "sharp")]) // 2].trader, "A typical 50+-trade wallet labeled \"sharp\"") + \
        f'<p class="small" style="margin:14px 0 4px"><strong>The example CSV shipped with the tool</strong> (synthetic, 60 trades)</p><pre class="card">{card(evaluate(from_trades(pd.read_csv("/Users/advaymonga/Desktop/sif/sif-application-fall-2026/examples/sample_trades.csv"))))}</pre>'
SL = json.load(open("/Users/advaymonga/Desktop/sif/sif-application-fall-2026/deliverables/siflive_dashboard_2026-09-08.json"))
def sl(k):
    h = SL["history"][k]; eq = pd.Series(h["equity"], index=pd.to_datetime(h["timestamps"], unit="s")); r = eq.pct_change().dropna(); rep = evaluate_returns(r.values)
    return rep, f'<p class="small" style="margin:14px 0 4px"><strong>SIF Live, {k} window</strong> ({eq.index[0].date()} → {eq.index[-1].date()}, ${eq.iloc[0]:,.0f} → ${eq.iloc[-1]:,.0f})</p><pre class="card">{card_returns(rep, "day")}</pre>'
rep1y, card1y = sl("1Y"); rep3m, card3m = sl("3M"); yrs = rep1y["periods_needed_for_95pct"] / 252
P_ = pd.DataFrame(SL["positions"]); P_["mv"] = P_.market_value.astype(float); P_["upl"] = P_.unrealized_pl.astype(float); P_["px"] = P_.current_price.astype(float)
pos_n = len(P_); pos_win = int((P_.upl > 0).sum()); gross = P_.mv.abs().sum(); net = P_.mv.sum(); shorts = P_[P_.side == "short"]; cheap_shorts = int((shorts.px < 10).sum())
def srow(name, addr=None, trades_df=None):
    agg = from_wallet_row(df[df.trader.str.lower() == addr.lower()].iloc[0]) if addr else from_trades(trades_df); rp = evaluate(agg); p = rp["posterior"]
    return f"<tr><td>{name}</td><td>{agg['n']:,}</td><td>{100*agg['edge_per_share']:+.1f}¢/sh</td><td>{100*p['mean']:+.2f}¢/sh</td><td>{p['p_meaningful_positive']:.0%}</td><td>{rp['verdict'].split(':')[0]}</td></tr>"
mid_sharp = df[(df.n >= 50) & (df.trader_label == "sharp")].sort_values("trader_pnl"); mid_addr = mid_sharp.iloc[len(mid_sharp) // 2].trader
summary_rows = srow("biggest winner in the file", "0x2728d99B2405a52db60160837E130B3ba3c1A83c") + srow('one-trade wallet labeled "sharp"', "0x99C538dB47a2cBc0A56EbF465309d678a6f7d406") + srow('typical 50+-trade wallet labeled "sharp"', mid_addr) + \
               srow("example CSV (synthetic)", trades_df=pd.read_csv("/Users/advaymonga/Desktop/sif/sif-application-fall-2026/examples/sample_trades.csv")) + \
               f"<tr><td>SIF Live, 1Y (returns mode)</td><td>{rep1y['n_periods']} days</td><td>{rep1y['annualized_return']:+.1%}/yr</td><td>Sharpe {rep1y['sharpe']:.2f}</td><td>{rep1y['p_positive']:.0%}</td><td>{rep1y['verdict'].split(':')[0]}</td></tr>" + \
               f"<tr><td>SIF Live, 3M (returns mode)</td><td>{rep3m['n_periods']} days</td><td>{rep3m['annualized_return']:+.1%}/yr</td><td>Sharpe {rep3m['sharpe']:.2f}</td><td>{rep3m['p_positive']:.0%}</td><td>{rep3m['verdict'].split(':')[0]}</td></tr>"

page = f"""<title>SIF Application, Fall 2026 (v3)</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
{css}
<style>pre.card{{background:var(--card);border-radius:6px;padding:12px 14px;font:12.5px/1.5 "IBM Plex Mono",ui-monospace,Menlo,monospace;overflow-x:auto;white-space:pre-wrap;margin:0 0 6px;color:var(--ink)}} .chart .ref2{{fill:var(--muted);opacity:.55}} .chart rect.loss{{fill:var(--loss)}} .chart .p-farm{{fill:#eda100}} .chart .p-bust{{fill:var(--loss)}}</style>
<div class="wrap">
<h1>SIF Application, Fall 2026</h1>
<p class="sub">Advay Monga · An evaluator that answers "is this track record skill or luck?", built from 604,578 Polymarket wallets and pointed at the census and at SIF Live.</p>

<h2>Results: SIF Live</h2>
<p>The club's dashboard publishes its paper account's daily equity (dollar-neutral cross-sectional mean reversion; {pos_n} open positions on 8 Sep 2026, ${gross:,.0f} gross, ${net:+,.0f} net). Run through the evaluator:</p>
{card1y}{card3m}
<ul class="lede">
<li><strong>{rep1y['annualized_return']:+.1%} a year at Sharpe {rep1y['sharpe']:.2f} is a fine result that is not yet evidence.</strong> P(true mean return &gt; 0) = {rep1y['p_positive']:.0%}; the 95% band on annual return runs {rep1y['annualized_return_ci95'][0]:+.0%} to {rep1y['annualized_return_ci95'][1]:+.0%}. A strategy with zero edge produces a year like this about a third of the time. At this Sharpe, ~{yrs:.0f} years would be needed to be 95% sure.</li>
<li><strong>The positions table flatters the strategy.</strong> {pos_win} of {pos_n} open positions are in profit while the year made {rep1y['annualized_return']:+.1%}: the 5% stop-loss removes losers from the table and leaves winners. It shows survivors, not performance.</li>
<li><strong>The paper result is optimistic for live trading.</strong> {cheap_shorts} of {len(shorts)} shorts are stocks under $10 — free to borrow on paper, often hard-to-borrow or fee-bearing live — and the daily 2 pm market-order rebalance carries no slippage model. Those costs are of the same order as the return.</li>
<li><strong>A design tension worth testing:</strong> a hard −5% stop on a mean-reversion entry exits exactly when the signal says the bounce is most likely.</li>
</ul>

<h2>Results: the census</h2>
<p>The same evaluator on the dataset. The dataset's own "sharp / awful" labels are mostly luck: confident learning flags {pct(lab['flag_rate'],0)} of labels as likely wrong, {lab['flag_by_label']['sharp']*100:.0f}% of the "sharp" ones. Once luck is removed, skill is predictable from behavior alone (AUC {a50:.2f} vs {r50:.2f} on the raw label), and the strongest tell is bet-size dispersion.</p>
<div class="tw"><table><thead><tr><th>record</th><th>trades</th><th>realized edge</th><th>luck-adjusted</th><th>P(real edge)</th><th>verdict</th></tr></thead><tbody>{summary_rows}</tbody></table></div>
{cards}
{umap_svg()}
<div class="cap"><strong>Figure 1.</strong> The population, clustered without labels (autoencoder + HDBSCAN on 124,064 active wallets; 14,000 shown). Two clusters are the machines (${ct.loc[[0,11],'notional_M'].sum():,.0f}M notional, +${ct.loc[[0,11],'pnl_M'].sum():.1f}M); three are reward farms (zero-P&amp;L makers 3–6× over-represented); two are wallets that lost everything. Retail loses 0.1–7¢ per dollar. "Sharp retail" is mostly a label artifact.</div>

<h2>How the evaluator works</h2>
<p>Every record is treated as <em>true edge + noise</em>. For bets, the noise is known exactly: it shrinks with the number of trades and depends on the prices traded (variance ≈ 0.87·p(1−p)/n, calibrated on 75k resolved single bets in the census). The prior over true edge is the population's own distribution, recovered from the census. The posterior gives the luck-adjusted edge, its interval, and the trades still needed for proof. A trade list also gets five more fronts: calibration against the price, whether sizing added or cost money, how much of the P&amp;L is one lucky trade, first-half vs second-half edge, and drawdown risk at a stated bankroll fraction. A return series gets a t-test, split-half Sharpe, streaks vs chance, autocorrelation, next-year drawdown odds, and alpha/beta against a benchmark.</p>
<p class="small"><strong>How to read a card.</strong> <em>Realized edge</em> is what happened. <em>Luck-adjusted edge</em> is the posterior mean after shrinking toward the population. <em>P(edge &gt; +0.5¢)</em> is the probability the record reflects real, meaningful edge. <em>Trades for 95% proof</em> is how much more record it would take at the current pace. Percentiles compare against the {124064:,} wallets with 20+ trades.</p>

<h2>How it was built and checked on the dataset</h2>
<h3>Why the dataset's labels can't be used as-is</h3>
<p>The census labels each wallet awful / bad / good / sharp. The label is a cut on realized return, and most wallets made a handful of bets, so a wallet that bet three times and won twice is "sharp". Confident learning flags {pct(lab['flag_rate'],0)} of labels as likely wrong, {lab['flag_by_label']['sharp']*100:.0f}% of the "sharp" ones. The tool's first job is to replace that label with a probability.</p>
<h3>Removing luck: the prior the tool ships with</h3>
<p>Nonparametric empirical-Bayes deconvolution over all wallets with 2+ trades recovers the distribution of true edge in the population: 91% of wallets within ±1¢ per share of zero, thin tails either side. That distribution ships with the tool as its reference prior. Each wallet's posterior probability of positive edge becomes the denoised target for everything below.</p>
<h3>Does the denoised target mean anything? Behavior should predict it</h3>
{grouped_auc()}
<div class="cap"><strong>Figure 2.</strong> Five-fold cross-validated AUC of a gradient-boosted model on 27 behavior-only features (nothing derived from profit). Same features, same wallets, three targets.</div>
<p>Behavior predicts the denoised target at {a20:.2f} / {a50:.2f} / {auc['n>=100 | denoised skill']['auc']:.2f} for wallets with 20 / 50 / 100+ trades; the same model on the given label scores 0.68 / 0.67 / 0.66 and on raw profit 0.72 / 0.70 / 0.69. The target, not the features, was the problem.</p>
{importance()}
<div class="cap"><strong>Figure 3.</strong> What the model uses. Bet-size dispersion is {attr.importance.iloc[0]/attr.importance.iloc[1]:.1f}× more important than anything else — which is why the evaluator reports a sizing front.</div>
<h3>Checked against real money</h3>
{deciles()}
<div class="cap"><strong>Figure 4.</strong> Wallets with 50+ trades sorted into deciles by out-of-fold predicted skill; bars are realized profit per dollar. Only the top decile is positive ({top.c_per_dollar:+.2f}¢/$, {pct(top.frac_profitable,0)} profitable, ${top.notional_M:,.0f}M notional); deciles 4–8 lose 2–7¢/$.</div>

<h2>What SIF can do with it</h2>
<ul>
<li><strong>Grade any track record before trusting it</strong> — a wallet to copy, a member's account, a backtest, the live book — with one command and the same yardstick.</li>
<li><strong>Decide on evidence, not on a good quarter.</strong> The tool turns "up 6% this year" into "P(real) = {rep1y['p_positive']:.0%}, ~{yrs:.0f} years to prove." That is the number that should sit next to any decision to add capital.</li>
<li><strong>Watch sizing before P&amp;L.</strong> Bet-size dispersion is visible after a dozen trades and needs no outcomes; the sizing front flags flat, conviction-free sizing early.</li>
<li><strong>Get evidence faster than waiting.</strong> Longer backtests with borrow and slippage, or a second uncorrelated alpha, both shorten the proof time the tool reports.</li>
</ul>

<h2>Limitations</h2>
<ul>
<li>The skill target is itself an estimate (a posterior under a Gaussian noise model). The AUC measures agreement with that estimate, validated by the decile ledger against realized P&amp;L.</li>
<li>The autoencoder embedding underperforms the raw features as a predictor (linear probe 0.82 vs 0.86); its value here is the map, not the model.</li>
<li>Cross-sectional data: the model says which wallets <em>have been</em> skilled, not that they will stay so.</li>
</ul>

<p class="small">Repository: <code>sif-application-fall-2026/</code> — <code>trackrecord/</code> (tool, tests, reference prior), <code>analysis/</code> (everything above), <code>README.md</code>.</p>

<h2>Method</h2>
<ul class="small">
<li>Decoding: <code>trader_volume</code> is shares, <code>mean_tx_value</code> dollars; price = dollars ÷ shares. Time fields are milliseconds of the UTC day.</li>
<li>Deconvolution: NPMLE by EM on a 401-point grid, noise sd √(0.87·p(1−p)/n) per wallet. Confident learning: class-conditional thresholds on out-of-fold probabilities.</li>
<li>Models: HistGradientBoosting (300 iters, 31 leaves, L2 = 1), 5-fold stratified CV; permutation importance with 5 repeats; MLP denoising autoencoder (64-12-64, input noise σ = 0.3) on quantile-normalized features; UMAP (n_neighbors 40) + HDBSCAN (min cluster 3,000). Scripts: <code>analysis/ml_*.py</code>.</li>
</ul>
</div>"""
open(ROOT + "memo_ml_artifact.html", "w").write(page); print("page written", len(page))
