"""Build the ML memo page (v3): out-of-fold skill scores, SVG figures, published as its own artifact."""
import json, re, sys, math, numpy as np, pandas as pd
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

def cone_svg():
    """SIF Live 1Y equity (indexed to 100) inside the band a zero-edge strategy with the same daily volatility would produce."""
    h = SL["history"]["1Y"]; eq = pd.Series(h["equity"], index=pd.to_datetime(h["timestamps"], unit="s")); idx = 100 * eq.values / eq.values[0]
    r = eq.pct_change().dropna().values; sd = r.std(ddof=1); n = len(idx); t = np.arange(n)
    W, H, L, R, T, B = 820, 360, 54, 70, 26, 40; ymin, ymax = min(idx.min(), 100 - 2.2 * 100 * sd * math.sqrt(n)) - 1, max(idx.max(), 100 + 2.2 * 100 * sd * math.sqrt(n)) + 1
    xs = lambda i: L + (W - L - R) * i / (n - 1); ys = lambda v: T + (H - T - B) * (ymax - v) / (ymax - ymin)
    band = lambda z: (100 + z * 100 * sd * np.sqrt(t), 100 - z * 100 * sd * np.sqrt(t))
    def poly(up, lo, cls): return f'<polygon class="{cls}" points="' + " ".join(f"{xs(i):.1f},{ys(up[i]):.1f}" for i in range(n)) + " " + " ".join(f"{xs(i):.1f},{ys(lo[i]):.1f}" for i in range(n - 1, -1, -1)) + '"/>'
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="SIF Live equity inside the zero-edge band">']
    for v in range(int(ymin) // 5 * 5, int(ymax) + 5, 5):
        if ymin < v < ymax: o.append(f'<line x1="{L}" x2="{W-R}" y1="{ys(v):.1f}" y2="{ys(v):.1f}" class="grid"/><text x="{L-8}" y="{ys(v)+4:.1f}" class="tick" text-anchor="end">{v}</text>')
    u2, l2 = band(1.96); u1, l1 = band(1.0); o.append(poly(u2, l2, "cone")); o.append(poly(u1, l1, "cone2"))
    o.append(f'<line x1="{L}" x2="{W-R}" y1="{ys(100):.1f}" y2="{ys(100):.1f}" class="zero"/>')
    o.append('<polyline class="eq" points="' + " ".join(f"{xs(i):.1f},{ys(idx[i]):.1f}" for i in range(n)) + '"/>')
    o.append(f'<text x="{W-R+6}" y="{ys(idx[-1])+4:.1f}" class="lbl accent">{idx[-1]-100:+.1f}%</text>')
    o.append(f'<text x="{W-R+6}" y="{ys(u2[-1])+4:.1f}" class="lbl muted">+{u2[-1]-100:.0f}%</text><text x="{W-R+6}" y="{ys(l2[-1])+4:.1f}" class="lbl muted">{l2[-1]-100:.0f}%</text>')
    o.append(f'<text x="{xs(int(n*0.55))}" y="{ys(u2[int(n*0.55)])-8:.1f}" class="tick" text-anchor="middle">95% of zero-edge strategies with this volatility end inside the shaded band</text>')
    for k in range(0, n, 50): o.append(f'<text x="{xs(k):.1f}" y="{H-12}" class="tick" text-anchor="middle">{eq.index[k].strftime("%b %y")}</text>')
    o.append(f'<line x1="{L}" x2="{W-R}" y1="{H-B}" y2="{H-B}" class="axis"/></svg>'); return "".join(o)

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

page = f"""<title>SIF Application, Fall 2026</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=JetBrains+Mono:wght@400;500&family=Poppins:wght@300;400;500;700&display=swap">
<style>
:root{{--sif-red:#8f151d;--sif-red-dark:#651016;--ink:#151515;--muted:#747474;--line:#dedede;--soft-line:#eeeeee;--paper:#ffffff;--wash:#f7f5f2;--positive:#1a8a52;--negative:#c0392b;--gold:#b8860b;--font-num:"JetBrains Mono",monospace;--font-ui:"Inter",system-ui,sans-serif}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--sif-red:#b8202a;--sif-red-dark:#8f151d;--ink:#e8e4df;--muted:#909090;--line:#2a2a2a;--soft-line:#222222;--paper:#1e1b18;--wash:#141210;--positive:#3fb87f;--negative:#d8584f;--gold:#d4a017}}}}
:root[data-theme="dark"]{{--sif-red:#b8202a;--sif-red-dark:#8f151d;--ink:#e8e4df;--muted:#909090;--line:#2a2a2a;--soft-line:#222222;--paper:#1e1b18;--wash:#141210;--positive:#3fb87f;--negative:#d8584f;--gold:#d4a017}}
*{{box-sizing:border-box}} body{{margin:0;font-family:'Poppins',Arial,Helvetica,sans-serif;color:var(--ink);background:linear-gradient(180deg,rgba(143,21,29,.08),rgba(143,21,29,0) 190px),var(--wash);font-size:15.5px;line-height:1.55}}
.wrap{{max-width:980px;margin:0 auto;padding:34px 32px 60px}}
.topbar{{padding-bottom:18px;margin-bottom:26px;border-bottom:2px solid var(--sif-red)}}
.kicker{{color:var(--sif-red);font-size:.72rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;margin:0 0 6px}}
h1{{font-size:clamp(2rem,5vw,3.4rem);line-height:1;font-weight:500;margin:0 0 10px;letter-spacing:0}}
.sub{{color:var(--muted);font-size:.9rem;margin:0}}
h2{{font-size:1.15rem;font-weight:700;letter-spacing:.02em;margin:40px 0 12px;padding-left:12px;border-left:4px solid var(--sif-red)}}
h3{{font-size:.95rem;font-weight:500;margin:24px 0 8px}}
p{{margin:0 0 12px}} ul{{margin:0 0 12px;padding-left:22px}} li{{margin-bottom:6px}} strong{{font-weight:600}}
ul.lede{{list-style:none;padding:0;margin:6px 0 16px}} ul.lede li{{background:var(--paper);border:1px solid var(--line);border-left:4px solid var(--sif-red);padding:12px 14px;margin:0 0 10px}}
a{{color:var(--sif-red);text-decoration:none;border-bottom:1px solid transparent}} a:hover,a:focus-visible{{border-bottom-color:var(--sif-red);outline:none}}
.chart{{width:100%;height:auto;display:block;margin:14px 0 6px;background:var(--paper);border:1px solid var(--line);font-family:var(--font-ui)}}
.chart .grid{{stroke:var(--soft-line);stroke-width:1}} .chart .axis{{stroke:var(--line);stroke-width:1}} .chart .tick{{fill:var(--muted);font-size:12px}}
.chart .lbl{{fill:var(--ink);font-size:12.5px}} .chart .lbl.strong{{fill:var(--ink);font-weight:600}} .chart .lbl.accent{{fill:var(--sif-red);font-weight:600}} .chart .lbl.loss{{fill:var(--negative);font-weight:600}} .chart .lbl.muted{{fill:var(--muted)}}
.chart .ref{{fill:var(--muted)}} .chart .ref2{{fill:var(--line)}} .chart .acc{{fill:var(--sif-red);stroke:var(--paper);stroke-width:2}} .chart .ci{{stroke:var(--sif-red);stroke-width:2}} .chart .gap{{stroke:var(--muted);stroke-width:1.5;stroke-dasharray:3 3}}
.chart .bar{{fill:var(--sif-red)}} .chart rect.loss{{fill:var(--negative)}} .chart .p-farm{{fill:var(--gold)}} .chart .p-bust{{fill:var(--negative)}} .chart .mark:hover .bar,.chart .mark:hover .acc{{filter:brightness(1.15)}} .chart .mark{{cursor:default}}
.chart .cone{{fill:var(--muted);opacity:.14}} .chart .cone2{{fill:var(--muted);opacity:.10}} .chart .eq{{fill:none;stroke:var(--sif-red);stroke-width:2.2;stroke-linejoin:round}} .chart .zero{{stroke:var(--ink);stroke-width:1;stroke-dasharray:4 4;opacity:.5}}
.cap{{color:var(--muted);font-size:.8rem;margin:0 0 18px;line-height:1.45}}
.tw{{overflow-x:auto;margin:8px 0 18px;background:var(--paper);border:1px solid var(--line)}} table{{border-collapse:collapse;width:100%;font-family:var(--font-ui);font-size:.82rem;font-variant-numeric:tabular-nums}}
th,td{{padding:8px 10px;text-align:right;border-bottom:1px solid var(--soft-line);white-space:nowrap}} th{{color:var(--sif-red);font-weight:700;font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;border-bottom:1px solid var(--line)}}
td:first-child,th:first-child{{text-align:left;white-space:normal}} td{{font-family:var(--font-num)}} td:first-child{{font-family:var(--font-ui)}}
pre.card{{background:var(--paper);border:1px solid var(--line);border-left:4px solid var(--sif-red);padding:12px 14px;font:12.5px/1.5 var(--font-num);overflow-x:auto;white-space:pre-wrap;margin:0 0 8px;color:var(--ink)}}
.small{{font-size:.82rem;color:var(--muted)}} code{{font-family:var(--font-num);font-size:.85em;background:var(--paper);border:1px solid var(--soft-line);padding:1px 5px}}
@media (prefers-reduced-motion: reduce){{*{{transition:none}}}}
</style>
<div class="wrap">
<div class="topbar"><p class="kicker">Smith Investment Fund · Is a track record skill, or luck?</p><h1>SIF Application, Fall 2026</h1>
<p class="sub">Advay Monga · An evaluator built from 604,578 Polymarket wallets, pointed at the census and at SIF Live.</p></div>

{cone_svg()}
<div class="cap"><strong>SIF Live, one year of daily equity, indexed to 100.</strong> The shaded band is where a strategy with <em>zero</em> true edge and the same daily volatility would end up (68% inner, 95% outer). SIF Live finished at {100*(SL['history']['1Y']['equity'][-1]/SL['history']['1Y']['equity'][0]-1):+.1f}% — inside the band. Fine result; not yet evidence.</div>

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
<div class="cap"><strong>Figure 1.</strong> The population, grouped with no labels and no profit data: the 27 behavior features are compressed to 12 numbers per wallet by an <em>autoencoder</em> (a network trained to reproduce its own input through a narrow layer, which forces it to keep only what matters), then wallets close together in that space are grouped and the groups named afterwards by what they turned out to contain (124,064 active wallets; 14,000 shown). Two clusters are the machines (${ct.loc[[0,11],'notional_M'].sum():,.0f}M notional, +${ct.loc[[0,11],'pnl_M'].sum():.1f}M); three are reward farms (zero-P&amp;L makers 3–6× over-represented); two are wallets that lost everything. Retail loses 0.1–7¢ per dollar. "Sharp retail" is mostly a label artifact.</div>

<h2>How the evaluator works</h2>
<p>Every record is treated as <em>true edge + noise</em> — the result you see is the trader's real ability plus randomness. For bets, the size of the randomness is known exactly: it shrinks with the number of trades and depends on the prices traded (variance ≈ 0.87·p(1−p)/n, calibrated on 75k resolved single bets in the census). The <em>prior</em> — what edges are common in this population before looking at any one record — is the census's own distribution of true edge. Combining the two gives the <em>posterior</em>: the range of true edges consistent with this record, which is where the luck-adjusted edge, its interval, and the trades still needed for proof come from.</p>
<p class="small"><strong>How to read a card.</strong> <em>Realized edge</em> is what happened. <em>Luck-adjusted edge</em> is the posterior mean after shrinking toward the population. <em>P(edge &gt; +0.5¢)</em> is the probability the record reflects real, meaningful edge. <em>Trades for 95% proof</em> is how much more record it would take at the current pace. Percentiles compare against the {124064:,} wallets with 20+ trades.</p>

<h2>How it was built and checked on the dataset</h2>
<h3>Why the dataset's labels can't be used as-is</h3>
<p>The census labels each wallet awful / bad / good / sharp. The label is a cut on realized return, and most wallets made a handful of bets, so a wallet that bet three times and won twice is "sharp". <em>Confident learning</em> — checking each label against what a model trained on everyone else predicts for that wallet — flags {pct(lab['flag_rate'],0)} of labels as likely wrong, {lab['flag_by_label']['sharp']*100:.0f}% of the "sharp" ones. The tool's first job is to replace that label with a probability.</p>
<h3>Removing luck: the prior the tool ships with</h3>
<p><em>Deconvolution</em> — working backwards from the spread of observed results to the narrower spread of true abilities that produced them, once the known randomness is subtracted — over all wallets with 2+ trades recovers the distribution of true edge in the population: 91% of wallets within ±1¢ per share of zero, thin tails either side. That distribution ships with the tool as its reference prior. Each wallet's posterior probability of positive edge becomes the denoised target for everything below.</p>
<h3>Does the denoised target mean anything? Behavior should predict it</h3>
{grouped_auc()}
<div class="cap"><strong>Figure 2.</strong> Five-fold cross-validated AUC of a gradient-boosted model on 27 behavior-only features (nothing derived from profit). Same features, same wallets, three targets. <em>AUC</em> is how often the model ranks a randomly chosen skilled wallet above a randomly chosen unskilled one: 0.50 is a coin flip, 1.00 is perfect. <em>Cross-validated</em> means each wallet is scored by a model that never saw it during training.</div>
<p>Behavior predicts the denoised target at {a20:.2f} / {a50:.2f} / {auc['n>=100 | denoised skill']['auc']:.2f} for wallets with 20 / 50 / 100+ trades; the same model on the given label scores 0.68 / 0.67 / 0.66 and on raw profit 0.72 / 0.70 / 0.69. The target, not the features, was the problem.</p>
{importance()}
<div class="cap"><strong>Figure 3.</strong> What the model uses: how much AUC it loses when each feature is scrambled, so a longer bar means the model relied on that feature more. <em>Bet-size dispersion</em> is the standard deviation of a wallet's bet sizes divided by its average bet — high when it varies its stake, near zero when every bet is the same size. It is {attr.importance.iloc[0]/attr.importance.iloc[1]:.1f}× more important than anything else — which is why the evaluator reports a sizing front.</div>
<h3>Checked against real money</h3>
{deciles()}
<div class="cap"><strong>Figure 4.</strong> Wallets with 50+ trades sorted into ten equal groups by predicted skill, each wallet scored by a model that never trained on it; bars are realized profit per dollar. Only the top decile is positive ({top.c_per_dollar:+.2f}¢/$, {pct(top.frac_profitable,0)} profitable, ${top.notional_M:,.0f}M notional); deciles 4–8 lose 2–7¢/$.</div>

<h2>What the evaluator reports</h2>
<p>Given any record — a wallet address, a CSV of trades, or a series of daily returns — it returns: the realized edge, the luck-adjusted edge with a 90% interval, the probability the edge is real, the additional trades or days needed before the record would be conclusive at the current pace, and percentiles against the {124064:,} census wallets with 20+ trades. A trade list also returns calibration against the price, whether larger bets did better than smaller ones, how much of the P&amp;L rests on the single best trade, first-half versus second-half edge, and drawdown probabilities at a stated bankroll fraction. A return series returns split-half Sharpe, losing streaks against what chance would give, autocorrelation, next-year drawdown probabilities, and alpha and beta against a benchmark if one is supplied.</p>
<p>It runs from a 16 KB reference file and needs no access to the original dataset.</p>

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
