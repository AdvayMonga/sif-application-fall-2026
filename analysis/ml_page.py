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

def flow_svg():
    """How a record becomes a luck-adjusted edge."""
    steps = [("The record", "2,072 trades · +44¢ profit per dollar", "Everything one wallet, account or strategy actually did."),
             ("How much of that could be luck?", "spread of luck = 0.87 · p(1−p) / n", "Known exactly for betting data: it shrinks with the number of trades, and grows the closer the prices are to 50¢."),
             ("How much real skill exists at all?", "91% of wallets sit within ±1¢ of zero", "Recovered from all 604,578 wallets by subtracting that luck from the spread of results."),
             ("Combine the two", "range of real edges that fit this record", "A short record with a big result is pulled towards zero. A long record is left alone."),
             ("What comes out", "edge · chance it is real · record still needed", "The same three answers for a wallet, a file of trades, or a series of daily account values.")]
    W, BW, BH, GAP, L = 820, 700, 78, 26, 60
    H = 16 + len(steps) * (BH + GAP)
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="How the luck adjustment works">']
    for i, (t, mid, sub) in enumerate(steps):
        y = 8 + i * (BH + GAP)
        o.append(f'<rect x="{L}" y="{y}" width="{BW}" height="{BH}" class="fbox"/><rect x="{L}" y="{y}" width="4" height="{BH}" class="bar"/>')
        o.append(f'<text x="{L+18}" y="{y+24}" class="lbl strong">{i+1}. {t}</text>')
        o.append(f'<text x="{L+18}" y="{y+44}" class="flow-mid">{mid}</text>')
        o.append(f'<text x="{L+18}" y="{y+64}" class="tick">{sub}</text>')
        if i < len(steps) - 1:
            cx = L + BW / 2
            o.append(f'<line x1="{cx}" x2="{cx}" y1="{y+BH}" y2="{y+BH+GAP-8}" class="axis"/><path d="M{cx-5},{y+BH+GAP-12} L{cx},{y+BH+GAP-4} L{cx+5},{y+BH+GAP-12}" class="arrow"/>')
    return "".join(o) + "</svg>"

def badge(v):
    k = "gain" if v.startswith("skilled") else ("loss" if v.startswith(("losing", "negative")) else "warn")
    txt = {"gain": "REAL EDGE", "loss": "REAL NEGATIVE EDGE", "warn": "NOT PROVEN"}[k]
    return f'<span class="badge {k}">{txt}</span>'

def rowsf(pairs):
    return "".join(f'<div class="k">{k}</div><div class="v">{v}</div>' for k, v in pairs)

def html_card(title, rep, note=""):
    a, p, q = rep["inputs"], rep["posterior"], rep["percentiles_vs_active_wallets"]
    need = rep["trades_needed_for_95pct"]
    more = "not applicable, the edge is not positive" if math.isinf(need) else ("none, already conclusive" if need <= a["n"] else f"{need - a['n']:,} more trades (about {need:,} in total)")
    pairs = [("What actually happened", f"{100*a['edge_per_dollar']:+.2f}¢ profit per dollar wagered"),
             ("After removing luck", f"{100*p['mean']:+.2f}¢ per share &nbsp;<span class=\"dim\">(90% range {100*p['ci90'][0]:+.2f}¢ to {100*p['ci90'][1]:+.2f}¢)</span>"),
             ("Chance the edge is real", f"{p['p_meaningful_positive']:.0%} positive &nbsp;·&nbsp; {p['p_meaningful_negative']:.0%} negative"),
             ("Record still needed", more),
             ("Compared with 124,064 wallets", f"profit {q['edge_per_dollar']:.0f}th percentile &nbsp;·&nbsp; bet-size variation {q['sizing_dispersion']:.0f}th &nbsp;·&nbsp; number of trades {q['trades']:.0f}th"),
             ("Bet sizing", "varies with conviction, top 30% of the population" if q["sizing_dispersion"] >= 70 else ("every bet the same size, the population's coin-flipper pattern" if q["sizing_dispersion"] <= 30 else "typical variation"))]
    return f'<div class="rc"><div class="rc-h"><div><div class="rc-t">{title}</div><div class="rc-s">{a["n"]:,} trades · ${a["dollars"]:,.0f} wagered · {"profit" if a["pnl"]>=0 else "loss"} ${abs(a["pnl"]):,.0f}{note}</div></div>{badge(rep["verdict"])}</div><div class="rc-g">{rowsf(pairs)}</div></div>'

def html_card_returns(title, rep, sub):
    need = rep["periods_needed_for_95pct"]
    more = "not applicable, the average day is not positive" if math.isinf(need) else ("none, already conclusive" if need <= rep["n_periods"] else f"about {need/252:.0f} more years of trading ({need:,} days in total)")
    pairs = [("Return", f"{rep['annualized_return']:+.1%} a year &nbsp;<span class=\"dim\">(95% range {rep['annualized_return_ci95'][0]:+.0%} to {rep['annualized_return_ci95'][1]:+.0%})</span>"),
             ("Return per unit of risk", f"Sharpe {rep['sharpe']:.2f} &nbsp;·&nbsp; worst drop {rep['max_drawdown']:.1%}"),
             ("Chance the return is real", f"{rep['p_positive']:.0%}"),
             ("Record still needed", more),
             ("First half vs second half", f"Sharpe {rep['first_half_sharpe']:.2f} then {rep['second_half_sharpe']:.2f}"),
             ("Longest losing run", f"{rep['longest_losing_streak']} days in a row &nbsp;<span class=\"dim\">(about {rep['expected_longest_streak']:.0f} expected by chance)</span>"),
             ("Chance of a big drop next year", f"−20%: {rep['p_20pct_drawdown_next_year']:.0%} &nbsp;·&nbsp; −50%: {rep['p_50pct_drawdown_next_year']:.1%}")]
    return f'<div class="rc"><div class="rc-h"><div><div class="rc-t">{title}</div><div class="rc-s">{sub}</div></div>{badge(rep["verdict"])}</div><div class="rc-g">{rowsf(pairs)}</div></div>'

def cluster_bars():
    """Edge per dollar with 90% wallet-bootstrap CI, per discovered cluster."""
    t = pd.read_csv(OUT + "cluster_eval.csv").sort_values("edge_per_dollar", ascending=False)
    W, L, R, rowh, T = 820, 118, 96, 26, 30; H = T + rowh * len(t) + 18
    lo, hi = min(t.ci_lo_d.min(), -8), max(t.ci_hi_d.max(), 2); span = hi - lo
    xs = lambda v: L + (W - L - R) * (v - lo) / span
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Edge per dollar by cluster with confidence intervals">']
    for g in range(int(lo) // 2 * 2, int(hi) + 2, 2):
        if lo < g < hi: o.append(f'<line x1="{xs(g):.1f}" x2="{xs(g):.1f}" y1="{T-6}" y2="{H-14}" class="grid"/><text x="{xs(g):.1f}" y="{T-12}" class="tick" text-anchor="middle">{g:+d}¢</text>')
    z = xs(0); o.append(f'<line x1="{z:.1f}" x2="{z:.1f}" y1="{T-6}" y2="{H-14}" class="axis"/>')
    for i, r in enumerate(t.itertuples()):
        y = T + i * rowh + rowh / 2; cls = "gain" if r.edge_per_dollar >= 0 else "loss"
        x0, x1 = (z, xs(r.edge_per_dollar)) if r.edge_per_dollar >= 0 else (xs(r.edge_per_dollar), z)
        o.append(f'<g class="mark"><title>{r.name}: {r.edge_per_dollar:+.2f}¢ per dollar (90% CI {r.ci_lo_d:+.2f} to {r.ci_hi_d:+.2f}), {r.wallets:,} wallets, {r.trades:,} trades, ${r.notional_M:,.0f}M, {r.verdict}</title>')
        o.append(f'<rect x="{x0:.1f}" y="{y-7:.1f}" width="{max(x1-x0,1):.1f}" height="14" class="{cls}"/>')
        o.append(f'<line x1="{xs(r.ci_lo_d):.1f}" x2="{xs(r.ci_hi_d):.1f}" y1="{y:.1f}" y2="{y:.1f}" class="axis"/><line x1="{xs(r.ci_lo_d):.1f}" x2="{xs(r.ci_lo_d):.1f}" y1="{y-5:.1f}" y2="{y+5:.1f}" class="axis"/><line x1="{xs(r.ci_hi_d):.1f}" x2="{xs(r.ci_hi_d):.1f}" y1="{y-5:.1f}" y2="{y+5:.1f}" class="axis"/>')
        o.append(f'<text x="{L-8}" y="{y+4:.1f}" class="lbl" text-anchor="end">{r.name}</text>')
        star = "\u2713" if r.p_positive > .95 else ("\u2717" if r.p_positive < .05 else "")
        o.append(f'<text x="{W-R+8}" y="{y+4:.1f}" class="lbl {"gain" if r.p_positive > .95 else ("loss" if r.p_positive < .05 else "muted")}">{r.edge_per_dollar:+.2f}¢ {star}</text></g>')
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

FAM = pd.read_csv(OUT + "cluster_eval_family.csv")
fam_rows = "".join(f"<tr><td>{r.family}</td><td>{r.wallets:,}</td><td>${r.notional_M:,.0f}M</td><td>{r.pnl_M:+.1f}</td><td>{r.edge_per_dollar:+.2f}</td><td>{r.p_positive:.0%}</td><td>{"makes money" if r.p_positive > .95 else ("loses money" if r.p_positive < .05 else "no edge")}</td></tr>" for r in FAM.itertuples())
MACH = FAM[FAM.family == "machines"].iloc[0]; FARM = FAM[FAM.family == "farm"].iloc[0]; RET = FAM[FAM.family == "retail"].iloc[0]
CE = pd.read_csv(OUT + "cluster_eval.csv"); mach_prof = CE[CE.name.str.startswith("machines")].frac_profitable.mean()
named = {0: "machines A", 11: "machines B", 3: "farm A", 13: "farm B", 14: "farm C", 8: "bust A", 10: "bust B"}
rows = "".join(f"<tr><td>{named.get(i, f'retail {i}')}</td><td>{int(r.wallets):,}</td><td>${r.notional_M:,.0f}M</td><td>{r.pnl_M:+.1f}</td><td>{r.c_per_dollar:+.2f}</td><td>{pct(r.frac_profitable,0)}</td><td>{int(r.median_n)}</td><td>{r.avg_price:.2f}</td><td>{pct(r.maker_share,0)}</td><td>{r.ev_zero_pnl_maker_lift:.1f}×</td><td>{r.ev_bust_lift:.1f}×</td></tr>" for i, r in ct.sort_values("pnl_M", ascending=False).iterrows())
a20, a50 = auc["n>=20 | denoised skill"]["auc"], auc["n>=50 | denoised skill"]["auc"]; r50, p50 = auc["n>=50 | raw label sharp-vs-awful"]["auc"], auc["n>=50 | profit sign"]["auc"]
top = led.iloc[-1]; lab = json.load(open(OUT + "labels_summary.json"))
def rc(addr, title):
    row = df[df.trader.str.lower() == addr.lower()].iloc[0]; return html_card(title, evaluate(from_wallet_row(row)))
card_whale = rc("0x2728d99B2405a52db60160837E130B3ba3c1A83c", "The biggest winner in the dataset")
card_oneshot = rc("0x99C538dB47a2cBc0A56EbF465309d678a6f7d406", 'A wallet that made one trade, labelled "sharp" by the dataset')
mid_sharp0 = df[(df.n >= 50) & (df.trader_label == "sharp")].sort_values("trader_pnl"); mid_addr0 = mid_sharp0.iloc[len(mid_sharp0) // 2].trader
card_typical = rc(mid_addr0, 'A typical wallet with 50+ trades, also labelled "sharp"')

SL = json.load(open("/Users/advaymonga/Desktop/sif/sif-application-fall-2026/deliverables/siflive_dashboard_2026-09-08.json"))
def sl(k):
    h = SL["history"][k]; eq = pd.Series(h["equity"], index=pd.to_datetime(h["timestamps"], unit="s")); r = eq.pct_change().dropna(); rep = evaluate_returns(r.values)
    return rep, html_card_returns("SIF Live: " + ("the past year" if k == "1Y" else "the past three months"), rep, f"{eq.index[0].date()} to {eq.index[-1].date()} · ${eq.iloc[0]:,.0f} to ${eq.iloc[-1]:,.0f}")
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
.chart .bar{{fill:var(--sif-red)}} .chart rect.loss{{fill:var(--negative)}} .chart rect.gain{{fill:var(--positive)}} .chart .lbl.gain{{fill:var(--positive);font-weight:600}} .chart .p-farm{{fill:var(--gold)}} .chart .p-bust{{fill:var(--negative)}} .chart .mark:hover .bar,.chart .mark:hover .acc{{filter:brightness(1.15)}} .chart .mark{{cursor:default}}
.chart .fbox{{fill:var(--wash);stroke:var(--line);stroke-width:1}} .chart .arrow{{fill:none;stroke:var(--muted);stroke-width:1.5}} .chart .flow-mid{{fill:var(--sif-red);font-family:var(--font-num);font-size:12.5px}}
.chart .cone{{fill:var(--muted);opacity:.14}} .chart .cone2{{fill:var(--muted);opacity:.10}} .chart .eq{{fill:none;stroke:var(--sif-red);stroke-width:2.2;stroke-linejoin:round}} .chart .zero{{stroke:var(--ink);stroke-width:1;stroke-dasharray:4 4;opacity:.5}}
.cap{{color:var(--muted);font-size:.8rem;margin:0 0 18px;line-height:1.45}}
.tw{{overflow-x:auto;margin:8px 0 18px;background:var(--paper);border:1px solid var(--line)}} table{{border-collapse:collapse;width:100%;font-family:var(--font-ui);font-size:.82rem;font-variant-numeric:tabular-nums}}
th,td{{padding:8px 10px;text-align:right;border-bottom:1px solid var(--soft-line);white-space:nowrap}} th{{color:var(--sif-red);font-weight:700;font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;border-bottom:1px solid var(--line)}}
td:first-child,th:first-child{{text-align:left;white-space:normal}} td{{font-family:var(--font-num)}} td:first-child{{font-family:var(--font-ui)}}
pre.card{{background:var(--paper);border:1px solid var(--line);border-left:4px solid var(--sif-red);padding:12px 14px;font:12.5px/1.5 var(--font-num);overflow-x:auto;white-space:pre-wrap;margin:0 0 8px;color:var(--ink)}}
.rc{{background:var(--paper);border:1px solid var(--line);border-left:4px solid var(--sif-red);margin:0 0 14px}}
.rc-h{{display:flex;justify-content:space-between;align-items:flex-start;gap:14px;padding:12px 16px;border-bottom:1px solid var(--soft-line)}}
.rc-t{{font-weight:600;font-size:.98rem}} .rc-s{{color:var(--muted);font-size:.8rem;font-family:var(--font-num);margin-top:3px}}
.badge{{font-size:.62rem;font-weight:700;letter-spacing:.1em;padding:4px 9px;white-space:nowrap;border:1px solid currentColor}}
.badge.gain{{color:var(--positive)}} .badge.loss{{color:var(--negative)}} .badge.warn{{color:var(--muted)}}
.rc-g{{display:grid;grid-template-columns:minmax(150px,34%) 1fr;gap:0}} .rc-g>div{{padding:8px 16px;border-bottom:1px solid var(--soft-line)}}
.rc-g>div:nth-last-child(-n+2){{border-bottom:none}} .rc-g .k{{color:var(--muted);font-size:.8rem}} .rc-g .v{{font-family:var(--font-num);font-size:.84rem;font-variant-numeric:tabular-nums}}
.rc-g .dim{{color:var(--muted)}}
ol.steps{{padding-left:20px}} ol.steps li{{margin-bottom:9px}} .dim{{color:var(--muted)}}
.dfn{{background:var(--paper);border:1px solid var(--line);border-left:4px solid var(--muted);padding:10px 14px;margin:0 0 14px;font-size:.88rem}}
.small{{font-size:.82rem;color:var(--muted)}} code{{font-family:var(--font-num);font-size:.85em;background:var(--paper);border:1px solid var(--soft-line);padding:1px 5px}}
@media (prefers-reduced-motion: reduce){{*{{transition:none}}}}
</style>
<div class="wrap">
<div class="topbar"><p class="kicker">Smith Investment Fund · Skill or luck?</p><h1>SIF Application, Fall 2026</h1>
<p class="sub">Advay Monga · 604,578 Polymarket wallets. Telling traders with a real edge apart from traders who got lucky.</p></div>

<h2>1. The dataset, grouped by how people trade</h2>
<p>Each row is one wallet: how often it traded, how big its bets were, at what prices, at what times, and how much it made. I grouped wallets on the first four and ignored the last.</p>
{umap_svg()}
<div class="cap"><strong>Figure 1.</strong> Each dot is one of the 124,064 wallets with 20 or more trades (14,000 shown). Each wallet starts as 26 measures of how it trades, which a neural network squeezes to 12 numbers, and UMAP places on two axes. The axes have no units; only closeness means anything. No profit data was used. Names were added afterwards.</div>
<ul>
<li><strong>Machines.</strong> Two groups trading around the clock, no breaks for sleep.</li>
<li><strong>Farms.</strong> Three groups placing risk-free trades at 99¢ to collect platform rewards.</li>
<li><strong>Bust.</strong> Two groups that lost everything they deposited.</li>
<li><strong>Everyone else.</strong> About 52,000 ordinary traders.</li>
</ul>

<h2>2. Which groups make money</h2>
<p>Profit data goes back in. If wallets that trade alike also earn alike, the grouping found something real.</p>
{cluster_bars()}
<div class="cap"><strong>Figure 2.</strong> Profit per dollar wagered, by group, with the 90% range. A tick means the group makes money with at least 95% confidence, a cross that it loses money, no mark that its edge cannot be told apart from zero.</div>
<div class="tw"><table><thead><tr><th>group</th><th>wallets</th><th>traded</th><th>profit $M</th><th>edge ¢/$</th><th>P(edge&gt;0)</th><th>verdict</th></tr></thead><tbody>{fam_rows}</tbody></table></div>
<ul>
<li>Machines are the only group that makes money: <strong>{MACH.edge_per_dollar:+.2f}¢ per dollar</strong>, <strong>{MACH.pnl_M:+.1f}M</strong> profit.</li>
<li>Farms sit at zero. They are not gambling badly, they are not gambling at all. Their income is the reward programs, which are not in this data.</li>
<li>Ordinary traders lose {-RET.edge_per_dollar:.2f}¢ per dollar. The bust group loses {-FAM[FAM.family == "bust"].iloc[0].edge_per_dollar:.1f}¢.</li>
<li>Only {mach_prof:.0%} of machine wallets individually made money, yet the group's edge is certain.</li>
</ul>

<h2>3. The same method on SIF Live</h2>
<p>The club's dashboard publishes its paper account's daily value, so the same test applies.</p>
<p class="dfn"><strong>Sharpe</strong> = return ÷ how much the account value swings. Same return with a smoother ride gives a higher number. Roughly: 0.5 ordinary, 1.0 good, 2.0 rare.</p>
{card1y}
<ul>
<li><strong>{rep1y['annualized_return']:+.1%} a year is a fine result, but not proof.</strong> A strategy with no edge would do this well or better {1-rep1y['p_positive']:.0%} of the time. Proving it takes about {yrs:.0f} more years.</li>
<li><strong>The open-positions list flatters it.</strong> {pos_win} of {pos_n} positions show a profit while the year returned {rep1y['annualized_return']:+.1%}. The 5% stop closes losers, so they drop off the list.</li>
<li><strong>Paper trading hides two costs.</strong> {cheap_shorts} of {len(shorts)} shorts are stocks under $10, free to borrow on paper but often not in reality, and the daily rebalance ignores price impact. Together they are about the size of the return.</li>
<li><strong>One design choice to test.</strong> The strategy buys stocks that fell too far, then sells if they fall another 5%. The stop fires when its own logic says the bounce is closest.</li>
</ul>

<h2>4. How luck is measured</h2>
<p>We never ask why a price moved. We use how a bet pays. Buy a share at price p and you either gain 1 - p or lose p. Nothing else can happen.</p>
<ul>
<li>If the price is a fair probability, the swing in profit per share is <strong>p(1-p)</strong>. At 50¢ that is 0.25, at 97¢ it is 0.03.</li>
<li>Over n bets, luck is <strong>p(1-p) / n</strong>. More trades, less luck.</li>
<li>Checked against the data: the 75,855 one-bet wallets swung by <strong>0.87 × p(1-p)</strong>, close to theory. The tool uses 0.87.</li>
<li>Across all wallets, subtracting that luck leaves the real skill: <strong>91% sit within one cent per share of zero</strong>. So a big result over a few bets is almost certainly luck.</li>
</ul>
{flow_svg()}
<div class="cap"><strong>Figure 3.</strong> The five steps from a record to an answer.</div>
{card_whale}{card_oneshot}
<p class="small">Both look excellent on raw profit. The first traded 2,072 times, so its result is not luck. The second traded once, so it proves nothing, and the tool says so.</p>

<h2>5. Checks</h2>
<p>If the luck-adjusted number is real, how a wallet trades should predict it.</p>
{grouped_auc()}
<div class="cap"><strong>Figure 4.</strong> How well a model predicts each target from 26 measures of behaviour and no profit data. AUC is how often it ranks a skilled wallet above an unskilled one: 0.50 is a coin flip, 1.00 perfect. Each wallet is scored by a model that never saw it in training.</div>
<ul>
<li>Luck-adjusted skill: <strong>{a50:.2f}</strong>. The dataset's own "sharp / awful" label, same model, same wallets: <strong>{r50:.2f}</strong>.</li>
<li>That label is mostly luck. Checking each one against model predictions flags {pct(lab['flag_rate'],0)} as probably wrong, {lab['flag_by_label']['sharp']*100:.0f}% of the "sharp" ones.</li>
</ul>
{importance()}
<div class="cap"><strong>Figure 5.</strong> What the model relies on. A longer bar means more accuracy is lost when that measure is scrambled.</div>
<ul>
<li><strong>Bet-size variation matters most</strong>, {attr.importance.iloc[0]/attr.importance.iloc[1]:.1f} times more than anything else. Skilled wallets vary their stake. Unskilled wallets bet the same every time.</li>
<li>Wallets trading mid-range prices beat wallets buying 99¢ near-certainties.</li>
</ul>
{deciles()}
<div class="cap"><strong>Figure 6.</strong> Wallets with 50 or more trades, sorted into ten groups by predicted skill, against the money they actually made. Each wallet scored by a model that never trained on it.</div>
<ul>
<li>Only the top group makes money ({top.c_per_dollar:+.2f}¢ per dollar, {pct(top.frac_profitable,0)} in profit). Groups 4 to 8 lose 2 to 7¢.</li>
<li>The model never saw profit, and its ranking still lines up with real money. That is the check that the method works.</li>
</ul>

<h2>6. Every step, in order</h2>
<ol class="steps">
<li><strong>Decode the raw columns.</strong> The volume column is shares, not dollars, so price = dollars ÷ shares. The time columns are milliseconds into the day. For wallets with one trade, profit matches a settlement payout 85% of the time, so that bet's outcome is recoverable.</li>
<li><strong>Build 26 measures of behaviour per wallet.</strong> Trade count and frequency, bet size and how much it varies, prices traded, time of day and its spread, how much of the order book each trade eats, topic mix. Profit, profit per share and the dataset's label are excluded.</li>
<li><strong>Group the wallets.</strong> Normalise the 26 measures, compress to 12 numbers with a denoising autoencoder, reduce to 2 axes with UMAP, then find groups with HDBSCAN. 124,064 wallets with 20 or more trades. <span class="dim">(Figure 1)</span></li>
<li><strong>Score each group.</strong> Pool its wallets into one record and compute profit per dollar. The 90% range comes from resampling wallets 2,000 times, so wallets betting on the same events do not count separately. <span class="dim">(Figure 2)</span></li>
<li><strong>Measure luck.</strong> Profit per share swings by p(1-p) per bet and p(1-p)/n over a record. The multiplier 0.87 is fitted on the 75,855 one-bet wallets rather than assumed.</li>
<li><strong>Remove luck from the population.</strong> Subtracting that swing from the spread of everyone's results leaves the spread of real skill: 91% of wallets within one cent per share of zero.</li>
<li><strong>Score a single record.</strong> Combine its result, its own luck size and that population spread. Out comes the luck-adjusted edge, a 90% range, the chance the edge is real, and how much more trading would settle it. <span class="dim">(Figure 3)</span></li>
<li><strong>Check the whole thing.</strong> Train a model on the 26 behaviour measures to predict the luck-adjusted result, five-fold cross-validated: AUC {a50:.2f}, against {r50:.2f} for the dataset's own label. Then sort wallets by predicted skill and compare with profit the model never saw. <span class="dim">(Figures 4 to 6)</span></li>
</ol>

<h2>The tool</h2>
<p>All of this runs from a small Python package, <code>trackrecord</code>. Give it a wallet address, a file of trades, or daily account values, and it returns the raw result, the result after removing luck, the range that could sit in, the chance the edge is real, how much more trading would settle it, and where the record ranks against the 124,064 wallets with 20 or more trades.</p>
<div class="tw"><table><thead><tr><th>command</th><th>input</th></tr></thead><tbody>
<tr><td>trackrecord eval trades.csv</td><td>a file of trades: price, size, won</td></tr>
<tr><td>trackrecord returns equity.csv</td><td>daily account values</td></tr>
<tr><td>trackrecord wallet 0x...</td><td>any wallet in the dataset</td></tr>
</tbody></table></div>
<p>It carries the population's skill distribution in a 16 KB file, so it runs on a new record without needing the dataset. Six tests cover it.</p>
<p><strong>Code and README:</strong> <a href="https://github.com/AdvayMonga/sif-application-fall-2026">github.com/AdvayMonga/sif-application-fall-2026</a></p>

<h2>Limits</h2>
<ul>
<li>The luck-adjusted number is an estimate. Figure 6 is the check that it tracks real money.</li>
<li>One snapshot only. It shows who <em>has had</em> an edge, not who keeps it.</li>
<li>The 12-number version of each wallet used for Figure 1 predicts worse than the full 26. It is a map, not a model.</li>
</ul>

<h2>Method</h2>
<ul class="small">
<li>Decoding: <code>trader_volume</code> is shares, <code>mean_tx_value</code> dollars; price = dollars ÷ shares. Time fields are milliseconds of the UTC day.</li>
<li>Deconvolution: NPMLE by EM on a 401-point grid, noise sd √(0.87·p(1−p)/n) per wallet. Confident learning: class-conditional thresholds on out-of-fold probabilities.</li>
<li>Models: HistGradientBoosting (300 iters, 31 leaves, L2 = 1), 5-fold stratified CV; permutation importance with 5 repeats; MLP denoising autoencoder (64-12-64, input noise σ = 0.3) on quantile-normalized features; UMAP (n_neighbors 40) + HDBSCAN (min cluster 3,000). Scripts: <code>analysis/ml_*.py</code>.</li>
</ul>
</div>"""
open(ROOT + "memo_ml_artifact.html", "w").write(page); print("page written", len(page))
