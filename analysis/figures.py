"""Hero figures for the memo: funnel, recovered skill distribution, label-vs-truth, calibration, 50c trap."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load, noise_constant
from skill_luck import run, GRID, SHARP, AWFUL

OUT = "/Users/advaymonga/Desktop/sif/sif-application-fall-2026/analysis/out/"
BLUE, RED, GRAY = "#2a78d6", "#e34948", "#898781"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab"]        # ordinal: awful -> sharp
SURF, INK, INK2, MUTED, GRID_C, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "sans-serif", "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID_C,
                     "grid.linewidth": 1, "axes.titlelocation": "left", "axes.titleweight": "semibold", "axes.titlesize": 12,
                     "font.size": 10, "legend.frameon": False})


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT + name, dpi=200, facecolor=SURF); plt.close(fig)


def funnel(df, k):
    d = df[df.n >= 2]
    bins = np.unique(np.geomspace(2, 5000, 22).astype(int))
    g = d.groupby(pd.cut(d.n, bins), observed=True).agg(n=("n", "median"), sd=("trader_ppv", "std"), pq=("pq", "mean"), cnt=("n", "size"))
    g = g[g.cnt >= 200]
    luck = np.sqrt(k * g.pq / g.n)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(g.n, luck, color=GRAY, lw=2, solid_capstyle="round", label="if nobody had skill (pure luck, shrinks as 1/√n)")
    ax.plot(g.n, g.sd, color=BLUE, lw=2, marker="o", ms=5, mec=SURF, mew=1.5, label="observed spread of outcomes")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of trades a wallet made"); ax.set_ylabel("spread of profit per share (sd)")
    ax.set_title("Outcomes shrink slower than luck predicts")
    ax.legend(loc="lower left"); ax.grid(True, which="major"); ax.grid(False, which="minor")
    ax.annotate(f"{g.sd.iloc[-1]:.3f}", (g.n.iloc[-1], g.sd.iloc[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK2, fontsize=9)
    ax.annotate(f"{luck.iloc[-1]:.3f}", (g.n.iloc[-1], luck.iloc[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK2, fontsize=9)
    save(fig, "fig1_funnel.png")


def skill_dist(prior_all, prior_mid):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = GRID * 100
    ax.step(x, np.cumsum(prior_all) * 100, where="post", color=GRAY, lw=2, label="all wallets")
    ax.step(x, np.cumsum(prior_mid) * 100, where="post", color=BLUE, lw=2, label="wallets betting on uncertain outcomes (avg price 10–90¢)")
    ax.set_xlim(-30, 30); ax.set_ylim(0, 100)
    for v, lab in [(AWFUL * 100, "awful | bad"), (0, "bad | good"), (SHARP * 100, "good | sharp")]:
        ax.axvline(v, color=AXIS, lw=1)
    for pr, dy in [(prior_mid, 0), (prior_all, -9)]:
        pos = pr[GRID > 0].sum() * 100
        ax.annotate(f"{pos:.0f}% have positive edge", (0.3, 100 - pos), xytext=(6, dy), textcoords="offset points", va="center", color=INK2, fontsize=9)
    ax.set_xlabel("true expected edge, cents per share"); ax.set_ylabel("% of wallets with edge below this value")
    ax.set_title("Recovered distribution of true skill (cumulative)")
    ax.legend(loc="center right"); save(fig, "fig2_skill_distribution.png")


def label_truth(post, labels, name, title):
    bins = [(-1, AWFUL, "truly awful"), (AWFUL, 0, "truly bad"), (0, SHARP, "truly good"), (SHARP, 1, "truly sharp")]
    order = ["awful", "bad", "good", "sharp"]
    M = np.array([[post[labels == l][:, (GRID > lo) & (GRID <= hi)].sum(1).mean() for lo, hi, _ in bins] for l in order]) * 100
    fig, ax = plt.subplots(figsize=(7, 3.4))
    left = np.zeros(4)
    for j, (_, _, lab) in enumerate(bins):
        ax.barh(order, M[:, j], left=left, color=RAMP[j], height=0.55, label=lab, edgecolor=SURF, linewidth=2)
        for i in range(4):
            if i == j and M[i, j] > 6:  # label the diagonal (label matches truth)
                ax.text(left[i] + M[i, j] / 2, i, f"{M[i, j]:.0f}%", ha="center", va="center", color="white" if j >= 2 else INK, fontsize=9, fontweight="semibold")
        left += M[:, j]
    ax.set_xlim(0, 100); ax.set_xlabel("where the wallet's true edge actually lies (%)"); ax.set_ylabel("dataset label")
    ax.set_title(title); ax.grid(False, axis="y"); ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.42), fontsize=9)
    save(fig, name)
    return pd.DataFrame(M, index=order, columns=[b[2] for b in bins])


def calibration():
    q = pd.read_csv(OUT + "calibration_quantile.csv"); trap = pd.read_csv(OUT + "trap50.csv")
    t50 = trap[trap.group == "p=0.50"].iloc[0]
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
    ax.text(0.74, 0.70, "honest price (win rate = price)", color=MUTED, fontsize=8, ha="center", va="bottom", rotation=45)
    ax.errorbar(q.price, q.win, yerr=[q.win - q.win_lo, q.win_hi - q.win], fmt="o", color=BLUE, ms=6, mec=SURF, mew=1.5, ecolor=BLUE, elinewidth=1.2, capsize=0, label="one-time bettors, resolved bets (95% CI)")
    ax.plot(0.5, t50.win, "o", color=RED, ms=9, mec=SURF, mew=1.5, zorder=5)
    ax.annotate(f"bets placed at exactly 50¢\nwon {t50.win*100:.0f}% (n={int(t50.n):,})", (0.5, t50.win), xytext=(0.56, 0.08), color=INK2, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    mid = q[(q.price > 0.2) & (q.price < 0.9)]
    ax.annotate("across 20–90¢ the side people buy\nwins 15 points less than the price says", (0.62, 0.50), xytext=(0.08, 0.80), color=INK2, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("price paid (implied probability)"); ax.set_ylabel("share of those bets that won")
    ax.set_title("How wrong is the crowd? Win rate vs price paid")
    ax.legend(loc="upper left"); save(fig, "fig3_calibration.png")


def trap_by_topic():
    t = pd.read_csv(OUT + "trap50.csv"); nb = t[t.group.str.startswith("neigh")].iloc[0]
    t = t[t.group.str.startswith("p=0.50 ")].copy(); t["topic"] = t.group.str.replace("p=0.50 ", "").str.replace("arts, culture, entertainment and media", "entertainment").str.replace("economy, business and finance", "economy")
    t = t.sort_values("win")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(t.topic, t.win * 100, color=BLUE, height=0.55)
    ax.errorbar(t.win * 100, t.topic, xerr=[(t.win - t.lo) * 100, (t.hi - t.win) * 100], fmt="none", ecolor=INK2, elinewidth=1.2, capsize=0)
    ax.axvline(50, color=AXIS, lw=1); ax.text(50.5, 6.2, "price = 50%", color=MUTED, fontsize=8, va="top")
    ax.axvline(nb.win * 100, color=GRAY, lw=2); ax.text(nb.win * 100 + 0.5, 6.2, f"bets at 44–56¢: {nb.win*100:.0f}%", color=INK2, fontsize=8, va="top")
    for y, (w, hi, n) in enumerate(zip(t.win, t.hi, t.n)): ax.text(hi * 100 + 1.0, y, f"{w*100:.0f}%  (n={int(n)})", va="center", color=INK2, fontsize=9)
    ax.set_xlim(0, 62); ax.set_ylim(-0.6, 6.4); ax.set_xlabel("win rate of bets placed at exactly 50¢ (%, 95% CI)"); ax.grid(False, axis="y")
    ax.set_title("The 50¢ trap holds in every topic"); save(fig, "fig4_trap50_by_topic.png")


if __name__ == "__main__":
    df = load(); k = noise_constant(df)
    funnel(df, k)
    d_all = df[df.n >= 2]; d_mid = d_all[d_all.pq >= 0.09]
    w_all, post_all = run(d_all, k, "gauss"); w_mid, post_mid = run(d_mid, k, "gauss")
    skill_dist(w_all, w_mid)
    m = label_truth(post_mid, d_mid.trader_label.values, "fig2b_label_vs_truth.png", "What the labels actually mean (uncertain-outcome bettors)")
    m.to_csv(OUT + "label_vs_truth_mid.csv"); print(m.round(1))
    calibration(); trap_by_topic()
    print("figures written")
