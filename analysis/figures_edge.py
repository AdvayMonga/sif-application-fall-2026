"""Figures for the edge memo (dataviz palette; blue accent, gray reference, orange/aqua only in the 3-series fan)."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = "/Users/advaymonga/Desktop/sif/sif-application-fall-2026/analysis/out/"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
SURF, INK, INK2, MUTED, GRID_C, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "sans-serif", "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID_C, "grid.linewidth": 1, "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlesize": 12, "font.size": 10, "legend.frameon": False})
def save(fig, n): fig.tight_layout(); fig.savefig(OUT + n, dpi=200, facecolor=SURF); plt.close(fig)

t = pd.read_csv(OUT + "edge_bands.csv", index_col=0)
labels = ["90–95¢", "95–98¢", "98–99¢", "99–99.5¢", "99.5–100¢"]
x = np.arange(len(t))
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(x, t.implied_loss * 100, "o", color=GRAY, ms=8, mec=SURF, mew=1.5, label="loss rate the price implies")
ax.errorbar(x, t.realized_loss * 100, yerr=[(t.realized_loss - t.loss_lo) * 100, (t.loss_hi - t.realized_loss) * 100], fmt="o", color=BLUE, ms=8, mec=SURF, mew=1.5, ecolor=BLUE, elinewidth=1.5, capsize=0, label="loss rate that actually happened (95% CI)")
for i, (imp, real, e) in enumerate(zip(t.implied_loss, t.realized_loss, t.edge_c_per_dollar)):
    ax.annotate(f"{e:+.1f}¢/$", (i, max(real * 100, 0.012)), xytext=(0, -16), textcoords="offset points", ha="center", color=INK2, fontsize=9)
ax.set_yscale("log"); ax.set_ylim(0.005, 20); ax.set_xticks(x); ax.set_xticklabels(labels)
from matplotlib.ticker import FuncFormatter
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
ax.set_ylabel("share of bets that lost (%, log scale)"); ax.set_xlabel("price paid for the favorite")
ax.set_title("Above 95¢, favorites lose far less often than the price says")
ax.legend(loc="upper right"); ax.grid(False, which="minor"); save(fig, "fig_e1_bands.png")

S = json.load(open(OUT + "edge_summary.json"))["mc"]
from scipy.stats import binom
L, b, N = S["realized_loss"], S["b"], 200
ks = [0, 1, 2, 3]; probs = [binom.pmf(k, N, L) for k in ks[:3]] + [1 - binom.cdf(2, N, L)]
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.bar(["0 losses", "1 loss", "2 losses", "3+ losses"], [100 * p for p in probs], color=BLUE, width=0.55)
for i, p in enumerate(probs): ax.text(i, 100 * p + 1.2, f"{100*p:.0f}%", ha="center", color=INK2, fontsize=10, fontweight="bold")
for j, (f_, lab) in enumerate([(0.10, "10%/bet"), (0.25, "25%/bet"), (0.50, "50%/bet")]):
    for i, k in enumerate(ks):
        mult = (1 + f_ * b) ** (N - k) * (1 - f_) ** k
        ax.text(i, -7 - 6 * j, f"{lab}: ×{mult:.2f}", ha="center", color=INK2 if mult >= 1 else "#e34948", fontsize=8.5)
ax.set_ylim(-26, 45); ax.axhline(0, color=AXIS, lw=1); ax.set_yticks([0, 10, 20, 30, 40]); ax.grid(False, axis="x")
ax.set_ylabel("probability (%)"); ax.set_title("200 bets on 95–99.5¢ favorites: losses and outcomes")
ax.text(-0.45, -4, "bankroll multiple by sizing rule:", ha="left", color=MUTED, fontsize=8)
save(fig, "fig_e2_montecarlo.png")

import sys; sys.path.insert(0, "."); from common import load
df = load(); df["avgp"] = df.notional / df.trader_volume
m = df[(df.n >= 4) & (df.price_levels_per_transaction == 0) & (df.trader_pnl.abs() <= 0.001 * df.notional)]
bins = [0, .05, .5, .95, .99, 1.0001]; lab2 = ["≤5¢", "5–50¢", "50–95¢", "95–99¢", "99–100¢"]
c = pd.cut(m.avgp, bins, include_lowest=True).value_counts().sort_index()
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.bar(lab2, c.values, color=BLUE, width=0.55)
for i, v in enumerate(c.values): ax.text(i, v + 300, f"{v:,}", ha="center", color=INK2, fontsize=9)
ax.set_ylabel("wallets"); ax.set_xlabel("average price the wallet traded at"); ax.grid(False, axis="x")
ax.set_title(f"{len(m):,} zero-P&L maker wallets: price they park at"); save(fig, "fig_e3_farm.png")
print("figures written")
