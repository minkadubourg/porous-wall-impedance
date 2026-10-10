"""Render the README figures from results/sweeps.npz (run scripts/compute.py first)."""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from porous_alpha import MATERIALS  # noqa: E402
from porous_alpha.model import lambda1  # noqa: E402

COLORS = {"melamine": "#2a78d6", "isorel": "#eb6834", "birch": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})

d = np.load(os.path.join(ROOT, "results", "sweeps.npz"))
om = d["omegas"]
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)


def label_end(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x[-1], y[-1]), xytext=(6, dy), textcoords="offset points",
                va="center", fontsize=9, color=INK)
    ax.plot(x[-1], y[-1], "o", ms=4, color=color)


# 1. alpha*(omega): numerical optimum vs. closed form, log-log
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
for key, mat in MATERIALS.items():
    exact = mat.eta1 * lambda1(np.array(0.0), om, mat)
    num = d[f"uniform_{key}_alpha"][::25]
    c = COLORS[key]
    nudge = {"melamine": 6, "isorel": -6}.get(key, 0)
    for ax, ex, nu, dy in ((axes[0], exact.real, num.real, 0), (axes[1], -exact.imag, -num.imag, nudge)):
        ax.plot(om, ex, color=c, label=mat.name)
        ax.plot(om[::25], nu, "o", ms=5, mfc="white", mec=c, mew=1.5)
        label_end(ax, om, ex, mat.name, c, dy)
for ax, title in ((axes[0], r"Re $\alpha^*$"), (axes[1], r"$-$Im $\alpha^*$")):
    ax.set(xscale="log", yscale="log", xlabel=r"angular frequency $\omega$ (rad/s)", title=title)
    ax.title.set_color(INK)
    ax.set_xlim(om[0], om[-1] * 2.2)
w = np.array([150, 1500])
axes[1].plot(w, 0.9 * np.sqrt(w), color=MUTED, lw=1, ls="--")
axes[1].text(w[1], 0.9 * np.sqrt(w[1]), r"  $\propto\sqrt{\omega}$", color=MUTED, va="center")
mel = MATERIALS["melamine"]
wc = mel.a / mel.xi1
for ax in axes:
    ax.axvline(wc, color=COLORS["melamine"], lw=1, ls=":")
axes[0].text(wc, 3.0, r" $\omega_c$ melamine", color=MUTED, fontsize=8)
axes[0].plot([], [], "o", ms=5, mfc="white", mec=MUTED, mew=1.5, label="numerical optimum")
axes[0].plot([], [], color=MUTED, label=r"closed form $\eta_1\lambda_1(0,\omega)$")
axes[0].legend(handles=axes[0].lines[-2:], loc="upper left", fontsize=8)
fig.savefig(os.path.join(FIG, "alpha_vs_omega.png"))
plt.close(fig)

# 2. optimiser artefact with a narrow-band source
fig, (top, bot) = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 2.4]})
top.fill_between(om, d["narrowband_envelope"], color="#cde2fb", lw=0)
top.plot(om, d["narrowband_envelope"], color="#2a78d6", lw=1.5)
top.set(ylabel="source\namplitude", yticks=[0, 1])
naive, good = d["narrowband_naive_alpha"], d["uniform_isorel_alpha"]
bot.plot(om, good.imag, color=COLORS["isorel"], label=r"normalised objective $e/\sum|g_k|^2$")
bot.plot(om, naive.imag, color=INK, lw=1.2, ls="--", label=r"raw objective $|e|$ (L-BFGS-B)")
bot.axhline(0, color=MUTED, lw=0.8)
bot.set(xlabel=r"angular frequency $\omega$ (rad/s)", ylabel=r"Im $\alpha^*$  (ISOREL)")
bot.legend(loc="lower left", fontsize=9)
bot.annotate("optimiser never leaves its\ninitial guess (1 + 1i):\nwrong sign, non-physical",
             xy=(25000, 1), xytext=(21500, 45), fontsize=8, color=INK,
             arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
fig.savefig(os.path.join(FIG, "normalisation_artefact.png"))
plt.close(fig)

# 3. residual mismatch for a spatially structured source (smoothed trend)
def rolling_median(x, n=13):
    pad = np.pad(x, n // 2, mode="edge")
    return np.array([np.median(pad[i:i + n]) for i in range(len(x))])


fig, ax = plt.subplots(figsize=(8, 3.6))
plane = np.max([d[f"uniform_{k}_err"].max() for k in MATERIALS])
for key, mat in MATERIALS.items():
    e = rolling_median(d[f"gaussian_{key}_err"])
    ax.plot(om, e, color=COLORS[key])
    label_end(ax, om, e, mat.name, COLORS[key], {"isorel": -6, "birch": 6}.get(key, 0))
ax.axhline(plane, color=MUTED, lw=1, ls="--")
ax.text(om[len(om) // 2], 2 * plane, "plane wave (all materials): exact up to round-off", color=MUTED, fontsize=8)
ax.set(yscale="log", xlim=(om[0], om[-1] * 1.18), xlabel=r"angular frequency $\omega$ (rad/s)",
       ylabel=r"residual $e(\alpha^*)\,/\,\|g\|^2$")
ax.set_title("Gaussian-beam source (rolling median over ~1000 rad/s)", fontsize=9, color=MUTED, loc="left")
fig.savefig(os.path.join(FIG, "residual_gaussian_beam.png"))
plt.close(fig)
print("figures written to", os.path.normpath(FIG))
