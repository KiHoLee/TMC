"""Regenerate all paper figures from the CSVs in ../data with
publication-quality layout (no legend/curve overlap, consistent styling,
conventional-scheme baselines included).

This is the canonical figure generator; experiment scripts write the CSVs.
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "fig")
DATA = os.path.join(HERE, "..", "data")

plt.rcParams.update({
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.15,
    "lines.markersize": 3.2,
})


FIGW, FIGH = 2.9, 2.25
AXRECT = [0.185, 0.18, 0.77, 0.7444]   # exact 8:6 axes box, identical everywhere


def new_fig():
    """Canvas and axes rectangle identical for every figure, so every plot
    box renders at exactly the same size in the paper."""
    fig = plt.figure(figsize=(FIGW, FIGH))
    ax = fig.add_axes(AXRECT)
    return fig, ax


def load(name):
    with open(os.path.join(DATA, name)) as f:
        return list(csv.DictReader(f))


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name))
    print("saved", name)


S = {"OMA": ("0.45", ":", "v"), "NOMA": ("tab:brown", ":", "P"),
     "SR": ("tab:red", "--", "s"), "SC": ("tab:green", "-.", "^"),
     "LMMSE": ("tab:blue", "-", "o"), "DR": ("k", "-", "d")}
LBL = {"OMA": "OMA", "NOMA": "NOMA-SIC", "SR": "SR",
       "SC": "SC", "LMMSE": "LMMSE", "DR": "Proposed DR"}
ORDER = ["OMA", "NOMA", "SR", "SC", "LMMSE", "DR"]

# ---------------------------------------------------------------- E1 beta
rows = load("e1_beta.csv")
betas = [float(r["beta"]) for r in rows]

fig, ax = new_fig()
for n in ORDER:
    c, ls, mk = S[n]
    ax.semilogy(betas, [float(r[f"{n}_nmse"]) for r in rows], ls, color=c,
                marker=mk, label=LBL[n])
ax.semilogy(betas, [float(r["LMMSE_cf"]) for r in rows], 'x',
            color="tab:blue", ms=6.5, mew=1.5, ls="none",
            label="LMMSE closed form")
ax.semilogy(betas, [float(r["SR_cf"]) for r in rows], '+', color="tab:red",
            ms=7.5, mew=1.5, ls="none", label="SR closed form")
ax.set_xlabel(r"affinity $\beta$")
ax.set_ylabel("NMSE")
ax.set_ylim(6e-2, 8e6)
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper left", ncol=2, columnspacing=0.7, handletextpad=0.4,
          labelspacing=0.3)
savefig(fig, "fig_e1_beta_nmse.pdf")

fig, ax = new_fig()
for n in ORDER:
    c, ls, mk = S[n]
    ax.plot(betas, [float(r[f"{n}_cos"]) for r in rows], ls, color=c,
            marker=mk, label=LBL[n])
ax.set_xlabel(r"affinity $\beta$")
ax.set_ylabel("mean cosine recovery")
ax.set_ylim(0.0, 1.05)
ax.grid(alpha=0.3)
ax.legend(loc="upper left", ncol=2, columnspacing=0.7, handletextpad=0.4,
          labelspacing=0.3)
savefig(fig, "fig_e1_beta_cos.pdf")

# ---------------------------------------------------------------- E1 snr
rows = load("e1_snr.csv")
snrs = [float(r["snr"]) for r in rows]
fig, ax = new_fig()
for n in ORDER:
    c, ls, mk = S[n]
    ax.semilogy(snrs, [max(float(r[f"{n}_ser"]), 1e-4) for r in rows], ls,
                color=c, marker=mk, label=LBL[n])
ax.set_xlabel("per-user SNR (dB)")
ax.set_ylabel("semantic error rate")
ax.set_ylim(8e-4, 2.5)
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="lower left", ncol=1, fontsize=6.1, handletextpad=0.4,
          labelspacing=0.25, borderpad=0.3)
savefig(fig, "fig_e1_snr.pdf")

# ---------------------------------------------- E2 spectrum (multi-curve)
rows_s = load("e2_spectrum_multi.csv")
fig, ax = new_fig()
for n_cal, c, ls in ((100, "tab:orange", "-."), (400, "tab:green", "--"),
                     (1600, "tab:blue", "-")):
    pts = [(int(r["idx"]), float(r["eig"])) for r in rows_s
           if int(r["N"]) == n_cal]
    ax.semilogy([p[0] for p in pts],
                np.maximum([p[1] for p in pts], 1e-12), ls, color=c,
                lw=1.15, label=f"$N{{=}}{n_cal}$")
ax.axvline(128, color="k", ls=":", lw=0.9)
ax.annotate(r"$d_c=128$", xy=(128, 1e-6), xytext=(150, 3e-7), fontsize=7.5,
            arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"))
ax.set_xlabel("eigenvalue index")
ax.set_ylabel(r"eigenvalue of $\hat{\mathbf{\Sigma}}$")
ax.set_ylim(1e-8, 3e-1)
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper right", labelspacing=0.3)
savefig(fig, "fig_e2_spectrum.pdf")

# ---------------------------------------------- E2 subspace (multi-curve)
rows_n = load("e2_subspace_multi.csv")
fig, ax = new_fig()
for Um, c, mk, ls in ((2, "tab:orange", "s", "-."),
                      (4, "tab:blue", "o", "-"),
                      (8, "tab:green", "^", "--")):
    pts = [(int(r["N"]), float(r["err"])) for r in rows_n
           if int(r["U"]) == Um]
    ax.loglog([p[0] for p in pts], [p[1] for p in pts], ls, color=c,
              marker=mk, label=f"$U{{=}}{Um}$")
ax.set_xlabel("paired calibration samples $N$")
ax.set_ylabel("subspace recovery error")
# plain tick labels at the measured N values; suppress the crowded
# mantissa minor labels of the default log locator. The y axis uses
# short decimal labels so that the axis title fits inside the fixed
# canvas (wide mantissa labels pushed it off the left edge).
from matplotlib.ticker import NullLocator
Ns_ticks = [200, 400, 800, 1600, 3200, 6400]
ax.set_xticks(Ns_ticks)
ax.set_xticklabels([str(n) for n in Ns_ticks])
ax.xaxis.set_minor_locator(NullLocator())
y_ticks = [0.1, 0.2, 0.3, 0.4, 0.6]
ax.set_yticks(y_ticks)
ax.set_yticklabels([f"{v:g}" for v in y_ticks])
ax.yaxis.set_minor_locator(NullLocator())
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper right", labelspacing=0.3)
savefig(fig, "fig_e2_subspace.pdf")

# ---------------------------------------------------------------- E2 ladder
rows = load("e2_ladder.csv")
snrs = [float(r["snr"]) for r in rows]
S2 = {"OMA": ("0.45", ":", "v", "OMA"),
      "NOMA": ("tab:brown", ":", "P", "NOMA-SIC"),
      "LMMSE": ("tab:blue", "-", "o", "LMMSE"),
      "DR-spec": ("tab:orange", "--", "s", "DR + spectral"),
      "DR-adapt": ("tab:purple", "-", "^", "DR + adapter"),
      "DR-oracle": ("k", ":", "d", "DR + oracle basis")}
fig, ax = new_fig()
for k, (c, ls, mk, lb) in S2.items():
    ax.semilogy(snrs, [max(float(r[f"{k}_ser"]), 1e-4) for r in rows], ls,
                color=c, marker=mk, label=lb)
ax.set_xlabel("per-user SNR (dB)")
ax.set_ylabel("semantic error rate")
ax.set_ylim(3e-3, 2.7)
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="lower left", labelspacing=0.3)
savefig(fig, "fig_e2_ladder.pdf")

# ---------------------------------------------------------------- E3 time
rows = load("e3_timeseries.csv")
T = len(rows)
t = np.arange(T)
U = 4
METHODS = ["OMA", "NOMA", "SR", "SC", "LMMSE", "DR-static", "DR-tracked",
           "DR-genie"]
S3 = {"DR-genie": ("k", ":"), "DR-tracked": ("tab:purple", "-"),
      "DR-static": ("tab:orange", "--"), "SR": ("tab:red", "--"),
      "SC": ("tab:green", "-."), "LMMSE": ("tab:blue", "-"),
      "OMA": ("0.45", ":"), "NOMA": ("tab:brown", ":")}


def roll(x, w=15):
    """Moving average with edge-truncated windows (no zero-padding bias:
    endpoints average only the samples that exist)."""
    x = np.asarray(x, float)
    num = np.convolve(x, np.ones(w), mode="same")
    den = np.convolve(np.ones_like(x), np.ones(w), mode="same")
    return num / den


fig = plt.figure(figsize=(2.9, 4.2))
_bh = 0.77 * 2.9 * 0.75 / 4.2            # same physical box height as new_fig
axes = [fig.add_axes([0.185, 0.549, 0.77, _bh]),
        fig.add_axes([0.185, 0.095, 0.77, _bh])]
axes[0].tick_params(labelbottom=False)
from matplotlib.lines import Line2D
for u in range(U):
    axes[0].plot(t, [float(r[f"a{u}"]) for r in rows], lw=1.1, color=f"C{u}")
    axes[0].plot(t, [float(r[f"ahat{u}"]) for r in rows], lw=0.9, ls="--",
                 color=f"C{u}", alpha=0.75)
axes[0].set_ylabel("share coefficient $a_u(t)$")
axes[0].set_ylim(0, 1.22)
axes[0].legend(handles=[
    Line2D([], [], color="k", ls="-", lw=1.1, label="true"),
    Line2D([], [], color="k", ls="--", lw=0.9, label="tracked")],
    loc="upper right", ncol=2, columnspacing=0.8)
axes[0].grid(alpha=0.3)
for m in METHODS:
    c, ls = S3[m]
    axes[1].semilogy(t, np.clip(roll([float(r[f"{m}_ser"]) for r in rows]),
                                1e-3, None), ls, color=c, label=m)
axes[1].set_xlabel("time slot $t$")
axes[1].set_ylabel("semantic error rate")
axes[1].set_ylim(2e-2, 30)
axes[1].grid(True, which="both", alpha=0.3)
axes[1].legend(loc="upper center", ncol=3, columnspacing=0.7,
               handletextpad=0.4, labelspacing=0.3, fontsize=6)
savefig(fig, "fig_e3_time.pdf")

# ---------------------------------------------------------------- E3 speed
rows = load("e3_speed.csv")
speeds = [float(r["speed"]) for r in rows]
marks = {"DR-genie": "d", "DR-tracked": "^", "DR-static": "s",
         "SR": "v", "SC": "x", "LMMSE": "o", "OMA": "1", "NOMA": "P"}
fig, ax = new_fig()
for m in METHODS:
    c, ls = S3[m]
    ax.semilogy(speeds, [max(float(r[f"{m}_ser"]), 1e-4) for r in rows], ls,
                color=c, marker=marks[m], label=m)
ax.set_xlabel("user speed (m/slot)")
ax.set_ylabel("mean semantic error rate")
ax.set_ylim(0.1, 40)
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper center", ncol=3, columnspacing=0.6, handletextpad=0.3,
          handlelength=1.4, labelspacing=0.25, fontsize=5.8, borderpad=0.3)
savefig(fig, "fig_e3_speed.pdf")

# ---------------------------------------------------------------- E4
rows = load("e4_mismatch.csv")
fig, ax = new_fig()
for snr_db, c, mk in ((5, "tab:red", "s"), (10, "tab:blue", "o"),
                      (15, "tab:green", "^")):
    pts = [(float(r["delta"]), float(r["cos"])) for r in rows
           if int(r["snr"]) == snr_db]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=c,
            marker=mk, label=f"{snr_db} dB")
ax.set_xlabel(r"share-coefficient estimation error $\delta$")
ax.set_ylabel("mean cosine recovery")
ax.set_ylim(0.40, 1.0)
ax.grid(alpha=0.3)
ax.legend(loc="upper center", ncol=3, columnspacing=0.9, handletextpad=0.4,
          borderpad=0.3)
savefig(fig, "fig_e4_mismatch.pdf")

# ---------------------------------------------------------------- E5
rows = load("e5_learned.csv")
betas5 = [float(r["beta"]) for r in rows]
fig, ax = new_fig()
S5 = {"Learned": ("tab:red", "--", "s", "learned attention"),
      "LMMSE": ("tab:blue", "-", "o", "LMMSE"),
      "DR": ("k", "-", "d", "Proposed DR")}
for k, (c, ls, mk, lb) in S5.items():
    ax.semilogy(betas5, [max(float(r[f"{k}_ser"]), 1e-3) for r in rows], ls,
                color=c, marker=mk, label=lb)
ax.set_xlabel(r"affinity $\beta$")
ax.set_ylabel("semantic error rate")
ax.set_ylim(8e-3, 3.2)
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="lower left", labelspacing=0.3)
savefig(fig, "fig_e5_learned.pdf")

print("all figures regenerated")
