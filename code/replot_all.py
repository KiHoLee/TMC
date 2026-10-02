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
    # thin dotted grid so the rules sit behind the curves rather than
    # competing with them at final column width
    "grid.linestyle": ":",
    "grid.linewidth": 0.45,
})


# canvas sized to the width the figures are actually placed at
# (0.79 of a 3.5 in column), so no PDF rescaling alters the text size
FIGW, FIGH = 2.765, 2.146
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


def _audit(fig, name, strict=True):
    """Check the drawn canvas, not the plotting calls.

    A text audit of this script cannot detect a clipped label or a curve
    hidden under a legend, because a label-setting call proves nothing
    about the rendered page. Only the drawn canvas can, so every figure is
    checked here before it is written."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = r.width, r.height
    # a text bounding box carries the font ascent and descent, so a few
    # pixels of overhang are normal; real clipping is far larger
    TOL = 5.0
    problems = []

    def outside(b):
        return (b.x0 < -TOL or b.y0 < -TOL or b.x1 > W + TOL or b.y1 > H + TOL)

    for ax in fig.axes:
        for lbl in (ax.xaxis.label, ax.yaxis.label):
            if lbl.get_text() and outside(lbl.get_window_extent(renderer=r)):
                problems.append(f"axis label {lbl.get_text()!r} clipped")
        for axis, lo_hi in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = sorted(lo_hi)
            for t in axis.get_ticklabels():
                # ticks outside the view interval still exist as objects but
                # are never drawn, so only the visible ones are checked
                pos = t.get_position()[0 if axis is ax.xaxis else 1]
                if not t.get_text() or not (lo <= pos <= hi):
                    continue
                if outside(t.get_window_extent(renderer=r)):
                    problems.append(f"tick label {t.get_text()!r} clipped")
        leg = ax.get_legend()
        if leg is None:
            continue
        lb = leg.get_window_extent(renderer=r)
        for line in ax.get_lines():
            xy = line.get_xydata()
            if len(xy) == 0:
                continue
            d = ax.transData.transform(xy)
            hit = ((d[:, 0] > lb.x0) & (d[:, 0] < lb.x1) &
                   (d[:, 1] > lb.y0) & (d[:, 1] < lb.y1))
            if hit.any():
                problems.append(
                    f"curve {line.get_label()!r} passes under the legend")
                break
    if problems:
        msg = name + ": " + "; ".join(sorted(set(problems)))
        if strict:
            raise AssertionError(msg)
        print("  FIGURE AUDIT:", msg)
    return problems


# pre-revision figures kept unchanged so the difference from the original
# submission stays visible; the manuscript includes only the _r1 versions
FROZEN = {"fig_e1_beta_cos.pdf", "fig_e1_snr.pdf", "fig_e3_time.pdf",
          "fig_e3_speed.pdf", "fig_e5_learned.pdf"}


def savefig(fig, name):
    _audit(fig, name, strict=name not in FROZEN)
    fig.savefig(os.path.join(FIG, name))
    print("saved", name)


S = {"OMA": ("0.45", ":", "v"), "NOMA": ("tab:brown", ":", "P"),
     "SR": ("tab:red", "--", "s"), "SC": ("tab:green", "-.", "^"),
     "LMMSE": ("tab:blue", "-", "o"), "DR": ("k", "-", "d")}
LBL = {"OMA": "OMA", "NOMA": "NOMA-SIC", "SR": "SimR",
       "SC": "SimC", "LMMSE": "LMMSE", "DR": "DR"}
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
            ms=7.5, mew=1.5, ls="none", label="SimR closed form")
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
ax.set_ylabel("eigenvalue")
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
      "DR-spec": ("tab:orange", "--", "s", "DR-spectral"),
      "DR-adapt": ("tab:purple", "-", "^", "DR-adapter"),
      "DR-oracle": ("k", ":", "d", "DR")}
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
      "DR": ("k", "-", "d", "DR")}
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

# ======================================================================
#  Revision 1 (TMC-2026-07-3202). New figures for the experiments added
#  in response to the reviewers, plus corrected versions of the figures
#  whose defects the reviewers flagged. The original files above are
#  regenerated unchanged so the two rounds remain comparable.
# ======================================================================

# styles for the receivers introduced in the revision, kept distinct from
# the existing triples in S
S.update({
    "BLMMSE": ("tab:purple", "-", "*"),
    "Learned": ("tab:orange", "--", "X"),
    "LearnedSI": ("tab:olive", "--", "h"),
})
LBL.update({
    "BLMMSE": "DR-MMSE",
    "Learned": "Learned",
    "LearnedSI": "Learned + side info",
})

# ------------------------------------------------- E1 SNR, corrected axis
# Reviewer 3 noted values outside the axis range: the DR curve reaches
# 1.25e-4 at 20 dB while the lower limit was 8e-4, so its last point fell
# outside the box. The limit now covers every plotted value.
rows = load("e1_snr.csv")
snrs = [float(r["snr"]) for r in rows]
fig, ax = new_fig()
vals = [float(r[f"{n}_ser"]) for n in ORDER for r in rows]
for n in ORDER:
    c, ls, mk = S[n]
    ax.semilogy(snrs, [float(r[f"{n}_ser"]) for r in rows], ls,
                color=c, marker=mk, label=LBL[n])
ax.set_xlabel("per-user SNR (dB)")
ax.set_ylabel("semantic error rate")
ax.set_ylim(min(v for v in vals if v > 0) * 0.5, 2.5)
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="lower left", ncol=1, fontsize=6.1, handletextpad=0.4,
          labelspacing=0.25, borderpad=0.3)
savefig(fig, "fig_e1_snr_r1.pdf")

# ------------------------------- E6/E7 receiver comparison, matched side info
rows = load("e7_learned_sideinfo.csv")
bt = [float(r["beta"]) for r in rows]
ORDER_SI = ["OMA", "NOMA", "LMMSE", "Learned", "LearnedSI", "BLMMSE", "DR"]
fig, ax = new_fig()
vals_si = [float(r[f"{n}_ser"]) for n in ORDER_SI for r in rows]
for n in ORDER_SI:
    c, ls, mk = S[n]
    y = [float(r[f"{n}_ser"]) for r in rows]
    # DR-MMSE and the DR coincide almost everywhere, which is
    # the finding, so they are separated by line width and by staggered
    # marker positions rather than left overlapping
    if n == "BLMMSE":
        ax.semilogy(bt, y, ls, color=c, marker=mk, label=LBL[n],
                    lw=2.8, alpha=0.45, markevery=(0, 2), ms=5.0)
    elif n == "DR":
        ax.semilogy(bt, y, ls, color=c, marker=mk, label=LBL[n],
                    markevery=(1, 2))
    else:
        ax.semilogy(bt, y, ls, color=c, marker=mk, label=LBL[n])
ax.set_xlabel(r"affinity $\beta$")
ax.set_ylabel("semantic error rate")
ax.set_ylim(min(vals_si) * 0.08, 9.0)
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="lower left", ncol=2, fontsize=5.7, columnspacing=0.6,
          handletextpad=0.3, handlelength=1.5, labelspacing=0.2,
          borderpad=0.28)
savefig(fig, "fig_e6_matched.pdf")

# ---------------------------------------------- E8 partial sharing sweep
rows = load("e8_partial_sharing.csv")
r05 = [r for r in rows if abs(float(r["beta_nominal"]) - 0.5) < 1e-9]
r08 = [r for r in rows if abs(float(r["beta_nominal"]) - 0.8) < 1e-9]
ks = [float(r["kappa"]) for r in r05]
fig, ax = new_fig()
for n in ["OMA", "NOMA", "LMMSE", "BLMMSE", "DR"]:
    c, ls, mk = S[n]
    y = [max(float(r[f"{n}_ser"]), 2e-4) for r in r05]
    if n == "BLMMSE":
        ax.semilogy(ks, y, ls, color=c, marker=mk, label=LBL[n],
                    lw=2.8, alpha=0.45, markevery=(0, 2), ms=5.0)
    elif n == "DR":
        ax.semilogy(ks, y, ls, color=c, marker=mk, label=LBL[n],
                    markevery=(1, 2))
    else:
        ax.semilogy(ks, y, ls, color=c, marker=mk, label=LBL[n])
# the second nominal affinity is reported in the text, not plotted,
# because a curve at a different operating point invites the figure
# to be read as a receiver comparison
ax.set_xlabel(r"non-common fraction $\kappa$")
ax.set_ylabel("semantic error rate")
ax.set_ylim(min(float(r["DR_ser"]) for r in r05) * 0.5, 6.0)
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="lower right", ncol=1, fontsize=5.9, handletextpad=0.3,
          handlelength=1.6, labelspacing=0.22, borderpad=0.3)
savefig(fig, "fig_e8_partial.pdf")

# -------------------------------------- E11 fairness under power imbalance
rows = load("e11_imbalance.csv")
rd = [float(r["ratio_db"]) for r in rows]
U_ = 4
fig, ax = new_fig()
allv = []
for n in ("OMA", "NOMA"):
    c, ls, mk = S[n]
    y = [float(r[f"{n}_ser"]) for r in rows]
    allv += y
    ax.semilogy(rd, y, ls, color=c, marker=mk, label=LBL[n])
for u in range(U_):
    yd = [float(r[f"DR_ser_u{u}"]) for r in rows]
    yl = [float(r[f"LMMSE_ser_u{u}"]) for r in rows]
    allv += yd + yl
    ax.semilogy(rd, yl, "-", color="tab:blue", marker="o",
                alpha=0.4 + 0.18 * u, lw=1.0, ms=2.8,
                label="LMMSE (per user)" if u == 0 else None)
    ax.semilogy(rd, yd, "-", color="k", marker="d",
                alpha=0.4 + 0.18 * u, lw=1.0, ms=2.8,
                label="DR (per user)" if u == 0 else None)
ax.set_xlabel("per-user power imbalance (dB)")
ax.set_ylabel("semantic error rate")
ax.set_ylim(min(allv) * 0.16, max(allv) * 2.2)
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="lower left", ncol=2, fontsize=5.9, columnspacing=0.6,
          handletextpad=0.3, handlelength=1.5, labelspacing=0.2,
          borderpad=0.28)
savefig(fig, "fig_e11_imbalance.pdf")

# ------------------------------- E3 figures with the uniform scheme labels
# Reviewer 3 noted that these two figures carried "NOMA" while every other
# figure carried "NOMA-SIC". The label dictionary is now applied here too,
# so one name per scheme is used throughout the paper.
LBL3 = {"OMA": "OMA", "NOMA": "NOMA-SIC", "SR": "SimR", "SC": "SimC",
        "LMMSE": "LMMSE", "DR-static": "DR-static", "DR-tracked": "DR-tracked",
        "DR-genie": "DR"}

rows = load("e3_timeseries.csv")
T = len(rows)
t = np.arange(T)
fig = plt.figure(figsize=(2.9, 4.2))
axes = [fig.add_axes([0.185, 0.549, 0.77, _bh]),
        fig.add_axes([0.185, 0.095, 0.77, _bh])]
axes[0].tick_params(labelbottom=False)
for u in range(4):
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
                                1e-3, None), ls, color=c, label=LBL3[m])
axes[1].set_xlabel("time slot $t$")
axes[1].set_ylabel("semantic error rate")
axes[1].set_ylim(2e-2, 30)
axes[1].grid(True, which="both", alpha=0.3)
axes[1].legend(loc="upper center", ncol=3, columnspacing=0.7,
               handletextpad=0.4, labelspacing=0.3, fontsize=6)
savefig(fig, "fig_e3_time_r1.pdf")

rows = load("e3_speed.csv")
speeds = [float(r["speed"]) for r in rows]
fig, ax = new_fig()
vals3 = [float(r[f"{m}_ser"]) for m in METHODS for r in rows]
for m in METHODS:
    c, ls = S3[m]
    ax.semilogy(speeds, [float(r[f"{m}_ser"]) for r in rows], ls,
                color=c, marker=marks[m], label=LBL3[m])
ax.set_xlabel("user speed (m/slot)")
ax.set_ylabel("mean semantic error rate")
ax.set_ylim(min(vals3) * 0.6, max(vals3) * 12)
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper center", ncol=3, columnspacing=0.6, handletextpad=0.3,
          handlelength=1.4, labelspacing=0.25, fontsize=5.8, borderpad=0.3)
savefig(fig, "fig_e3_speed_r1.pdf")

# ------------------------------- E12 trade-off set by the shared dimension
# On real corpus data the correlation is diffuse rather than low rank, so
# d_c controls how much of each embedding is subjected to combining. The
# two metrics move in opposite directions with it, and both approach the
# unstructured receiver as d_c grows.
rows = load("e12_corpus.csv")
dcs = [float(r["d_c"]) for r in rows]
fig, ax = new_fig()
# both quantities are reported as error rates so that one axis serves
# both and lower is better throughout
ax.semilogy(dcs, [float(r["DR_ser"]) for r in rows], "-", color="k",
            marker="d", label="DR (semantic)")
ax.semilogy(dcs, [1.0 - float(r["DR_scene"]) for r in rows], "--",
            color="tab:purple", marker="*", ms=5.0, label="DR (scene)")
# the MMSE complement recovers part of the scene loss at no cost in the
# semantic error rate, so both of its curves are shown
ax.semilogy(dcs, [float(r["BLMMSE_ser"]) for r in rows], "-",
            color="tab:red", marker="^", ms=4.0, label="DR-MMSE (semantic)")
ax.semilogy(dcs, [1.0 - float(r["BLMMSE_scene"]) for r in rows], "--",
            color="tab:orange", marker="X", ms=4.0, label="DR-MMSE (scene)")
ax.semilogy(dcs, [float(rows[0]["LMMSE_ser"])] * len(dcs), ":",
            color="tab:blue", marker="o", label="LMMSE (semantic)")
ax.semilogy(dcs, [1.0 - float(rows[0]["LMMSE_scene"])] * len(dcs), ":",
            color="tab:cyan", marker="s", label="LMMSE (scene)")
ax.set_xscale("log", base=2)
ax.set_xticks(dcs)
ax.set_xticklabels([str(int(d)) for d in dcs])
ax.set_xlabel(r"shared dimension $d_c$")
ax.set_ylabel("error rate")
ax.set_ylim(5e-3, 9.0)
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper left", ncol=2, fontsize=5.9, columnspacing=0.6,
          handletextpad=0.3, handlelength=1.6, labelspacing=0.2,
          borderpad=0.28)
savefig(fig, "fig_e12_tradeoff.pdf")

print("revision-1 figures done")
