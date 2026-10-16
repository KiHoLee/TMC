"""Regenerate all paper figures from the CSVs in ../data with
publication-quality layout (no legend/curve overlap, consistent styling,
conventional-scheme baselines included).

This is the canonical figure generator; experiment scripts write the CSVs.

Conventions enforced here (paper-requirement Section 9):
  * one style dictionary S, read by every plot call in every figure, so a
    scheme keeps one colour, one line style and one marker throughout;
  * one legend order, enforced by legend() from the RANK list;
  * one marker size and one legend font size, both set once in rcParams;
  * one canvas and one axes rectangle, placed 1:1 in the manuscript
    (0.68 of a 3.487 in column), so every box prints at the same size;
  * every figure is audited on the drawn canvas before it is written.
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "fig")
DATA = os.path.join(HERE, "..", "data")

plt.rcParams.update({
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "legend.fontsize": 6.0,       # the one legend size; never set per call
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.15,
    "lines.markersize": 3.2,      # the one marker size; never set per call
    "mathtext.fontset": "cm",     # figure math in the body font family
    # thin dotted grid so the rules sit behind the curves rather than
    # competing with them at final column width
    "grid.linestyle": ":",
    "grid.linewidth": 0.45,
})


# canvas equal to the placed width, WFRAC x 3.487 in (the IEEEtran column),
# so the manuscript includes every result figure at scale 1 and no rescaling
# alters text size. The margins are absolute: 0.57 in left and 0.40 in
# bottom are the measured extents of an 8.5 pt axis label plus 8 pt log-axis
# tick labels at scale 1 with a 2 px allowance, 0.08 in right and 0.147 in
# top. The axes box keeps an exact 8:6 shape at every width.
WFRAC = float(os.environ.get("TMC_FIG_WFRAC", "0.9"))
FIGW = round(WFRAC * 3.487, 3)
_ML, _MB, _MR, _MT = 0.569, 0.405, 0.085, 0.147
_AXW = FIGW - _ML - _MR
_AXH = 0.75 * _AXW
FIGH = round(_MB + _AXH + _MT, 3)
AXRECT = [_ML / FIGW, _MB / FIGH, _AXW / FIGW, _AXH / FIGH]
BOXH_IN = AXRECT[3] * FIGH             # printed box height in inches
# two-panel canvas: same width, each panel the same printed box as AXRECT,
# the same bottom margin, a 0.19 in gap and a 0.15 in top margin
FIGH2 = round(AXRECT[1] * FIGH + 2 * BOXH_IN + 0.19 + 0.15, 3)
AXRECT2 = [[AXRECT[0], (AXRECT[1] * FIGH + BOXH_IN + 0.19) / FIGH2,
            AXRECT[2], BOXH_IN / FIGH2],
           [AXRECT[0], AXRECT[1] * FIGH / FIGH2, AXRECT[2], BOXH_IN / FIGH2]]


def new_fig():
    """Canvas and axes rectangle identical for every figure, so every plot
    box renders at exactly the same size in the paper."""
    fig = plt.figure(figsize=(FIGW, FIGH))
    ax = fig.add_axes(AXRECT)
    return fig, ax


def new_fig2():
    """Two stacked panels, each with the printed box of new_fig()."""
    fig = plt.figure(figsize=(FIGW, FIGH2))
    axes = [fig.add_axes(AXRECT2[0]), fig.add_axes(AXRECT2[1])]
    axes[0].tick_params(labelbottom=False)
    return fig, axes


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
    px_per_pt = fig.dpi / 72.0
    # a text bounding box carries the font ascent and descent, so a pixel
    # or two of overhang is normal; real clipping is larger
    TOL = 2.0
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
        if outside(lb):
            problems.append("legend box leaves the canvas")
        for line in ax.get_lines():
            if not line.get_visible():
                continue
            xy = line.get_xydata()
            if len(xy) == 0:
                continue
            d = ax.transData.transform(xy)
            d = d[np.isfinite(d).all(axis=1)]
            if len(d) == 0:
                continue
            # a marker whose centre clears the box can still touch its
            # frame, so the box is inflated by the marker radius plus the
            # line width; segment midpoints catch a stroke that crosses
            # the box between two samples
            has_marker = line.get_marker() not in (None, "None", "", " ")
            drawn_line = line.get_linestyle() not in ("None", "none", "", " ")
            pad = ((line.get_markersize() / 2 if has_marker else 0.0)
                   + (line.get_linewidth() if drawn_line else 0.0)) * px_per_pt
            pts = d
            if drawn_line and len(d) > 1:
                pts = np.vstack([d, 0.5 * (d[1:] + d[:-1])])
            hit = ((pts[:, 0] > lb.x0 - pad) & (pts[:, 0] < lb.x1 + pad) &
                   (pts[:, 1] > lb.y0 - pad) & (pts[:, 1] < lb.y1 + pad))
            if hit.any():
                problems.append(
                    f"curve {line.get_label()!r} passes under the legend")
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


# ---------------------------------------------------------------- styles
# One (colour, line style, marker) triple per scheme, used by every figure.
# No two schemes share a triple. Solid marks a legitimate receiver, dashed
# and dash-dot the comparison schemes, dotted a reference level.
S = {
    "DR":         ("k",          "-",  "d"),
    "BLMMSE":     ("tab:purple", "-",  "*"),
    "DR-adapt":   ("tab:pink",   "-",  "p"),
    "DR-spec":    ("tab:cyan",   "--", "<"),
    "DR-tracked": ("tab:purple", "-",  "D"),
    "DR-static":  ("tab:orange", "--", ">"),
    "LMMSE":      ("tab:blue",   "-",  "o"),
    "SR":         ("tab:red",    "--", "s"),
    "SC":         ("tab:green",  "-.", "^"),
    "LearnedSI":  ("tab:olive",  "--", "h"),
    "Learned":    ("tab:orange", "--", "X"),
    "NOMA":       ("tab:brown",  ":",  "P"),
    "OMA":        ("0.45",       ":",  "v"),
}
# derived entries read colour and marker from the scheme they belong to:
# a reference level (dotted) and the closed-form overlays (markers only)
S["LMMSE-ref"] = (S["LMMSE"][0], ":", S["LMMSE"][2])
S["LMMSE-cf"] = (S["LMMSE"][0], "none", "x")
S["SR-cf"] = (S["SR"][0], "none", "+")

LBL = {"DR": "DR", "BLMMSE": "DR-MMSE", "DR-adapt": "DR-adapter",
       "DR-spec": "DR-spectral", "DR-tracked": "DR-tracked",
       "DR-static": "DR-static", "LMMSE": "LMMSE", "SR": "SimR",
       "SC": "SimC", "LearnedSI": "Learned + side info",
       "Learned": "Learned", "NOMA": "NOMA-SIC", "OMA": "OMA",
       "LMMSE-ref": "LMMSE", "LMMSE-cf": "LMMSE closed form",
       "SR-cf": "SimR closed form"}

# CSV column names that are aliases of a scheme in S
ALIAS = {"DR-genie": "DR", "DR-oracle": "DR"}

# parameter levels of one scheme (N, U, SNR sweeps); not schemes, so they
# do not live in S, and their colours avoid the scheme colours drawn in
# the same figure (the DR is black, the LMMSE reference blue)
LEVEL = [("tab:orange", "-.", "s"), ("tab:green", "--", "^"),
         ("tab:purple", "-", "o")]

# one legend order for every figure: proposal and its variants first,
# then the comparison schemes, then the closed-form overlays
RANK = ["DR", "DR-MMSE", "DR-adapter", "DR-spectral", "DR-tracked",
        "DR-static", "LMMSE", "SimR", "SimC", "Learned + side info",
        "Learned", "NOMA-SIC", "OMA", "closed form"]


def _rank(label):
    """Position of a legend label in RANK; a qualified label such as
    'DR scene' or 'LMMSE per user' ranks with its scheme, a closed-form
    entry goes last, and an unlisted label keeps its call order after
    the ranked ones."""
    if label.endswith("closed form"):
        return RANK.index("closed form")
    best, best_len = len(RANK), -1
    for i, n in enumerate(RANK):
        if (label == n or label.startswith(n + " ")) and len(n) > best_len:
            best, best_len = i, len(n)
    return best


def legend(ax, handles=None, **kw):
    """The shared legend helper: ranks the handles before the legend is
    built, so the order never follows the plotting order."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    else:
        labels = [h.get_label() for h in handles]
    idx = sorted(range(len(handles)), key=lambda i: (_rank(labels[i]), i))
    return ax.legend([handles[i] for i in idx], [labels[i] for i in idx], **kw)



def group_ellipse(ax, xk, ys, text, label_xy, ha="right"):
    """Ring the markers of one group of curves at abscissa xk and label it.

    The ellipse is drawn in printed inches around the markers' positions on
    the page, so it keeps its shape on a log axis: its height spans the
    group's markers plus a marker's width of padding, and its width is one
    marker column. The label is placed at label_xy in data coordinates,
    in an empty part of the plot next to the ellipse."""
    from matplotlib.patches import Ellipse
    fig = ax.figure
    pts = ax.transData.transform([(xk, y) for y in ys]) / fig.dpi
    cx = pts[:, 0].mean()
    lo, hi = pts[:, 1].min(), pts[:, 1].max()
    pad = 0.055
    ax.add_patch(Ellipse(((cx), (lo + hi) / 2), 0.15, (hi - lo) + 2 * pad,
                         transform=fig.dpi_scale_trans, fill=False, ec="0.3",
                         lw=0.7, ls=(0, (2.5, 1.5)), zorder=1, clip_on=False))
    ax.text(*label_xy, text, ha=ha, va="center", fontsize=6.5, style="italic",
            color="0.15")

def plot(ax, x, y, key, label=None, log=True, **kw):
    """Every curve of a scheme is drawn through here, so colour, line style
    and marker come from S and the label from LBL."""
    c, ls, mk = S[ALIAS.get(key, key)]
    f = ax.semilogy if log else ax.plot
    return f(x, y, ls=ls, color=c, marker=mk,
             label=LBL[ALIAS.get(key, key)] if label is None else label, **kw)


ORDER = ["OMA", "NOMA", "SR", "SC", "LMMSE", "DR"]

# ---------------------------------------------------------------- E1 beta
rows = load("e1_beta.csv")
betas = [float(r["beta"]) for r in rows]

fig, ax = new_fig()
for n in ORDER:
    # simulated LMMSE and SimR coincide with their closed forms by
    # construction, so the two marker sets are staggered rather than
    # drawn on top of each other
    me = (0, 2) if n in ("LMMSE", "SR") else None
    plot(ax, betas, [float(r[f"{n}_nmse"]) for r in rows], n, markevery=me)
plot(ax, betas, [float(r["LMMSE_cf"]) for r in rows], "LMMSE-cf",
     markevery=(1, 2))
plot(ax, betas, [float(r["SR_cf"]) for r in rows], "SR-cf", markevery=(1, 2))
ax.set_xlabel(r"affinity $\beta$")
ax.set_ylabel("NMSE")
# one decade of headroom above the SimR tail so the legend clears it
ax.set_ylim(6e-2, 8e7)
ax.grid(True, which="both", alpha=0.3)
legend(ax, loc="upper left", ncol=2, columnspacing=0.7, handletextpad=0.4,
       labelspacing=0.3)
savefig(fig, "fig_e1_beta_nmse.pdf")

fig, ax = new_fig()
for n in ORDER:
    plot(ax, betas, [float(r[f"{n}_cos"]) for r in rows], n, log=False)
ax.set_xlabel(r"affinity $\beta$")
ax.set_ylabel("mean cosine recovery")
ax.set_ylim(0.0, 1.4)
ax.grid(alpha=0.3)
legend(ax, loc="upper left", ncol=2, columnspacing=0.7, handletextpad=0.4,
       labelspacing=0.3)
savefig(fig, "fig_e1_beta_cos.pdf")

# ---------------------------------------------------------------- E1 snr
rows = load("e1_snr.csv")
snrs = [float(r["snr"]) for r in rows]
fig, ax = new_fig()
for n in ORDER:
    plot(ax, snrs, [max(float(r[f"{n}_ser"]), 1e-4) for r in rows], n)
ax.set_xlabel("per-user SNR (dB)")
ax.set_ylabel("semantic error rate")
ax.set_ylim(8e-4, 1.5)
ax.grid(True, which="both", alpha=0.3)
legend(ax, loc="lower left", ncol=1, handletextpad=0.4, labelspacing=0.25,
       borderpad=0.3)
savefig(fig, "fig_e1_snr.pdf")

# ---------------------------------------------- E2 spectrum (multi-curve)
rows_s = load("e2_spectrum_multi.csv")
fig, ax = new_fig()
for i, n_cal in enumerate((100, 400, 1600)):
    c, ls, mk = LEVEL[i]
    pts = [(int(r["idx"]), float(r["eig"])) for r in rows_s
           if int(r["N"]) == n_cal]
    ax.semilogy([p[0] for p in pts],
                np.maximum([p[1] for p in pts], 1e-12), ls=ls, color=c,
                marker=mk, markevery=(6 * i, 24), label=f"$N{{=}}{n_cal}$")
ax.axvline(128, color="k", ls=":", lw=0.9)
ax.annotate(r"$d_c=128$", xy=(128, 1e-6), xytext=(150, 3e-7),
            arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"))
ax.set_xlabel("eigenvalue index")
ax.set_ylabel("eigenvalue")
ax.set_ylim(1e-8, 3e-1)
ax.grid(True, which="both", alpha=0.3)
legend(ax, loc="upper right", labelspacing=0.3)
savefig(fig, "fig_e2_spectrum.pdf")

# ---------------------------------------------- E2 subspace (multi-curve)
rows_n = load("e2_subspace_multi.csv")
fig, ax = new_fig()
for i, Um in enumerate((2, 4, 8)):
    c, ls, mk = LEVEL[i]
    pts = [(int(r["N"]), float(r["err"])) for r in rows_n
           if int(r["U"]) == Um]
    ax.loglog([p[0] for p in pts], [p[1] for p in pts], ls=ls, color=c,
              marker=mk, label=f"$U{{=}}{Um}$")
ax.set_xlabel("paired calibration samples $N$")
ax.set_ylabel("subspace recovery error")
# plain tick labels at the measured N values; suppress the crowded
# mantissa minor labels of the default log locator. The y axis uses
# short decimal labels so that the axis title fits inside the fixed
# canvas (wide mantissa labels pushed it off the left edge).
Ns_ticks = [200, 400, 800, 1600, 3200, 6400]
ax.set_xticks(Ns_ticks)
ax.set_xticklabels([str(n) for n in Ns_ticks])
ax.xaxis.set_minor_locator(NullLocator())
y_ticks = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6]
ax.set_yticks(y_ticks)
ax.set_yticklabels([f"{v:g}" for v in y_ticks])
ax.yaxis.set_minor_locator(NullLocator())
# the curves fall from the upper left to the lower right, so the legend
# sits in the lower-left corner, with the floor lowered to give it room
ax.set_ylim(0.04, 0.7)
ax.grid(True, which="both", alpha=0.3)
legend(ax, loc="lower left", labelspacing=0.3)
savefig(fig, "fig_e2_subspace.pdf")

# ---------------------------------------------------------------- E2 ladder
rows = load("e2_ladder.csv")
snrs = [float(r["snr"]) for r in rows]
fig, ax = new_fig()
for k in ("OMA", "NOMA", "LMMSE", "DR-spec", "DR-adapt", "DR-oracle"):
    plot(ax, snrs, [max(float(r[f"{k}_ser"]), 1e-4) for r in rows], k)
ax.set_xlabel("per-user SNR (dB)")
ax.set_ylabel("semantic error rate")
ax.set_ylim(3e-3, 1.5)
ax.grid(True, which="both", alpha=0.3)
legend(ax, loc="lower left", labelspacing=0.3)
savefig(fig, "fig_e2_ladder.pdf")

# ---------------------------------------------------------------- E3 time
rows = load("e3_timeseries.csv")
T = len(rows)
t = np.arange(T)
U = 4
METHODS = ["OMA", "NOMA", "SR", "SC", "LMMSE", "DR-static", "DR-tracked",
           "DR-genie"]


def roll(x, w=15):
    """Moving average with edge-truncated windows (no zero-padding bias:
    endpoints average only the samples that exist)."""
    x = np.asarray(x, float)
    num = np.convolve(x, np.ones(w), mode="same")
    den = np.convolve(np.ones_like(x), np.ones(w), mode="same")
    return num / den


def e3_time_figure(rows, labels):
    """Two stacked panels: share coefficients (top) and rolling SER
    (bottom). The eight SER curves carry sparse, staggered markers so each
    scheme keeps its marker from S without hiding the lines."""
    fig, axes = new_fig2()
    for u in range(U):
        axes[0].plot(t, [float(r[f"a{u}"]) for r in rows], lw=1.1,
                     color=f"C{u}")
        axes[0].plot(t, [float(r[f"ahat{u}"]) for r in rows], lw=0.9, ls="--",
                     color=f"C{u}", alpha=0.75)
    axes[0].set_ylabel("share coefficient $a_u(t)$")
    axes[0].set_ylim(0, 1.22)
    legend(axes[0], handles=[
        Line2D([], [], color="k", ls="-", lw=1.1, label="true"),
        Line2D([], [], color="k", ls="--", lw=0.9, label="tracked")],
        loc="upper right", ncol=2, columnspacing=0.8)
    axes[0].grid(alpha=0.3)
    for i, m in enumerate(METHODS):
        plot(axes[1], t, np.clip(roll([float(r[f"{m}_ser"]) for r in rows]),
                                 1e-3, None), m, label=labels[m],
             markevery=(4 * i, 30))
    axes[1].set_xlabel("time slot $t$")
    axes[1].set_ylabel("semantic error rate")
    axes[1].set_ylim(6e-3, 1.5)        # error rates never exceed one
    axes[1].grid(True, which="both", alpha=0.3)
    legend(axes[1], loc="lower center", ncol=3, columnspacing=0.5,
           handletextpad=0.3, handlelength=1.2, labelspacing=0.25,
           borderpad=0.25)
    return fig


# pre-revision labels, kept for the frozen originals only
LBL_OLD = {m: m for m in METHODS}
fig = e3_time_figure(rows, LBL_OLD)
savefig(fig, "fig_e3_time.pdf")

# ---------------------------------------------------------------- E3 speed
rows = load("e3_speed.csv")
speeds = [float(r["speed"]) for r in rows]


def e3_speed_figure(rows, labels, ylim):
    fig, ax = new_fig()
    for m in METHODS:
        plot(ax, speeds, [max(float(r[f"{m}_ser"]), 1e-4) for r in rows], m,
             label=labels[m])
    ax.set_xlabel("user speed (m/slot)")
    ax.set_ylabel("mean semantic error rate")
    ax.set_ylim(0.025, 1.5)            # legend in the empty band below DR
    ax.grid(True, which="both", alpha=0.3)
    legend(ax, loc="lower center", ncol=3, columnspacing=0.5,
           handletextpad=0.3, handlelength=1.2, labelspacing=0.25,
           borderpad=0.25)
    return fig


fig = e3_speed_figure(rows, LBL_OLD, (0.1, 40))
savefig(fig, "fig_e3_speed.pdf")

# ---------------------------------------------------------------- E4
rows = load("e4_mismatch.csv")
ref = load("e4_lmmse_ref.csv")
fig, ax = new_fig()
deltas = sorted({float(r["delta"]) for r in rows})
for i, snr_db in enumerate((5, 10, 15)):
    pts = [(float(r["delta"]), float(r["cos"])) for r in rows
           if int(r["snr"]) == snr_db]
    # one scheme (the DR) at three operating points: the DR line style and
    # marker from S, one level colour per SNR
    ax.plot([p[0] for p in pts], [p[1] for p in pts], ls=S["DR"][1],
            color=LEVEL[i][0], marker=S["DR"][2], label=f"DR {snr_db} dB")
# the LMMSE does not consume the share coefficient, so its cosine
# recovery at each SNR is a delta-independent reference level
for i, r in enumerate(ref):
    plot(ax, deltas, [float(r["cos"])] * len(deltas), "LMMSE-ref",
         label="LMMSE" if i == 0 else "_nolegend_", log=False,
         markevery=(1, 2))
ax.set_xlabel(r"share-coefficient error $\delta$")
ax.set_ylabel("mean cosine recovery")
ax.set_ylim(0.30, 1.0)
ax.grid(alpha=0.3)
legend(ax, loc="upper center", ncol=2, columnspacing=0.9, handletextpad=0.4,
       borderpad=0.3)
savefig(fig, "fig_e4_mismatch.pdf")

# ---------------------------------------------------------------- E5
rows = load("e5_learned.csv")
betas5 = [float(r["beta"]) for r in rows]
fig, ax = new_fig()
for k, lb in (("Learned", "learned attention"), ("LMMSE", None), ("DR", None)):
    plot(ax, betas5, [max(float(r[f"{k}_ser"]), 1e-3) for r in rows], k,
         label=lb)
ax.set_xlabel(r"affinity $\beta$")
ax.set_ylabel("semantic error rate")
ax.set_ylim(8e-3, 1.5)
ax.grid(True, which="both", alpha=0.3)
legend(ax, loc="upper right", labelspacing=0.3)
savefig(fig, "fig_e5_learned.pdf")

print("all figures regenerated")

# ======================================================================
#  Revision 1 (TMC-2026-07-3202). New figures for the experiments added
#  in response to the reviewers, plus corrected versions of the figures
#  whose defects the reviewers flagged. The original files above are
#  regenerated unchanged so the two rounds remain comparable.
# ======================================================================

# ------------------------------------------------- E1 SNR, corrected axis
# Reviewer 3 noted values outside the axis range: the DR curve reaches
# 1.25e-4 at 20 dB while the lower limit was 8e-4, so its last point fell
# outside the box. The limit now covers every plotted value.
rows = load("e1_snr.csv")
snrs = [float(r["snr"]) for r in rows]
fig, ax = new_fig()
vals = [float(r[f"{n}_ser"]) for n in ORDER for r in rows]
for n in ORDER:
    plot(ax, snrs, [float(r[f"{n}_ser"]) for r in rows], n)
ax.set_xlabel("per-user SNR (dB)")
ax.set_ylabel("semantic error rate")
ax.set_ylim(min(v for v in vals if v > 0) * 0.5, 1.5)
ax.grid(True, which="both", alpha=0.3)
legend(ax, loc="lower left", ncol=1, handletextpad=0.4, labelspacing=0.25,
       borderpad=0.3)
savefig(fig, "fig_e1_snr_r1.pdf")

# ------------------------------- E6/E7 receiver comparison, matched side info
rows = load("e7_learned_sideinfo.csv")
bt = [float(r["beta"]) for r in rows]
ORDER_SI = ["OMA", "NOMA", "LMMSE", "Learned", "LearnedSI", "BLMMSE", "DR"]
fig, ax = new_fig()
vals_si = [float(r[f"{n}_ser"]) for n in ORDER_SI for r in rows]
for n in ORDER_SI:
    y = [float(r[f"{n}_ser"]) for r in rows]
    # DR-MMSE and the DR coincide almost everywhere, which is
    # the finding, so they are separated by line width and by staggered
    # marker positions rather than left overlapping
    if n == "BLMMSE":
        plot(ax, bt, y, n, lw=2.8, alpha=0.45, markevery=(0, 2))
    elif n == "DR":
        plot(ax, bt, y, n, markevery=(1, 2))
    elif n in ("OMA", "NOMA"):
        # the two flat reference levels are named in the caption rather
        # than in the legend (writing standard 12.7), which frees the room
        # that seven entries would need
        plot(ax, bt, y, n, label="_nolegend_")
    else:
        plot(ax, bt, y, n)
ax.set_xlabel(r"affinity $\beta$")
ax.set_ylabel("semantic error rate")
# every curve starts near one at low affinity, so the lower-left corner is
# empty; the floor is lowered to host the legend there (error rates never
# exceed one, so no headroom above is admissible)
ax.set_ylim(min(vals_si) * 0.02, 1.5)
ax.grid(True, which="both", alpha=0.3)
legend(ax, loc="lower left", ncol=1, columnspacing=0.5, handletextpad=0.3,
       handlelength=1.2, labelspacing=0.15, borderpad=0.2)
savefig(fig, "fig_e6_matched.pdf")

# ---------------------------------------------- E8 partial sharing sweep
rows = load("e8_partial_sharing.csv")
r05 = [r for r in rows if abs(float(r["beta_nominal"]) - 0.5) < 1e-9]
r08 = [r for r in rows if abs(float(r["beta_nominal"]) - 0.8) < 1e-9]
ks = [float(r["kappa"]) for r in r05]
fig, ax = new_fig()
for n in ["OMA", "NOMA", "LMMSE", "BLMMSE", "DR"]:
    y = [max(float(r[f"{n}_ser"]), 2e-4) for r in r05]
    if n == "BLMMSE":
        plot(ax, ks, y, n, lw=2.8, alpha=0.45, markevery=(0, 2))
    elif n == "DR":
        plot(ax, ks, y, n, markevery=(1, 2))
    else:
        plot(ax, ks, y, n)
# the second nominal affinity is reported in the text, not plotted,
# because a curve at a different operating point invites the figure
# to be read as a receiver comparison
ax.set_xlabel(r"non-common fraction $\kappa$")
ax.set_ylabel("semantic error rate")
ax.set_ylim(min(float(r["DR_ser"]) for r in r05) * 0.5, 1.5)
ax.grid(True, which="both", alpha=0.3)
legend(ax, loc="lower right", ncol=1, handletextpad=0.3, handlelength=1.6,
       labelspacing=0.22, borderpad=0.3)
savefig(fig, "fig_e8_partial.pdf")

# -------------------------------------- E11 fairness under power imbalance
rows = load("e11_imbalance.csv")
rd = [float(r["ratio_db"]) for r in rows]
U_ = 4
fig, ax = new_fig()
allv = []
for n in ("OMA", "NOMA"):
    y = [float(r[f"{n}_ser"]) for r in rows]
    allv += y
    plot(ax, rd, y, n)
for u in range(U_):
    yd = [float(r[f"DR_ser_u{u}"]) for r in rows]
    yl = [float(r[f"LMMSE_ser_u{u}"]) for r in rows]
    allv += yd + yl
    plot(ax, rd, yl, "LMMSE", alpha=0.4 + 0.18 * u, lw=1.0,
         label="LMMSE per user" if u == 0 else "_nolegend_")
    plot(ax, rd, yd, "DR", alpha=0.4 + 0.18 * u, lw=1.0,
         label="DR per user" if u == 0 else "_nolegend_")
ax.set_xlabel("per-user power imbalance (dB)")
ax.set_ylabel("semantic error rate")
ax.set_ylim(min(allv) * 0.16, max(allv) * 2.2)
ax.grid(True, which="both", alpha=0.3)
legend(ax, loc="lower left", ncol=2, columnspacing=0.6, handletextpad=0.3,
       handlelength=1.5, labelspacing=0.2, borderpad=0.28)
savefig(fig, "fig_e11_imbalance.pdf")

# ------------------------------- E3 figures with the uniform scheme labels
# Reviewer 3 noted that these two figures carried "NOMA" while every other
# figure carried "NOMA-SIC". The label dictionary is now applied here too,
# so one name per scheme is used throughout the paper.
LBL3 = {m: LBL[ALIAS.get(m, m)] for m in METHODS}

rows = load("e3_timeseries.csv")
fig = e3_time_figure(rows, LBL3)
savefig(fig, "fig_e3_time_r1.pdf")

rows = load("e3_speed.csv")
vals3 = [float(r[f"{m}_ser"]) for m in METHODS for r in rows]
fig = e3_speed_figure(rows, LBL3, (min(vals3) * 0.6, max(vals3) * 12))
savefig(fig, "fig_e3_speed_r1.pdf")

# ------------------------------- E12 trade-off set by the shared dimension
# On real corpus data the correlation is diffuse rather than low rank, so
# d_c controls how much of each embedding is subjected to combining. The
# two metrics move in opposite directions with it, and both approach the
# unstructured receiver as d_c grows.
rows = load("e12_corpus.csv")
dcs = [float(r["d_c"]) for r in rows]
fig, ax = new_fig()
# both quantities are error rates on one axis, so lower is better
# throughout; each receiver keeps its style from S. As in every other
# figure the legend names the receivers only. The two metrics are told
# apart by an ellipse that groups their curves and is labeled on the plot,
# and the scene curves repeat the cue with open markers. The MMSE
# complement recovers part of the scene loss at no cost in the semantic
# error rate, so both of its curves are shown. The unstructured receiver
# does not depend on d_c and appears as two reference levels.
SEM = {"DR": [float(r["DR_ser"]) for r in rows],
       "BLMMSE": [float(r["BLMMSE_ser"]) for r in rows],
       "LMMSE-ref": [float(rows[0]["LMMSE_ser"])] * len(dcs)}
SCN = {"DR": [1.0 - float(r["DR_scene"]) for r in rows],
       "BLMMSE": [1.0 - float(r["BLMMSE_scene"]) for r in rows],
       "LMMSE-ref": [1.0 - float(rows[0]["LMMSE_scene"])] * len(dcs)}
for k in ("DR", "BLMMSE", "LMMSE-ref"):
    plot(ax, dcs, SEM[k], k)
    plot(ax, dcs, SCN[k], k, label="_nolegend_", mfc="none")
ax.set_xscale("log", base=2)
ax.set_xticks(dcs)
ax.set_xticklabels([str(int(d)) for d in dcs])
ax.set_xlabel(r"shared dimension $d_c$")
ax.set_ylabel("error rate")
# the band below the flat LMMSE scene level is empty across the whole
# width, so the floor leaves room for the one-row legend there
ax.set_ylim(2e-3, 1.5)
ax.grid(True, which="both", alpha=0.3)
legend(ax, loc="lower center", ncol=3, columnspacing=0.8, handletextpad=0.3,
       handlelength=1.6, borderpad=0.25)
# each group is ringed where its markers sit closest, at the largest d_c,
# and labeled in the empty space to the left of the ring: above the
# LMMSE semantic level, and between the DR-MMSE and LMMSE scene curves
group_ellipse(ax, dcs[-1], [v[-1] for v in SEM.values()], "semantic", (215, 1.0))
group_ellipse(ax, dcs[-1], [v[-1] for v in SCN.values()], "scene", (215, 0.0155))
savefig(fig, "fig_e12_tradeoff.pdf")

print("revision-1 figures done")
