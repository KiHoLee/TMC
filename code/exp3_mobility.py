"""E3: dynamic mobile environment -- time-varying share coefficients a_u(t)
from random-waypoint trajectories, with sparse affinity-pilot tracking.

Every K-th slot each user sends n_p orthogonal pilot embeddings (overhead
n_p/(K*batch) << 1); the tracker EWMA-smooths the triangulated observations.

Methods:
  DR-genie   : decomposition receiver with true a_u(t)         (upper ref)
  DR-tracked : DR with pilot-tracked a_hat(t)                  (proposed)
  DR-static  : DR designed for the time-averaged a (no adaptation)
  SR / SC / LMMSE : baselines with true B(t)

Outputs: data/e3_timeseries.csv, data/e3_speed.csv
(figures come from replot_all.py only)
"""
import os
import numpy as np
from semantic_mac import (affinity_matrix, matched_filter, demux_sr, demux_sc,
                          demux_lmmse, demux_dr, sample_latents_isotropic,
                          mobility_trajectories, AffinityTracker,
                          pilot_affinity_obs, metrics,
                          oma_observe, demux_noma_genie)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "fig")
DATA = os.path.join(HERE, "..", "data")
os.makedirs(FIG, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

from semantic_mac import U_MAIN as U, D_SYN as D, DC_SYN as DC  # Table IV
BATCH = 320
SNR_DB = 12
RHO = 10 ** (SNR_DB / 10)
T = 300
K_PILOT = 5          # pilot every K slots
N_PILOT = 64         # pilot embeddings per user per pilot slot
LAM = 0.3
METHODS = ["DR-genie", "DR-tracked", "DR-static", "SR", "SC", "LMMSE",
           "OMA", "NOMA"]
MOB = dict(box=50.0, r_scene=32.0, a_max=0.95)


def run_trace(a_t, rng, collect_ts=False):
    Vc = np.eye(D)[:, :DC]
    a_bar = a_t.mean(axis=0)
    tracker = AffinityTracker(U, lam=LAM, a_init=float(a_bar.mean()))
    sers = {m: [] for m in METHODS}
    coss = {m: [] for m in METHODS}
    a_hat_log = []
    for t in range(a_t.shape[0]):
        a = a_t[t]
        B = affinity_matrix(a)
        z = sample_latents_isotropic(BATCH, U, D, DC, a, rng)
        tilde, h = matched_filter(z, B, RHO, rng)

        if t % K_PILOT == 0:
            zp = sample_latents_isotropic(N_PILOT, U, D, DC, a, rng)
            hp = np.clip(np.abs((rng.standard_normal(U) +
                                 1j * rng.standard_normal(U)) / np.sqrt(2)),
                         0.2, None)
            tracker.update(pilot_affinity_obs(zp, hp, RHO, rng))
        a_hat = np.clip(tracker.a, 0.02, 0.95)

        out = {}
        out["DR-genie"] = demux_dr(tilde, B, h, RHO, a, Vc)
        out["DR-tracked"] = demux_dr(tilde, affinity_matrix(a_hat), h, RHO,
                                     a_hat, Vc)
        out["DR-static"] = demux_dr(tilde, affinity_matrix(a_bar), h, RHO,
                                    a_bar, Vc)
        out["SR"] = demux_sr(tilde, B, h)
        out["SC"] = demux_sc(tilde, h)
        out["LMMSE"], _ = demux_lmmse(tilde, B, h, RHO)
        out["OMA"] = oma_observe(z, h, RHO, rng)
        out["NOMA"] = demux_noma_genie(z, h, RHO, rng)

        a_hat_log.append(a_hat)
        for m in METHODS:
            c, _, s = metrics(out[m], z)
            sers[m].append(s)
            coss[m].append(c)
    res = {m: (float(np.mean(coss[m])), float(np.mean(sers[m])))
           for m in METHODS}
    if collect_ts:
        return res, sers, np.array(a_hat_log)
    return res


def drive_through_profile(T, peaks=(80, 110, 140, 170), width=45.0,
                          a_lo=0.05, a_hi=0.9):
    """Scene pass-by: each user approaches the shared scene, dwells, leaves."""
    t = np.arange(T)[:, None]
    pk = np.asarray(peaks)[None, :]
    return a_lo + (a_hi - a_lo) * np.exp(-(t - pk) ** 2 / (2 * width ** 2))


def main():
    # --- time-series: scene pass-by, averaged over REPS_TS runs -------------
    a_t = drive_through_profile(T)
    REPS_TS = 5
    sers_acc = None
    means_acc = {m: [] for m in METHODS}
    for rep_i in range(REPS_TS):
        rng = np.random.default_rng(3 + rep_i)
        res_i, sers_i, a_hat_i = run_trace(a_t, rng, collect_ts=True)
        if rep_i == 0:
            a_hat_log = a_hat_i
        if sers_acc is None:
            sers_acc = {m: np.array(sers_i[m], float) for m in METHODS}
        else:
            for m in METHODS:
                sers_acc[m] += np.array(sers_i[m], float)
        for m in METHODS:
            means_acc[m].append(res_i[m])
        print(f"  ts rep {rep_i+1}/{REPS_TS} done")
    sers = {m: (sers_acc[m] / REPS_TS).tolist() for m in METHODS}
    res = {m: (float(np.mean([x[0] for x in means_acc[m]])),
               float(np.mean([x[1] for x in means_acc[m]])))
           for m in METHODS}
    print("time-series means:", {m: f"cos={v[0]:.3f},ser={v[1]:.3f}"
                                 for m, v in res.items()})
    with open(os.path.join(DATA, "e3_timeseries.csv"), "w") as f:
        f.write("t," + ",".join(f"a{u}" for u in range(U)) + ","
                + ",".join(f"ahat{u}" for u in range(U)) + ","
                + ",".join(f"{m}_ser" for m in METHODS) + "\n")
        for t in range(T):
            f.write(f"{t}," + ",".join(f"{a_t[t,u]:.4f}" for u in range(U))
                    + "," + ",".join(f"{a_hat_log[t,u]:.4f}" for u in range(U))
                    + "," + ",".join(f"{sers[m][t]:.4f}" for m in METHODS) + "\n")

    # --- speed sweep (averaged over trajectory seeds) -----------------------
    speeds = [0.5, 1.0, 2.0, 4.0, 8.0]
    reps = 8
    rows = []
    for si, sp in enumerate(speeds):
        acc = {m: [] for m in METHODS}
        for rep in range(reps):
            # disjoint seed blocks per speed point (no seed reuse across
            # speeds)
            rng_s = np.random.default_rng(1000 + 100 * si + rep)
            a_tr = mobility_trajectories(U, T, sp, rng_s, **MOB)
            r = run_trace(a_tr, rng_s)
            for m in METHODS:
                acc[m].append(r[m])
        rows.append({m: (float(np.mean([x[0] for x in acc[m]])),
                         float(np.mean([x[1] for x in acc[m]])))
                     for m in METHODS})
        print(f"speed={sp} " + " ".join(f"{m}:ser={rows[-1][m][1]:.3f}"
                                        for m in METHODS))
    with open(os.path.join(DATA, "e3_speed.csv"), "w") as f:
        f.write("speed," + ",".join(f"{m}_cos,{m}_ser" for m in METHODS) + "\n")
        for sp, r in zip(speeds, rows):
            f.write(f"{sp}," + ",".join(f"{r[m][0]},{r[m][1]}"
                                        for m in METHODS) + "\n")

    # figures are produced only by the canonical replot_all.py (uniform
    # geometry); experiment scripts write CSVs exclusively.
    print("E3 done. Run replot_all.py to regenerate the figures.")


if __name__ == "__main__":
    main()
