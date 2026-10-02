"""E13: which receiver tolerates share-coefficient error better.

The DR and DR-MMSE (Remark 1) hold the same side
information, so a natural question is whether the simpler private branch
of the DR also behaves differently when that side information is wrong.

The two branches use the estimate in different ways. The DR cancels on
the complement with zero forcing, which inverts the coupling and is
unbiased whatever the signal prior is, so the share-coefficient estimate
enters its private branch only through a scalar shrinkage. The subspace
LMMSE receiver instead builds its private filter from the prior
covariance diag(1-a_u^2)/(d-d_c), so an error in the estimate perturbs
the filter itself. This experiment measures whether that difference is
visible.

Both receivers are given the same perturbed estimate a_hat = a + delta
and the affinity matrix built from it, exactly as E4 does for the DR
alone, so the comparison isolates the receiver.

Outputs: data/e13_mismatch_compare.csv, data/e13_notes.txt
"""
import math
import os
import numpy as np

from semantic_mac import (affinity_matrix, matched_filter, demux_dr,
                          sample_latents_isotropic, metrics)
from exp6_sideinfo import demux_block_lmmse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)

from semantic_mac import U_MAIN as U, D_SYN as D, DC_SYN as DC  # Table IV
BATCH = 4000
BETA = 0.5


def main():
    rng = np.random.default_rng(11)
    a = math.sqrt(BETA) * np.ones(U)
    B = affinity_matrix(a)
    Vc = np.eye(D)[:, :DC]
    deltas = np.arange(-0.20, 0.201, 0.04)
    rows = []
    notes = [f"beta={BETA}, U={U}, d={D}, d_c={DC}, batch={BATCH}",
             "both receivers receive the same perturbed estimate"]
    for snr_db in (5, 10, 15):
        rho = 10 ** (snr_db / 10)
        z = sample_latents_isotropic(BATCH, U, D, DC, a, rng)
        tilde, h = matched_filter(z, B, rho, rng)
        base = {}
        for d0 in deltas:
            a_hat = np.clip(a + d0, 0.02, 0.98)
            B_hat = affinity_matrix(a_hat)
            c_dr, _, s_dr = metrics(
                demux_dr(tilde, B_hat, h, rho, a_hat, Vc), z)
            c_bl, _, s_bl = metrics(
                demux_block_lmmse(tilde, B_hat, h, rho, a_hat, Vc), z)
            rows.append((snr_db, float(d0), c_dr, s_dr, c_bl, s_bl))
            if abs(d0) < 1e-9:
                base = {"dr": (c_dr, s_dr), "bl": (c_bl, s_bl)}
            print(f"snr={snr_db} delta={d0:+.2f} "
                  f"DR cos={c_dr:.4f} ser={s_dr:.4f} | "
                  f"subspace cos={c_bl:.4f} ser={s_bl:.4f}")
        # degradation at the extremes of the swept interval
        for tag, dd in (("-0.20", -0.20), ("+0.20", 0.20)):
            r = [x for x in rows if x[0] == snr_db
                 and abs(x[1] - dd) < 1e-9][0]
            notes.append(
                f"{snr_db:2d} dB, delta={tag}: DR cosine loss "
                f"{base['dr'][0]-r[2]:+.4f}, DR-MMSE cosine loss "
                f"{base['bl'][0]-r[4]:+.4f}")
    with open(os.path.join(DATA, "e13_mismatch_compare.csv"), "w") as f:
        f.write("snr,delta,DR_cos,DR_ser,BLMMSE_cos,BLMMSE_ser\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    with open(os.path.join(DATA, "e13_notes.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")
    print()
    print("\n".join(notes[2:]))
    print("E13 done.")


if __name__ == "__main__":
    main()
