"""E4: robustness of the decomposition receiver to affinity estimation error.

DR runs with a_hat = a + delta; theory predicts O(delta^2) degradation.

Outputs: data/e4_mismatch.csv
(figures come from replot_all.py only)
"""
import math
import os
import numpy as np
from semantic_mac import (affinity_matrix, matched_filter, demux_dr,
                          sample_latents_isotropic, metrics)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "fig")
DATA = os.path.join(HERE, "..", "data")
os.makedirs(FIG, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

U, D, DC = 4, 64, 16
BATCH = 4000
BETA = 0.5


def main():
    rng = np.random.default_rng(11)
    a = math.sqrt(BETA) * np.ones(U)
    B = affinity_matrix(a)
    Vc = np.eye(D)[:, :DC]
    deltas = np.arange(-0.20, 0.201, 0.04)
    rows = []
    for snr_db in (5, 10, 15):
        rho = 10 ** (snr_db / 10)
        z = sample_latents_isotropic(BATCH, U, D, DC, a, rng)
        tilde, h = matched_filter(z, B, rho, rng)
        for d0 in deltas:
            a_hat = np.clip(a + d0, 0.02, 0.98)
            B_hat = affinity_matrix(a_hat)
            cos, _, ser = metrics(
                demux_dr(tilde, B_hat, h, rho, a_hat, Vc), z)
            rows.append((snr_db, float(d0), cos, ser))
            print(f"snr={snr_db} delta={d0:+.2f} cos={cos:.4f} ser={ser:.4f}")
    with open(os.path.join(DATA, "e4_mismatch.csv"), "w") as f:
        f.write("snr,delta,cos,ser\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    # figures are produced only by the canonical replot_all.py (uniform
    # geometry); experiment scripts write CSVs exclusively.
    print("E4 done. Run replot_all.py to regenerate the figures.")


if __name__ == "__main__":
    main()
