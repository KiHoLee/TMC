"""E4 reference: LMMSE cosine recovery at the E4 operating point.

The LMMSE receiver does not consume the share coefficient, so its cosine
recovery is independent of the estimation error delta swept by
exp4_mismatch.py. One value per SNR is computed here at the same
configuration (U, d, d_c, batch, beta) and drawn by replot_all.py as a
horizontal reference level in fig_e4_mismatch.pdf.

Outputs: data/e4_lmmse_ref.csv  (columns snr, cos, ser)
"""
import math
import os
import numpy as np
from semantic_mac import (affinity_matrix, matched_filter, demux_lmmse,
                          sample_latents_isotropic, metrics)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)

# identical to exp4_mismatch.py
U, D, DC = 4, 64, 16
BATCH = 4000
BETA = 0.5
SEED = 1104   # own fixed seed; the DR sweep keeps seed 11


def main():
    rng = np.random.default_rng(SEED)
    a = math.sqrt(BETA) * np.ones(U)
    B = affinity_matrix(a)
    rows = []
    for snr_db in (5, 10, 15):
        rho = 10 ** (snr_db / 10)
        z = sample_latents_isotropic(BATCH, U, D, DC, a, rng)
        tilde, h = matched_filter(z, B, rho, rng)
        est, _ = demux_lmmse(tilde, B, h, rho)
        cos, _, ser = metrics(est, z)
        rows.append((snr_db, cos, ser))
        print(f"snr={snr_db} LMMSE cos={cos:.4f} ser={ser:.4f}")
    with open(os.path.join(DATA, "e4_lmmse_ref.csv"), "w") as f:
        f.write("snr,cos,ser\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print("E4 LMMSE reference done. Run replot_all.py to regenerate the figure.")


if __name__ == "__main__":
    main()
