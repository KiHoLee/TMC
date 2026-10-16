"""E1 check: the 20 dB point of the SNR sweep (Fig. 3) with 4x10^4 draws.

The plotted DR value at 20 dB rests on two errors in 1.6x10^4 decisions.
This run repeats the point (and 16 dB) with 4x10^4 draws, 1.6x10^5
per-user decisions, the count the manuscript says resolves an error rate
of 1e-4 (Comment 3.5 of the review).

Output: data/e1_snr20_check.txt"""
import math, sys, os
import numpy as np
from semantic_mac import (affinity_matrix, matched_filter, demux_sr, demux_sc,
                          demux_lmmse, demux_dr, sample_latents_isotropic, metrics)
U, D, DC = 4, 64, 16
BATCH = 40000
lines = []
for snr_db, seed in ((20, 2020), (16, 1616)):
    rng = np.random.default_rng(seed)
    rho = 10 ** (snr_db / 10); beta = 0.4
    a = math.sqrt(beta) * np.ones(U); B = affinity_matrix(a)
    z = sample_latents_isotropic(BATCH, U, D, DC, a, rng)
    tilde, h = matched_filter(z, B, rho, rng)
    Vc = np.eye(D)[:, :DC]
    dr = metrics(demux_dr(tilde, B, h, rho, a, Vc), z)
    lm = metrics(demux_lmmse(tilde, B, h, rho)[0], z)
    sc = metrics(demux_sc(tilde, h), z)
    n = BATCH * U
    lines.append(f"snr={snr_db} draws={BATCH} decisions={n} DR_ser={dr[2]:.6f} ({dr[2]*n:.0f} errors) "
          f"LMMSE_ser={lm[2]:.5f} SC_ser={sc[2]:.5f} LMMSE/DR={lm[2]/max(dr[2],1e-12):.1f}")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "e1_snr20_check.txt")
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
