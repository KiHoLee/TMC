"""E11 draw-to-draw spread of the Table VI entries (Comment 4.2 of the review). Each Table VI column uses its own generator (seed 101+deg or
202+100*sigma_e in code/exp11_impairments.py). Here the zero-impairment point is
repeated with 16 independent seeds (BATCH=4000 draws each, as in the paper) to
measure how much an entry moves between independent draws alone.
Also recomputes the LMMSE/DR ratios of Table VI from data/e11_*.csv.
"""
import math, os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from semantic_mac import affinity_matrix, matched_filter, demux_lmmse, demux_dr, metrics, sample_latents_isotropic
from exp11_impairments import rayleigh, U, D, DC, BATCH, BETA, SNR_DB

rho = 10**(SNR_DB/10); a = math.sqrt(BETA)*np.ones(U); B = affinity_matrix(a); Vc = np.eye(D)[:, :DC]
lm_s, dr_s = [], []
for s in range(16):
    rng = np.random.default_rng(900000 + s)
    z = sample_latents_isotropic(BATCH, U, D, DC, a, rng)
    h = rayleigh(BATCH, rng)
    tilde, _ = matched_filter(z, B, rho, rng, h_fixed=h)
    lm, _ = demux_lmmse(tilde, B, h, rho)
    lm_s.append(metrics(lm, z)[2]); dr_s.append(metrics(demux_dr(tilde, B, h, rho, a, Vc), z)[2])
lm_s, dr_s = np.array(lm_s), np.array(dr_s)
print(f"zero impairment, 16 independent seeds x {BATCH} draws")
print(f"LMMSE SER mean {lm_s.mean():.4f} sd {lm_s.std(ddof=1):.4f} range {lm_s.min():.4f}-{lm_s.max():.4f}")
print(f"DR    SER mean {dr_s.mean():.4f} sd {dr_s.std(ddof=1):.4f} range {dr_s.min():.4f}-{dr_s.max():.4f}")
print(f"ratio LMMSE/DR per seed range {(lm_s/dr_s).min():.2f}-{(lm_s/dr_s).max():.2f}")

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
for fn, key in (("e11_phase.csv", "phase_deg"), ("e11_cest.csv", "sigma_e"), ("e11_imbalance.csv", "ratio_db")):
    with open(os.path.join(P, fn)) as f:
        rows = list(csv.DictReader(f))
    print(fn, [(r[key], round(float(r["LMMSE_ser"])/float(r["DR_ser"]), 2)) for r in rows])
