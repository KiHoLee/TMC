"""E11 mechanism: why all four DR curves of Fig. 13 rise with imbalance.

Reproduces the E11 imbalance draws exactly (seed 303+rdb, as in
exp11_impairments.py) and splits the DR error into its shared and private
branches by giving the receiver the true value of one branch at a time
(Comment 4.2 of the review).

Output: stdout, saved as data/e11_mechanism.txt.
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from semantic_mac import (affinity_matrix, matched_filter, sample_latents_isotropic, TAU)
from exp11_impairments import rayleigh, U, D, DC, BATCH, BETA, SNR_DB

def dr_parts(tilde, B, h, rho, a, z):
    batch = tilde.shape[0]
    sigma2 = 1.0 / rho
    Binv = np.linalg.inv(B)
    Zs = tilde[:, :, :DC]; Zp = tilde.copy(); Zp[:, :, :DC] = 0
    sh = np.zeros_like(tilde); pr = np.zeros_like(tilde); prec = np.zeros(batch)
    for b in range(batch):
        hb = h[b]; Hi = np.diag(1/hb)
        Gamma = Hi @ B @ np.diag(hb); Cn = sigma2 * (Hi @ B @ Hi)
        gamma = np.array([a[u] + sum(B[u, v]*a[v]*hb[v]/hb[u] for v in range(U) if v != u) for u in range(U)])
        Ci = np.linalg.inv(Cn); den = float(gamma @ Ci @ gamma); prec[b] = den
        c_hat = (gamma @ Ci @ Zs[b]) / den
        c_hat *= (1.0/DC) / (1.0/DC + 1.0/den)
        Gp = np.linalg.solve(Gamma, Zp[b])
        for u in range(U):
            sig_p = 1 - a[u]**2
            err_p = (D - DC) * sigma2 * Binv[u, u] / hb[u]**2
            sh[b, u, :DC] = a[u] * c_hat
            pr[b, u] = sig_p/(sig_p+err_p) * Gp[u]
    return sh, pr, prec

def ser_u(e, z):
    en = e / (np.linalg.norm(e, axis=2, keepdims=True) + 1e-12)
    return ((en*z).sum(-1) < TAU).mean(axis=0)

rho = 10**(SNR_DB/10); a = math.sqrt(BETA)*np.ones(U); B = affinity_matrix(a)
print("rdb | DR per-user SER (weak..strong) | genie-shared (private errors only) | genie-private (shared errors only) | mean shared precision a'HBHa/sigma2 | mean shared MSE per coord")
for rdb in (0, 3, 6, 10, 15):
    rng = np.random.default_rng(303 + rdb)
    g = np.logspace(-rdb/40, rdb/40, U); g = g/np.sqrt((g**2).mean())
    z = sample_latents_isotropic(BATCH, U, D, DC, a, rng)
    h = rayleigh(BATCH, rng, scale=g)
    tilde, _ = matched_filter(z, B, rho, rng, h_fixed=h)
    sh, pr, prec = dr_parts(tilde, B, h, rho, a, z)
    zs = z.copy(); zs[:, :, DC:] = 0; zp = z.copy(); zp[:, :, :DC] = 0
    full = ser_u(sh+pr, z); gsh = ser_u(zs+pr, z); gpr = ser_u(sh+zp, z)
    cerr = ((sh[:, 0, :DC]/a[0] - z[:, 0, :DC]/a[0])**2).sum(-1).mean()
    print(f"{rdb:2d} | {np.round(full,4)} | {np.round(gsh,4)} | {np.round(gpr,4)} | {prec.mean():.1f} | shared ||c_hat-c||^2 = {cerr:.4f}")

print("\nPer-branch squared error per user (weak..strong); shared error is common to all users")
for rdb in (0, 3, 6, 10, 15):
    rng = np.random.default_rng(303 + rdb)
    g = np.logspace(-rdb/40, rdb/40, U); g = g/np.sqrt((g**2).mean())
    z = sample_latents_isotropic(BATCH, U, D, DC, a, rng)
    h = rayleigh(BATCH, rng, scale=g)
    tilde, _ = matched_filter(z, B, rho, rng, h_fixed=h)
    sh, pr, prec = dr_parts(tilde, B, h, rho, a, z)
    zs = z.copy(); zs[:, :, DC:] = 0; zp = z.copy(); zp[:, :, :DC] = 0
    es = ((sh - zs)**2).sum(-1).mean(0); ep = ((pr - zp)**2).sum(-1).mean(0)
    cs = (np.sum(h, 1)**2).mean(); s2 = (h**2).sum(1).mean()
    print(f"{rdb:2d} dB: shared err {np.round(es,4)}  private err {np.round(ep,4)}  E[(sum h)^2]={cs:.2f} E[sum h^2]={s2:.2f}")
