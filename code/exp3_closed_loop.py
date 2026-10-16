"""E3 closed loop: transmit masks built from the fed-back estimate.

Experiment E3 simulates the front end at the true affinity, so tracking
error enters the receiver alone. Algorithm 1 builds the transmit masks from
the fed-back estimate instead. This script runs the E3 scene traversal both
ways on the same front-end draws (fading and noise) and reports the SER of
the DR given the true coefficients (genie), the open loop of E3 (open), the
closed loop with the masks from the current estimate (closed_same) and
from the estimate fed back one slot earlier (closed_delay). Five runs,
seeds 3 to 7 as in E3; the random stream differs from exp3_mobility.py
because the front end is drawn here explicitly.

Output: stdout, saved as data/e3_closed_loop.txt (Comment 1.2 of the review)."""
import os, sys, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp3_mobility as e3
from semantic_mac import (affinity_matrix, demux_dr, sample_latents_isotropic, AffinityTracker,
                          pilot_affinity_obs, metrics, H_MIN)
U, D, DC, BATCH, RHO, K, LAM = e3.U, e3.D, e3.DC, e3.BATCH, e3.RHO, e3.K_PILOT, e3.LAM
Vc = np.eye(D)[:, :DC]
def front(z, Bm, h, xi):
    t = np.einsum('uv,bvd,bv,bu->bud', Bm, z, h, 1.0 / h)
    A = np.linalg.cholesky(Bm)
    return t + np.einsum('uv,bvd->bud', A, xi) / h[:, :, None]
def run(a_t, rng):
    a_bar = a_t.mean(0); tr = AffinityTracker(U, lam=LAM, a_init=float(a_bar.mean()))
    a_prev = np.clip(tr.a, 0.02, 0.95)
    s = {k: [] for k in ("genie", "open", "closed_same", "closed_delay")}
    for t in range(a_t.shape[0]):
        a = a_t[t]; B = affinity_matrix(a)
        z = sample_latents_isotropic(BATCH, U, D, DC, a, rng)
        hc = (rng.standard_normal((BATCH, U)) + 1j * rng.standard_normal((BATCH, U))) / math.sqrt(2)
        h = np.clip(np.abs(hc), H_MIN, None)
        xi = rng.standard_normal((BATCH, U, D)) * math.sqrt(1.0 / RHO)
        if t % K == 0:
            zp = sample_latents_isotropic(e3.N_PILOT, U, D, DC, a, rng)
            hp = np.clip(np.abs((rng.standard_normal(U) + 1j * rng.standard_normal(U)) / np.sqrt(2)), 0.2, None)
            tr.update(pilot_affinity_obs(zp, hp, RHO, rng))
        a_hat = np.clip(tr.a, 0.02, 0.95); Bh = affinity_matrix(a_hat); Bp = affinity_matrix(a_prev)
        t_open = front(z, B, h, xi)
        s["genie"].append(metrics(demux_dr(t_open, B, h, RHO, a, Vc), z)[2])
        s["open"].append(metrics(demux_dr(t_open, Bh, h, RHO, a_hat, Vc), z)[2])
        t_cl = front(z, Bh, h, xi)
        s["closed_same"].append(metrics(demux_dr(t_cl, Bh, h, RHO, a_hat, Vc), z)[2])
        t_cd = front(z, Bp, h, xi)   # masks from the estimate fed back one slot earlier
        s["closed_delay"].append(metrics(demux_dr(t_cd, Bp, h, RHO, a_hat, Vc), z)[2])
        a_prev = a_hat
    return {k: float(np.mean(v)) for k, v in s.items()}
a_ts = e3.drive_through_profile(e3.T)
res = [run(a_ts, np.random.default_rng(3 + r)) for r in range(5)]
for k in res[0]:
    v = np.array([x[k] for x in res]); print(f"{k:13s} per run {np.round(v,4)} mean {v.mean():.4f}")
