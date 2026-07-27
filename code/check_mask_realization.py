"""Mask-realization check: the paper's experiments generate the
matched-filter outputs from the EXPECTED cross-Gram model
(semantic_mac.matched_filter). This script draws actual Haar-mixture
masks, passes the physical superposition r = sum_v h_v M_v x_v + n
through the realized masks, applies the same receivers, and compares
the SER against the expectation-model front end on identical latents,
gains, and noise seeds.

The realized per-entry Gram deviation is O(1/sqrt(d)); this check
quantifies its end-to-end effect at d=64 (theory/mobility setting) and
d=768 (real-embedding setting).

Outputs: data/e_mask_check.csv
"""
import math
import os
import numpy as np

from semantic_mac import (affinity_matrix, matched_filter, demux_sr,
                          demux_sc, demux_lmmse, demux_dr,
                          sample_latents_isotropic, random_orthogonal,
                          metrics)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

U = 4
BETA = 0.5


def haar_masks(B, d, rng):
    """M_u = sum_k A_uk U_k with A = chol(B), U_k independent Haar."""
    A = np.linalg.cholesky(B)
    Us = [random_orthogonal(d, rng) for _ in range(U)]
    return [sum(A[u, k] * Us[k] for k in range(U)) for u in range(U)]


def physical_front_end(z, h, Ms, rho, rng):
    """r = sum_v h_v M_v z_v + n (ambient AWGN), tilde_u = M_u^T r / h_u."""
    batch, Uu, d = z.shape
    sigma = math.sqrt(1.0 / rho)
    r = np.einsum('bv,vde,bve->bd', h,
                  np.stack(Ms), z) + rng.standard_normal((batch, d)) * sigma
    tilde = np.stack([(r @ Ms[u]) / h[:, u][:, None] for u in range(U)],
                     axis=1)
    return tilde


def run(d, d_c, batch, n_mask_draws, snr_db):
    RHO = 10 ** (snr_db / 10)
    a = math.sqrt(BETA) * np.ones(U)
    B = affinity_matrix(a)
    Vc = np.eye(d)[:, :d_c]
    diffs = {m: [] for m in ("SR", "SC", "LMMSE", "DR")}
    sers_r = {m: [] for m in diffs}
    sers_e = {m: [] for m in diffs}
    for t in range(n_mask_draws):
        rng = np.random.default_rng(7700 + 10 * t)
        z = sample_latents_isotropic(batch, U, d, d_c, a, rng)
        hc = (rng.standard_normal((batch, U)) +
              1j * rng.standard_normal((batch, U))) / math.sqrt(2)
        h = np.clip(np.abs(hc), 0.2, None)
        Ms = haar_masks(B, d, rng)
        gram_dev = max(np.abs(Ms[u].T @ Ms[v] - B[u, v] * np.eye(d)).max()
                       for u in range(U) for v in range(U))
        tilde_r = physical_front_end(z, h, Ms, RHO, rng)
        # expectation-model front end on the SAME latents and gains
        rng_e = np.random.default_rng(8800 + 10 * t)
        tilde_e = np.einsum('uv,bvd,bv,bu->bud', B, z, h, 1.0 / h)
        xi = rng_e.standard_normal((batch, U, d)) * math.sqrt(1.0 / RHO)
        A = np.linalg.cholesky(B)
        for b in range(batch):
            tilde_e[b] += (A @ xi[b]) / h[b][:, None]
        for name, tl in (("realized", tilde_r), ("expected", tilde_e)):
            out = {}
            out["SR"] = demux_sr(tl, B, h)
            out["SC"] = demux_sc(tl, h)
            out["LMMSE"], _ = demux_lmmse(tl, B, h, RHO)
            out["DR"] = demux_dr(tl, B, h, RHO, a, Vc)
            for m in diffs:
                _, _, s = metrics(out[m], z)
                (sers_r if name == "realized" else sers_e)[m].append(s)
        for m in diffs:
            diffs[m].append(sers_r[m][-1] - sers_e[m][-1])
        print(f"d={d} draw {t+1}/{n_mask_draws} gram_dev={gram_dev:.3f} "
              + " ".join(f"{m}:d={diffs[m][-1]:+.4f}" for m in diffs))
    return {m: (float(np.mean(sers_r[m])), float(np.mean(sers_e[m])),
                float(np.mean(diffs[m])), float(np.std(diffs[m])))
            for m in diffs}


def main():
    rows = []
    for d, d_c, batch, draws, snr_db in ((64, 16, 2000, 12, 10),
                                         (768, 128, 400, 6, 20)):
        res = run(d, d_c, batch, draws, snr_db)
        for m, (sr, se, md, sd) in res.items():
            rows.append((d, snr_db, m, sr, se, md, sd))
            print(f"d={d} snr={snr_db} {m}: realized={sr:.4f} "
                  f"expected={se:.4f} mean_diff={md:+.5f} std={sd:.5f}")
    with open(os.path.join(DATA, "e_mask_check.csv"), "w") as f:
        f.write("d,snr,method,ser_realized,ser_expected,mean_diff,"
                "std_diff\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print("mask check done.")


if __name__ == "__main__":
    main()
