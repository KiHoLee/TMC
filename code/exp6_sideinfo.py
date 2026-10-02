"""E6: receiver comparison at matched side information (Reviewer 1.1).

Reviewer 1 observed that the LMMSE receiver of Proposition 1 is restricted
to filters of the form (F (x) I_d), while the DR additionally uses the
shared basis V_c and the share coefficients {a_u}, so the reported gain
conflates the value of the side information with the value of the
decomposition. This experiment separates the two by adding a
subspace-aware LMMSE receiver that is given exactly the side information
the DR receives.

Structure of the added receiver
-------------------------------
The noise is isotropic across the d coordinates, so the estimation problem
separates over the shared subspace V_c and its orthogonal complement.

  Shared block.   V_c^T tilde_x_u = gamma_u c + noise, with
                  gamma_u = a_u + sum_{v!=u} beta_uv a_v h_v/h_u.
                  One unknown vector c is observed U times, so the MMSE
                  estimate under the prior E[c c^T] = I/d_c is
                  c_hat = gamma^T Cn^-1 Zs / (gamma^T Cn^-1 gamma + d_c).

  Private block.  Zp = Gamma Xp + noise with Xp the stacked private parts,
                  whose prior covariance is diagonal, Cp = diag(1-a_u^2)/d_p.
                  The MMSE estimate is the joint filter
                  Wp = Cp Gamma^T (Gamma Cp Gamma^T + Cn)^-1.

The DR uses the same shared branch but replaces Wp by zero forcing
followed by a per-user scalar Wiener shrinkage, so any gap between the two
curves is produced by the private branch alone and vanishes as the noise
falls, where both forms approach Gamma^-1.

Outputs: data/e6_sideinfo.csv (beta sweep), data/e6_sideinfo_snr.csv
(SNR sweep), data/e6_gap.txt (summary of the largest gaps).
"""
import math
import os
import numpy as np

from semantic_mac import (affinity_matrix, matched_filter, demux_sr,
                          demux_sc, demux_lmmse, demux_dr,
                          sample_latents_isotropic, metrics,
                          oma_observe, demux_noma_genie)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)

from semantic_mac import U_MAIN as U, D_SYN as D, DC_SYN as DC  # Table IV
BATCH = 4000
NAMES = ["OMA", "NOMA", "SR", "SC", "LMMSE", "BLMMSE", "DR"]


def demux_block_lmmse(tilde, B, h, rho, a, Vc):
    """Subspace-aware LMMSE given (V_c, {a_u}, B, h, rho).

    Exactly the side information the DR receives. Separates over the
    shared subspace and its complement, which is optimal here because the
    shared and private contents are uncorrelated and the noise is
    isotropic across coordinates."""
    batch, U_, d = tilde.shape
    d_c = Vc.shape[1]
    d_p = d - d_c
    sigma2 = 1.0 / rho
    a = np.broadcast_to(np.asarray(a, float), (U_,))
    out = np.empty_like(tilde)
    Zs = tilde @ Vc                       # (batch, U, d_c)
    Zp = tilde - (Zs @ Vc.T)              # complement, ambient frame
    Cp = np.diag((1.0 - a ** 2) / d_p)    # per-dimension private prior
    for b in range(batch):
        hb = h[b]
        Hi = np.diag(1.0 / hb)
        Gamma = Hi @ B @ np.diag(hb)
        Cn = sigma2 * (Hi @ B @ Hi)
        Cn_inv = np.linalg.inv(Cn)
        gamma = np.array([
            a[u] + sum(B[u, v] * a[v] * hb[v] / hb[u]
                       for v in range(U_) if v != u) for u in range(U_)])
        denom = float(gamma @ Cn_inv @ gamma)
        c_hat = (gamma @ Cn_inv @ Zs[b]) / (denom + d_c)
        S = Gamma @ Cp @ Gamma.T + Cn
        Wp = np.linalg.solve(S.T, (Cp @ Gamma.T).T).T
        Xp = Wp @ Zp[b]
        for u in range(U_):
            out[b, u] = a[u] * (Vc @ c_hat) + Xp[u]
    return out


def run_point(beta, rho, rng, conv_seed=0):
    a = math.sqrt(beta) * np.ones(U)
    B = affinity_matrix(a)
    z = sample_latents_isotropic(BATCH, U, D, DC, a, rng)
    tilde, h = matched_filter(z, B, rho, rng)
    Vc = np.eye(D)[:, :DC]
    res = {}
    res["SR"] = metrics(demux_sr(tilde, B, h), z)
    res["SC"] = metrics(demux_sc(tilde, h), z)
    lm, _ = demux_lmmse(tilde, B, h, rho)
    res["LMMSE"] = metrics(lm, z)
    res["BLMMSE"] = metrics(demux_block_lmmse(tilde, B, h, rho, a, Vc), z)
    res["DR"] = metrics(demux_dr(tilde, B, h, rho, a, Vc), z)
    rng_c = np.random.default_rng(90000 + conv_seed)
    res["OMA"] = metrics(oma_observe(z, h, rho, rng_c), z)
    res["NOMA"] = metrics(demux_noma_genie(z, h, rho, rng_c), z)
    return res


def write_csv(path, key, xs, rows):
    with open(path, "w") as f:
        f.write(key + "," + ",".join(f"{n}_cos,{n}_nmse,{n}_ser"
                                     for n in NAMES) + "\n")
        for x, r in zip(xs, rows):
            f.write(f"{x}," + ",".join(
                f"{r[n][0]},{r[n][1]},{r[n][2]}" for n in NAMES) + "\n")


def main():
    rng = np.random.default_rng(0)
    notes = []

    # ---- affinity sweep at the reference operating point ----------------
    rho = 10 ** (10 / 10)
    betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    rows = []
    for b in betas:
        r = run_point(b, rho, rng, conv_seed=int(b * 100))
        rows.append(r)
        print(f"beta={b:4.2f} " + " ".join(
            f"{n}:ser={r[n][2]:.4f}" for n in ("LMMSE", "BLMMSE", "DR")))
    write_csv(os.path.join(DATA, "e6_sideinfo.csv"), "beta", betas, rows)

    # largest DR-to-BLMMSE gap over the affinity sweep
    gaps = [(b, r["DR"][2] - r["BLMMSE"][2], r["DR"][0] - r["BLMMSE"][0])
            for b, r in zip(betas, rows)]
    wb, wser, wcos = max(gaps, key=lambda t: abs(t[1]))
    notes.append(f"beta sweep at 10 dB: max |SER gap| DR-BLMMSE = {wser:+.5f} "
                 f"at beta={wb} (cosine gap {wcos:+.5f})")
    notes.append("beta sweep per-point SER (beta, LMMSE, BLMMSE, DR):")
    for b, r in zip(betas, rows):
        notes.append(f"  {b:4.2f}  {r['LMMSE'][2]:.5f}  "
                     f"{r['BLMMSE'][2]:.5f}  {r['DR'][2]:.5f}")

    # ---- SNR sweep at intermediate affinity -----------------------------
    snrs = list(range(0, 21, 4))
    rows_s = []
    for s in snrs:
        r = run_point(0.4, 10 ** (s / 10), rng, conv_seed=1000 + s)
        rows_s.append(r)
        print(f"snr={s:2d} " + " ".join(
            f"{n}:ser={r[n][2]:.4f}" for n in ("LMMSE", "BLMMSE", "DR")))
    write_csv(os.path.join(DATA, "e6_sideinfo_snr.csv"), "snr", snrs, rows_s)

    gaps_s = [(s, r["DR"][2] - r["BLMMSE"][2]) for s, r in zip(snrs, rows_s)]
    ws, wsg = max(gaps_s, key=lambda t: abs(t[1]))
    notes.append(f"SNR sweep at beta=0.4: max |SER gap| DR-BLMMSE = "
                 f"{wsg:+.5f} at {ws} dB")
    notes.append("SNR sweep per-point SER (dB, LMMSE, BLMMSE, DR):")
    for s, r in zip(snrs, rows_s):
        notes.append(f"  {s:2d}  {r['LMMSE'][2]:.5f}  "
                     f"{r['BLMMSE'][2]:.5f}  {r['DR'][2]:.5f}")

    with open(os.path.join(DATA, "e6_gap.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")
    print("\n".join(notes[:2]))
    print("E6 done.")


if __name__ == "__main__":
    main()
