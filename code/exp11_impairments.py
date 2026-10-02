"""E11: robustness under practical channel conditions (Reviewer 4.2).

The analysis assumes coherent reception with known channel magnitudes and
symbol-level synchronization. Reviewer 4 asked how the framework behaves
when those idealizations are relaxed. Three impairments are studied, each
applied to the channel or to the receiver's knowledge of it, so that the
comparison between receivers stays matched.

(a) Residual phase error. After phase compensation a residual offset
    phi_u remains, so the in-phase projection retains cos(phi_u) of the
    signal while the receiver still assumes the nominal gain. The lost
    quadrature energy is not recoverable by any of the receivers.

(b) Channel estimation error. The receiver is given h_hat = h (1 + eps)
    with eps zero-mean Gaussian of standard deviation sigma_e, and builds
    every matrix from h_hat while the channel realizes h.

(c) Power imbalance. The users are assigned unequal average gains spanning
    an imbalance ratio. Per-user error rates are reported as well as the
    average, because a scheme that combines shared content could in
    principle concentrate the benefit on the strongest user.

Outputs: data/e11_phase.csv, data/e11_cest.csv, data/e11_imbalance.csv,
data/e11_notes.txt
"""
import math
import os
import numpy as np

from semantic_mac import (affinity_matrix, matched_filter, demux_sr,
                          demux_sc, demux_lmmse, demux_dr, metrics,
                          sample_latents_isotropic, oma_observe,
                          demux_noma_genie, TAU)
from exp6_sideinfo import demux_block_lmmse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)

from semantic_mac import U_MAIN as U, D_SYN as D, DC_SYN as DC  # Table IV
BATCH = 4000
BETA = 0.5
SNR_DB = 10
NAMES = ["OMA", "NOMA", "SR", "SC", "LMMSE", "BLMMSE", "DR"]


def receive(z, B, h_true, h_assumed, rho, a, rng, conv_seed):
    """Front end realizes h_true; every receiver is built from h_assumed."""
    Vc = np.eye(D)[:, :DC]
    tilde, _ = matched_filter(z, B, rho, rng, h_fixed=h_true)
    lm, _ = demux_lmmse(tilde, B, h_assumed, rho)
    rng_c = np.random.default_rng(96000 + conv_seed)
    return {
        "OMA": oma_observe(z, h_true, rho, rng_c),
        "NOMA": demux_noma_genie(z, h_true, rho, rng_c),
        "SR": demux_sr(tilde, B, h_assumed),
        "SC": demux_sc(tilde, h_assumed),
        "LMMSE": lm,
        "BLMMSE": demux_block_lmmse(tilde, B, h_assumed, rho, a, Vc),
        "DR": demux_dr(tilde, B, h_assumed, rho, a, Vc),
    }


def rayleigh(batch, rng, h_min=0.2, scale=None):
    hc = (rng.standard_normal((batch, U)) +
          1j * rng.standard_normal((batch, U))) / math.sqrt(2)
    h = np.clip(np.abs(hc), h_min, None)
    if scale is not None:
        h = h * np.asarray(scale, float)[None, :]
    return h


def per_user_ser(e_hat, z, tau=TAU):
    en = e_hat / (np.linalg.norm(e_hat, axis=2, keepdims=True) + 1e-12)
    cos = (en * z).sum(-1)
    return (cos < tau).mean(axis=0)


def write_rows(path, key, xs, rows):
    with open(path, "w") as f:
        f.write(key + "," + ",".join(f"{n}_cos,{n}_ser" for n in NAMES) + "\n")
        for x, r in zip(xs, rows):
            f.write(f"{x}," + ",".join(
                f"{r[n][0]},{r[n][2]}" for n in NAMES) + "\n")


def main():
    rho = 10 ** (SNR_DB / 10)
    a = math.sqrt(BETA) * np.ones(U)
    B = affinity_matrix(a)
    notes = [f"operating point beta={BETA}, SNR={SNR_DB} dB, U={U}, d={D}"]

    # ---- (a) residual phase error ---------------------------------------
    phis = [0, 5, 10, 20, 30]
    rows = []
    for deg in phis:
        rng = np.random.default_rng(101 + deg)
        z = sample_latents_isotropic(BATCH, U, D, DC, a, rng)
        h = rayleigh(BATCH, rng)
        phi = rng.uniform(-math.radians(deg), math.radians(deg),
                          size=(BATCH, U))
        est = receive(z, B, h * np.cos(phi), h, rho, a, rng,
                      conv_seed=deg)
        r = {n: metrics(e, z) for n, e in est.items()}
        rows.append(r)
        notes.append(f"phase +-{deg:2d} deg: LMMSE ser={r['LMMSE'][2]:.4f}, "
                     f"DR ser={r['DR'][2]:.4f}, "
                     f"DR cos={r['DR'][0]:.4f}")
        print(notes[-1])
    write_rows(os.path.join(DATA, "e11_phase.csv"), "phase_deg", phis, rows)

    # ---- (b) channel estimation error -----------------------------------
    sigmas = [0.0, 0.05, 0.1, 0.2, 0.3]
    rows = []
    for se in sigmas:
        rng = np.random.default_rng(202 + int(100 * se))
        z = sample_latents_isotropic(BATCH, U, D, DC, a, rng)
        h = rayleigh(BATCH, rng)
        h_hat = np.clip(h * (1.0 + se * rng.standard_normal(h.shape)), 0.05,
                        None)
        est = receive(z, B, h, h_hat, rho, a, rng, conv_seed=int(100 * se))
        r = {n: metrics(e, z) for n, e in est.items()}
        rows.append(r)
        notes.append(f"ch.est sigma_e={se:.2f}: LMMSE ser={r['LMMSE'][2]:.4f}, "
                     f"DR ser={r['DR'][2]:.4f}")
        print(notes[-1])
    write_rows(os.path.join(DATA, "e11_cest.csv"), "sigma_e", sigmas, rows)

    # ---- (c) power imbalance --------------------------------------------
    ratios_db = [0, 3, 6, 10, 15]
    rows = []
    with open(os.path.join(DATA, "e11_imbalance.csv"), "w") as f:
        f.write("ratio_db," + ",".join(f"{n}_cos,{n}_ser" for n in NAMES)
                + "," + ",".join(f"DR_ser_u{u}" for u in range(U))
                + "," + ",".join(f"LMMSE_ser_u{u}" for u in range(U)) + "\n")
        for rdb in ratios_db:
            rng = np.random.default_rng(303 + rdb)
            # gains spaced geometrically over the imbalance range, unit mean
            g = np.logspace(-rdb / 40, rdb / 40, U)
            g = g / np.sqrt((g ** 2).mean())
            z = sample_latents_isotropic(BATCH, U, D, DC, a, rng)
            h = rayleigh(BATCH, rng, scale=g)
            est = receive(z, B, h, h, rho, a, rng, conv_seed=rdb)
            r = {n: metrics(e, z) for n, e in est.items()}
            rows.append(r)
            pu = per_user_ser(est["DR"], z)
            pl = per_user_ser(est["LMMSE"], z)
            f.write(f"{rdb}," + ",".join(f"{r[n][0]},{r[n][2]}" for n in NAMES)
                    + "," + ",".join(f"{v}" for v in pu)
                    + "," + ",".join(f"{v}" for v in pl) + "\n")
            notes.append(
                f"imbalance {rdb:2d} dB: LMMSE ser={r['LMMSE'][2]:.4f}, "
                f"DR ser={r['DR'][2]:.4f}, DR per-user "
                f"[{', '.join(f'{v:.3f}' for v in pu)}] "
                f"(spread {pu.max()-pu.min():+.3f}); LMMSE spread "
                f"{pl.max()-pl.min():+.3f}")
            print(notes[-1])

    with open(os.path.join(DATA, "e11_notes.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")
    print("E11 done.")


if __name__ == "__main__":
    main()
