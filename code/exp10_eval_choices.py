"""E10: sensitivity of the conclusions to the two evaluation choices
(Reviewers 2.4 and 2.5).

Reviewer 2 asked for a justification of the SER threshold, because several
conclusions are reported in terms of SER, and for a sensitivity study on
the deep-fade clip. Both are answered here by re-running the reference
comparison while sweeping the choice in question and reporting whether the
ordering of the schemes and the size of the advantage survive.

Threshold. The SER counts a user as being in error when the cosine of its
recovered embedding falls below tau. The sweep reports every scheme at
several thresholds, together with the ratio between the best unstructured
receiver and the DR, which is the quantity the paper quotes.

Clip. The fading-averaged error laws contain E[1/h_u^2], which diverges
for unclipped Rayleigh fading, so a clip is required for the averages to
exist. The clip applies to the channel realization and therefore affects
every receiver identically. The sweep reports the comparison at several
clip levels.

Outputs: data/e10_threshold_sweep.csv, data/e10_clip_sweep.csv,
data/e10_notes.txt
"""
import math
import os
import numpy as np

from semantic_mac import (affinity_matrix, matched_filter, demux_sr,
                          demux_sc, demux_lmmse, demux_dr,
                          sample_latents_isotropic, oma_observe,
                          demux_noma_genie)
from exp6_sideinfo import demux_block_lmmse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)

from semantic_mac import U_MAIN as U, D_SYN as D, DC_SYN as DC  # Table IV
BATCH = 4000
NAMES = ["OMA", "NOMA", "SR", "SC", "LMMSE", "BLMMSE", "DR"]
TAUS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]  # 0.60, 0.65 added in revision: cover the E9 retrieval transition
CLIPS = [0.05, 0.1, 0.2, 0.3]
UNSTRUCTURED = ("SR", "SC", "LMMSE")


def cosines(beta, rho, rng, h_min, conv_seed=0):
    """Per-user cosines of every scheme at one operating point."""
    a = math.sqrt(beta) * np.ones(U)
    B = affinity_matrix(a)
    z = sample_latents_isotropic(BATCH, U, D, DC, a, rng)
    tilde, h = matched_filter(z, B, rho, rng, h_min=h_min)
    Vc = np.eye(D)[:, :DC]
    lm, _ = demux_lmmse(tilde, B, h, rho)
    rng_c = np.random.default_rng(95000 + conv_seed)
    est = {
        "OMA": oma_observe(z, h, rho, rng_c),
        "NOMA": demux_noma_genie(z, h, rho, rng_c),
        "SR": demux_sr(tilde, B, h),
        "SC": demux_sc(tilde, h),
        "LMMSE": lm,
        "BLMMSE": demux_block_lmmse(tilde, B, h, rho, a, Vc),
        "DR": demux_dr(tilde, B, h, rho, a, Vc),
    }
    out = {}
    for k, e in est.items():
        en = e / (np.linalg.norm(e, axis=2, keepdims=True) + 1e-12)
        out[k] = (en * z).sum(-1).ravel()
    return out


def main():
    rho = 10 ** (10 / 10)
    notes = []

    # ---- threshold sweep over the affinity grid --------------------------
    betas = [0.3, 0.4, 0.5, 0.6, 0.7]
    rng = np.random.default_rng(23)
    cos_by_beta = {b: cosines(b, rho, rng, 0.2, conv_seed=int(100 * b))
                   for b in betas}
    with open(os.path.join(DATA, "e10_threshold_sweep.csv"), "w") as f:
        f.write("tau,beta," + ",".join(f"{n}_ser" for n in NAMES)
                + ",best_unstructured_over_DR\n")
        for tau in TAUS:
            for b in betas:
                c = cos_by_beta[b]
                ser = {n: float((c[n] < tau).mean()) for n in NAMES}
                best_un = min(ser[n] for n in UNSTRUCTURED)
                ratio = best_un / ser["DR"] if ser["DR"] > 0 else float("inf")
                f.write(f"{tau},{b}," + ",".join(f"{ser[n]}" for n in NAMES)
                        + f",{ratio}\n")
                if b == 0.5:
                    notes.append(
                        f"tau={tau:.2f} beta=0.5: LMMSE {ser['LMMSE']:.4f}, "
                        f"BLMMSE {ser['BLMMSE']:.4f}, DR {ser['DR']:.4f}, "
                        f"best unstructured over DR {ratio:.2f}x")

    # ordering check across thresholds
    order_ok = True
    for tau in TAUS:
        for b in betas:
            c = cos_by_beta[b]
            ser = {n: float((c[n] < tau).mean()) for n in NAMES}
            if not (ser["DR"] <= min(ser[n] for n in UNSTRUCTURED) + 1e-12):
                order_ok = False
    notes.append(f"ordering DR at or below every unstructured receiver at "
                 f"every tau in {TAUS} and every beta in {betas}: {order_ok}")

    # ---- clip sweep at the reference operating point ---------------------
    with open(os.path.join(DATA, "e10_clip_sweep.csv"), "w") as f:
        f.write("h_min,beta," + ",".join(f"{n}_ser" for n in NAMES)
                + ",best_unstructured_over_DR,E_inv_h2\n")
        for hm in CLIPS:
            rng_h = np.random.default_rng(77)
            for b in (0.5, 0.7):
                c = cosines(b, rho, rng_h, hm, conv_seed=int(1000 * hm + b))
                ser = {n: float((c[n] < 0.45).mean()) for n in NAMES}
                best_un = min(ser[n] for n in UNSTRUCTURED)
                ratio = best_un / ser["DR"] if ser["DR"] > 0 else float("inf")
                _, h = matched_filter(np.zeros((4000, U, 4)),
                                      np.eye(U), rho,
                                      np.random.default_rng(5), h_min=hm)
                einv = float(np.mean(1.0 / h ** 2))
                f.write(f"{hm},{b}," + ",".join(f"{ser[n]}" for n in NAMES)
                        + f",{ratio},{einv}\n")
                if b == 0.5:
                    notes.append(
                        f"h_min={hm}: E[1/h^2]={einv:.3f}, beta=0.5 "
                        f"LMMSE {ser['LMMSE']:.4f}, DR {ser['DR']:.4f}, "
                        f"best unstructured over DR {ratio:.2f}x")

    with open(os.path.join(DATA, "e10_notes.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")
    print("\n".join(notes))
    print("E10 done.")


if __name__ == "__main__":
    main()
