"""E8: partial sharing, when the shared coordinates carry user-specific
residual content (Reviewer 1.3).

Reviewer 1 observed that a shared subspace fixed at calibration does not
guarantee that its content is common on every online sample. This
experiment relaxes exactly that idealization. The shared block of user u
becomes

    a_u ( sqrt(1-kappa) c + sqrt(kappa) r_u ),

where c is the common content and r_u is drawn independently per user, so
kappa is the fraction of shared-block energy that is NOT common. At
kappa = 0 the model of Section III is recovered, and at kappa = 1 nothing
in the shared block is common.

Two effects follow and the experiment separates them. The pairwise
affinity falls to a_u a_v (1-kappa), which the receiver measures and uses,
so the receiver is not mismatched. Beyond that, the residual r_u sits
inside the subspace on which the decomposition receiver combines, and
combining averages it away, so a part of each user's content becomes
unrecoverable by any receiver that combines there. The genie-aided
NOMA-SIC reference, which never combines, is included precisely because
it does recover the residual and therefore bounds what is lost.

Outputs: data/e8_partial_sharing.csv, data/e8_notes.txt
"""
import math
import os
import numpy as np

from semantic_mac import (affinity_matrix, matched_filter, demux_sr,
                          demux_sc, demux_lmmse, demux_dr, metrics,
                          oma_observe, demux_noma_genie)
from exp6_sideinfo import demux_block_lmmse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)

from semantic_mac import U_MAIN as U, D_SYN as D, DC_SYN as DC  # Table IV
BATCH = 4000
NAMES = ["OMA", "NOMA", "SR", "SC", "LMMSE", "BLMMSE", "DR"]


def sample_latents_partial(batch, U, d, d_c, a, kappa, rng):
    """Structured latents whose shared block is only partially common.

    shared_u = a_u ( sqrt(1-kappa) c + sqrt(kappa) r_u ), r_u independent.
    The private complement is unchanged. Every embedding stays unit norm
    because c and r_u are unit norm in the shared block and the two
    weights are the legs of a right triangle."""
    a = np.broadcast_to(np.asarray(a, float), (U,))
    z = np.zeros((batch, U, d))
    c = rng.standard_normal((batch, d_c))
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    r = rng.standard_normal((batch, U, d_c))
    r /= np.linalg.norm(r, axis=2, keepdims=True)
    p = rng.standard_normal((batch, U, d - d_c))
    p /= np.linalg.norm(p, axis=2, keepdims=True)
    s = (math.sqrt(1.0 - kappa) * c[:, None, :]
         + math.sqrt(kappa) * r)
    s /= np.linalg.norm(s, axis=2, keepdims=True)     # exact unit shared block
    z[:, :, :d_c] = a[None, :, None] * s
    z[:, :, d_c:] = np.sqrt(1.0 - a ** 2)[None, :, None] * p
    return z


def run_point(beta_nominal, kappa, rho, rng, conv_seed=0):
    """beta_nominal is a_u a_v before contamination; the effective affinity
    that the receiver measures and uses is a_u a_v (1-kappa)."""
    a = math.sqrt(beta_nominal) * np.ones(U)
    z = sample_latents_partial(BATCH, U, D, DC, a, kappa, rng)
    # effective share coefficients seen through the common component only
    a_eff = a * math.sqrt(1.0 - kappa)
    B_eff = affinity_matrix(a_eff)
    tilde, h = matched_filter(z, B_eff, rho, rng)
    Vc = np.eye(D)[:, :DC]
    res = {}
    res["SR"] = metrics(demux_sr(tilde, B_eff, h), z)
    res["SC"] = metrics(demux_sc(tilde, h), z)
    lm, _ = demux_lmmse(tilde, B_eff, h, rho)
    res["LMMSE"] = metrics(lm, z)
    res["BLMMSE"] = metrics(
        demux_block_lmmse(tilde, B_eff, h, rho, a_eff, Vc), z)
    res["DR"] = metrics(demux_dr(tilde, B_eff, h, rho, a_eff, Vc), z)
    rng_c = np.random.default_rng(80000 + conv_seed)
    res["OMA"] = metrics(oma_observe(z, h, rho, rng_c), z)
    res["NOMA"] = metrics(demux_noma_genie(z, h, rho, rng_c), z)
    return res


def main():
    rng = np.random.default_rng(11)
    rho = 10 ** (10 / 10)
    kappas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    notes = []
    all_rows = {}
    for beta_nom in (0.5, 0.8):
        rows = []
        for k in kappas:
            r = run_point(beta_nom, k, rho, rng,
                          conv_seed=int(1000 * beta_nom + 100 * k))
            rows.append(r)
            print(f"beta_nom={beta_nom} kappa={k:4.2f} " + " ".join(
                f"{n}:ser={r[n][2]:.4f}" for n in ("LMMSE", "BLMMSE", "DR", "NOMA")))
        all_rows[beta_nom] = rows
        base = rows[0]["DR"][2]
        # smallest kappa at which the DR no longer beats the best unstructured
        cross = None
        for k, r in zip(kappas, rows):
            best_unstruct = min(r["LMMSE"][2], r["SC"][2], r["SR"][2])
            if r["DR"][2] >= best_unstruct and cross is None:
                cross = k
        notes.append(
            f"beta_nominal={beta_nom}: DR SER {base:.4f} at kappa=0, "
            f"{rows[2]['DR'][2]:.4f} at kappa=0.1, {rows[4]['DR'][2]:.4f} at "
            f"kappa=0.3; DR stops beating every unstructured receiver at "
            f"kappa={cross}")
        notes.append(f"  kappa, LMMSE, BLMMSE, DR, NOMA-SIC (SER):")
        for k, r in zip(kappas, rows):
            notes.append(f"    {k:4.2f}  {r['LMMSE'][2]:.4f}  "
                         f"{r['BLMMSE'][2]:.4f}  {r['DR'][2]:.4f}  "
                         f"{r['NOMA'][2]:.4f}")

    with open(os.path.join(DATA, "e8_partial_sharing.csv"), "w") as f:
        f.write("beta_nominal,kappa," + ",".join(
            f"{n}_cos,{n}_nmse,{n}_ser" for n in NAMES) + "\n")
        for beta_nom, rows in all_rows.items():
            for k, r in zip(kappas, rows):
                f.write(f"{beta_nom},{k}," + ",".join(
                    f"{r[n][0]},{r[n][1]},{r[n][2]}" for n in NAMES) + "\n")
    with open(os.path.join(DATA, "e8_notes.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")
    print("\n".join(notes))
    print("E8 done.")


if __name__ == "__main__":
    main()
