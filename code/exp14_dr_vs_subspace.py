"""E14: where the decomposition receiver is preferable to the exact
DR-MMSE.  (CSV key: Sub)

DR-MMSE of Remark 1 is the minimum mean-square-error
linear receiver with the same side information, so the DR cannot beat it
in mean-square error and no such claim is made. The question this
experiment answers is where the DR is nevertheless the better choice.

Two candidate axes are measured.

(a) Task level. The DR cancels on the complement with zero forcing,
    which is unbiased, while the exact form shrinks toward the prior. A
    bias lowers the mean-square error and costs discriminability, so the
    DR is expected to retrieve better. E9 saw this at one operating
    point, and here it is swept over signal-to-noise ratio and affinity.

(b) Basis error. The exact private filter is built from the prior
    covariance diag(1-a_u^2)/(d-d_c), which assumes the complement holds
    private content only. An imperfect basis leaks shared content into
    the complement and therefore misspecifies that prior, whereas zero
    forcing inverts the coupling whatever the content is. The two are
    compared with the spectral basis recovered from N calibration pairs,
    which is what a deployment actually holds, and with the true basis.

Outputs: data/e14_dr_vs_subspace.csv, data/e14_notes.txt
"""
import math
import os
import numpy as np
import torch

from semantic_mac import (EmbeddingPool, affinity_matrix, matched_filter,
                          demux_dr, random_orthogonal,
                          learn_structure_spectral, subspace_error, metrics)
from exp6_sideinfo import demux_block_lmmse
from exp9_retrieval import sample_with_index, top1

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)

POOL_PT = os.path.join(HERE, "..", "data", "bert_agnews_8000.pt")
if not os.path.exists(POOL_PT):
    POOL_PT = os.path.join(HERE, "..", "..", "5. WCL-DRL",
                           "bert_agnews_8000.pt")

from semantic_mac import U_MAIN as U, D_REAL as D, DC_REAL as DC  # Table IV
N_PAIR = 200
BATCH_EVAL = 4000
N_TRAIN_POOL = 7000
IDX_TRAIN = np.arange(N_TRAIN_POOL)
IDX_EVAL = np.arange(N_TRAIN_POOL, 8000)


def main():
    rng = np.random.default_rng(7)
    obj = torch.load(POOL_PT, map_location="cpu", weights_only=False)
    X = obj.numpy() if torch.is_tensor(obj) else np.asarray(obj)
    pool = EmbeddingPool(X)
    R = random_orthogonal(D, rng)
    V_true = R[:, :DC]
    bank_p = pool.pca_coords(IDX_EVAL, D - DC)
    pos_of = {g: i for i, g in enumerate(IDX_EVAL)}

    rows, notes = [], []
    for beta in (0.3, 0.5, 0.7):
        a = math.sqrt(beta) * np.ones(U)
        B = affinity_matrix(a)
        z_cal, _ = sample_with_index(pool, N_PAIR, a, rng, IDX_TRAIN)
        V_spec, _ = learn_structure_spectral(z_cal @ R.T, DC)
        err = subspace_error(V_spec, V_true)
        for snr_db in (10, 15, 20):
            rho = 10 ** (snr_db / 10)
            rng_e = np.random.default_rng(1000 + int(100 * beta) + snr_db)
            z, idx = sample_with_index(pool, BATCH_EVAL, a, rng_e, IDX_EVAL)
            x = z @ R.T
            tilde, h = matched_filter(x, B, rho, rng_e)
            for bname, Vc in (("true", V_true), ("spectral", V_spec)):
                est = {"DR": demux_dr(tilde, B, h, rho, a, Vc),
                       "Subspace": demux_block_lmmse(tilde, B, h, rho, a, Vc)}
                out = {}
                for k, e in est.items():
                    cos, _, ser = metrics(e, x)
                    zhat = e @ R
                    hp = []
                    for u in range(U):
                        truth = np.array([pos_of[g] for g in idx[:, u + 1]])
                        hp.append(top1(zhat[:, u, DC:], bank_p, truth))
                    out[k] = (cos, ser, float(np.mean(np.concatenate(hp))))
                rows.append((beta, snr_db, bname, err,
                             *out["DR"], *out["Subspace"]))
                d, s = out["DR"], out["Subspace"]
                notes.append(
                    f"beta={beta} {snr_db:2d}dB basis={bname:8s} | "
                    f"SER DR {d[1]:.4f} vs sub {s[1]:.4f} "
                    f"({d[1]-s[1]:+.4f}) | private top-1 DR {d[2]:.4f} "
                    f"vs sub {s[2]:.4f} ({d[2]-s[2]:+.4f})")
                print(notes[-1])

    with open(os.path.join(DATA, "e14_dr_vs_subspace.csv"), "w") as f:
        f.write("beta,snr,basis,subspace_err,DR_cos,DR_ser,DR_top1,"
                "Sub_cos,Sub_ser,Sub_top1\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    with open(os.path.join(DATA, "e14_notes.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")
    print("E14 done.")


if __name__ == "__main__":
    main()
