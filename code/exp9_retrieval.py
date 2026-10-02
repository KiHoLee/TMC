"""E9: task-oriented retrieval metric and the justification of the SER
threshold (Reviewers 1.3, 2.4, and 3.3).

Reviewer 1 asked for a downstream task metric rather than a
cosine-threshold error rate alone, and Reviewer 2 asked why the threshold
takes the value it does. Both are answered by the same measurement.

Task. Each transmitted embedding is built from real BERT AG-News
sentences, one carrying the shared scene content and one carrying each
user's private content. After reception, the recovered embedding is
mapped back to the latent frame and its two blocks are matched against
the held-out sentence pool by cosine. Top-one retrieval asks whether the
correct sentence is returned, for the shared content and for the private
content separately. The chance level is one over the size of the held-out
pool, and the complement of the accuracy is a symbol-level error rate
that involves no threshold at all.

Threshold. The same run records, for every user and every scheme, the
cosine of the recovered embedding together with whether its private
content was retrieved correctly. Binning the retrieval accuracy against
the cosine locates the value at which retrieval stops being reliable,
which is the operational meaning the threshold should carry.

Outputs: data/e9_retrieval.csv, data/e9_threshold.csv, data/e9_notes.txt
"""
import math
import os
import numpy as np
import torch

from semantic_mac import (EmbeddingPool, affinity_matrix, matched_filter,
                          demux_lmmse, demux_dr, random_orthogonal,
                          learn_structure_spectral, subspace_error, metrics,
                          oma_observe, demux_noma_genie)
from exp6_sideinfo import demux_block_lmmse
from exp2_structure import train_adapter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)

POOL_PT = os.path.join(HERE, "..", "data", "bert_agnews_8000.pt")
if not os.path.exists(POOL_PT):
    POOL_PT = os.path.join(HERE, "..", "..", "5. WCL-DRL",
                           "bert_agnews_8000.pt")

from semantic_mac import U_MAIN as U, D_REAL as D, DC_REAL as DC  # Table IV
BETA = 0.5
N_PAIR = 200
BATCH_EVAL = 4000
N_TRAIN_POOL = 7000
IDX_TRAIN = np.arange(N_TRAIN_POOL)
IDX_EVAL = np.arange(N_TRAIN_POOL, 8000)
SNR_DB = 20
NAMES = ["OMA", "NOMA", "LMMSE", "DRspec", "DRadapt", "BLMMSE", "DRoracle"]


def sample_with_index(pool, batch, a, rng, idx_pool):
    """Same construction as sample_latents_pool, returning the pool indices
    so that retrieval can be scored against the correct sentences."""
    choices = np.asarray(idx_pool)
    idx = np.array([rng.choice(choices, size=U + 1, replace=False)
                    for _ in range(batch)])
    c = pool.pca_coords(idx[:, 0], DC)
    z = np.zeros((batch, U, D))
    for u in range(U):
        p = pool.pca_coords(idx[:, u + 1], D - DC)
        z[:, u, :DC] = a[u] * c
        z[:, u, DC:] = math.sqrt(1.0 - a[u] ** 2) * p
    return z, idx


def top1(query, bank, truth_pos):
    """Top-one retrieval accuracy of unit-normalized queries against a bank.

    query: (n, k); bank: (M, k) rows are the candidate sentences;
    truth_pos: (n,) index into the bank rows of the correct sentence."""
    q = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
    b = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-12)
    hit = np.empty(len(q), dtype=bool)
    step = 2000
    for s in range(0, len(q), step):
        sims = q[s:s + step] @ b.T
        hit[s:s + step] = sims.argmax(axis=1) == truth_pos[s:s + step]
    return hit


def main():
    rng = np.random.default_rng(7)
    obj = torch.load(POOL_PT, map_location="cpu")
    X = obj.numpy() if torch.is_tensor(obj) else np.asarray(obj)
    pool = EmbeddingPool(X)
    a = math.sqrt(BETA) * np.ones(U)
    B = affinity_matrix(a)
    rho = 10 ** (SNR_DB / 10)
    R = random_orthogonal(D, rng)
    V_true = R[:, :DC]

    # calibration exactly as in E2, on the training partition only
    z_cal, _ = sample_with_index(pool, N_PAIR, a, rng, IDX_TRAIN)
    x_cal = z_cal @ R.T
    V_spec, _ = learn_structure_spectral(x_cal, DC)
    err_spec = subspace_error(V_spec, V_true)
    # the adapter reuses the SAME calibration pairs, as in E2, and is
    # initialized at the spectral basis completed to a full rotation
    Q, _ = np.linalg.qr(np.hstack([V_spec, rng.standard_normal((D, D - DC))]))
    W_ad = train_adapter(x_cal, a, Q, rng)
    V_adapt = W_ad.T[:, :DC]
    err_adapt = subspace_error(V_adapt, V_true)

    # evaluation tuples from the held-out partition
    rng_e = np.random.default_rng(909)
    z, idx = sample_with_index(pool, BATCH_EVAL, a, rng_e, IDX_EVAL)
    x = z @ R.T
    tilde, h = matched_filter(x, B, rho, rng_e)

    lm, _ = demux_lmmse(tilde, B, h, rho)
    rng_c = np.random.default_rng(60001)
    est = {
        "OMA": oma_observe(x, h, rho, rng_c),
        "NOMA": demux_noma_genie(x, h, rho, rng_c),
        "LMMSE": lm,
        "DRspec": demux_dr(tilde, B, h, rho, a, V_spec),
        "DRadapt": demux_dr(tilde, B, h, rho, a, V_adapt),
        "BLMMSE": demux_block_lmmse(tilde, B, h, rho, a, V_true),
        "DRoracle": demux_dr(tilde, B, h, rho, a, V_true),
    }

    # retrieval banks built from the held-out pool, in the latent frame
    bank_s = pool.pca_coords(IDX_EVAL, DC)
    bank_p = pool.pca_coords(IDX_EVAL, D - DC)
    pos_of = {g: i for i, g in enumerate(IDX_EVAL)}
    truth_s = np.array([pos_of[g] for g in idx[:, 0]])

    notes = [f"subspace error at N={N_PAIR}: spectral {err_spec:.4f}, "
             f"adapter {err_adapt:.4f}",
             f"held-out pool size {len(IDX_EVAL)}, chance level "
             f"{1.0/len(IDX_EVAL):.4f}, SNR {SNR_DB} dB, beta {BETA}"]
    rows = {}
    thr_records = []
    for name in NAMES:
        e = est[name]
        cos, _, ser = metrics(e, x)
        zhat = e @ R                                  # back to the latent frame
        hs, hp = [], []
        for u in range(U):
            truth_p = np.array([pos_of[g] for g in idx[:, u + 1]])
            hit_s = top1(zhat[:, u, :DC], bank_s, truth_s)
            hit_p = top1(zhat[:, u, DC:], bank_p, truth_p)
            hs.append(hit_s)
            hp.append(hit_p)
            if name in ("DRspec", "DRadapt", "LMMSE", "BLMMSE"):
                cu = (e[:, u] / (np.linalg.norm(e[:, u], axis=1,
                                                keepdims=True) + 1e-12)
                      * x[:, u]).sum(-1)
                thr_records.append(np.stack([cu, hit_p.astype(float)], 1))
        acc_s = float(np.mean(np.concatenate(hs)))
        acc_p = float(np.mean(np.concatenate(hp)))
        rows[name] = (cos, ser, acc_s, acc_p)
        notes.append(f"{name:9s} cos={cos:.4f} ser={ser:.4f} "
                     f"shared_top1={acc_s:.4f} private_top1={acc_p:.4f}")
        print(notes[-1])

    with open(os.path.join(DATA, "e9_retrieval.csv"), "w") as f:
        f.write("scheme,cos,ser,shared_top1,private_top1\n")
        for n in NAMES:
            f.write(f"{n}," + ",".join(f"{v}" for v in rows[n]) + "\n")

    # --- retrieval accuracy against the recovered cosine -----------------
    rec = np.concatenate(thr_records, axis=0)
    edges = np.arange(0.0, 1.0001, 0.05)
    with open(os.path.join(DATA, "e9_threshold.csv"), "w") as f:
        f.write("cos_lo,cos_hi,n,private_top1\n")
        cross = None
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (rec[:, 0] >= lo) & (rec[:, 0] < hi)
            n = int(m.sum())
            acc = float(rec[m, 1].mean()) if n >= 30 else float("nan")
            f.write(f"{lo:.2f},{hi:.2f},{n},{acc}\n")
            if n >= 30 and acc >= 0.5 and cross is None:
                cross = lo
        notes.append(f"private top-1 retrieval first reaches 50% in the "
                     f"cosine bin starting at {cross}")
    print(notes[-1])

    with open(os.path.join(DATA, "e9_notes.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")
    print("E9 done.")


if __name__ == "__main__":
    main()
