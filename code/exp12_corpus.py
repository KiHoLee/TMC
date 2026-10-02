"""E12: co-observation on real embeddings with no imposed decomposition
(Reviewer 4.1).

Reviewer 4 observed that although E2 uses real BERT embeddings, the shared
and private components within them are constructed, which limits what the
experiment validates. This experiment removes the construction entirely.

Construction. An anchor sentence stands for the scene. The U users
transmit the embeddings of U sentences drawn from the anchor's K nearest
neighbors in the pool, which is the text-corpus analogue of several
devices reporting the same event. What each user transmits is an
unmodified pool embedding. Nothing is split into a shared part and a
private part, no common vector is injected, and no share coefficient is
set. Whatever the users have in common is whatever the encoder actually
produces for related sentences.

Measurement. The affinity is measured from the transmitted embeddings
rather than chosen, and it varies from tuple to tuple, so the experiment
also relaxes the single fixed affinity used elsewhere. The receiver is
given the affinity quantized to a grid, which is what a practical
feedback link would carry, and the share coefficients come from the
rank-one fit that the tracker of Section VI uses. The shared subspace is
recovered by the closed-form spectral estimator from calibration tuples
drawn only from the training partition.

Metrics. Top-one retrieval is well defined here, because each user
transmits an actual pool sentence, so the task asks whether the receiver
returns the sentence that was sent.

Outputs: data/e12_corpus.csv, data/e12_affinity.csv, data/e12_notes.txt
"""
import math
import os
import numpy as np
import torch

from semantic_mac import (affinity_matrix, matched_filter, demux_sr,
                          demux_sc, demux_lmmse, demux_dr, metrics,
                          learn_structure_spectral, oma_observe,
                          demux_noma_genie, EmbeddingPool)
from exp6_sideinfo import demux_block_lmmse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)

POOL_PT = os.path.join(HERE, "..", "data", "bert_agnews_8000.pt")
if not os.path.exists(POOL_PT):
    POOL_PT = os.path.join(HERE, "..", "..", "5. WCL-DRL",
                           "bert_agnews_8000.pt")

U, D = 4, 768
KNN = 8                 # neighbor pool that stands for one scene
N_CAL = 200             # calibration tuples, same budget as E2
N_EVAL = 3000
N_TRAIN_POOL = 7000
SNR_DB = 20
DC_GRID = [32, 64, 128, 256]
NAMES = ["OMA", "NOMA", "SR", "SC", "LMMSE", "BLMMSE", "DR"]


def build_tuples(Z, anchor_idx, n, rng):
    """Draw n co-observation tuples of U sentences each, restricted to the
    given anchor partition. Neighbors are searched within the same
    partition so that no evaluation sentence leaks into calibration."""
    out = np.empty((n, U), dtype=int)
    nbr = np.empty((n, KNN), dtype=int)
    sub = Z[anchor_idx]
    for i in range(n):
        A = rng.integers(len(anchor_idx))
        sims = sub @ sub[A]
        sims[A] = -9.0
        nn = np.argpartition(-sims, KNN)[:KNN]
        nbr[i] = anchor_idx[nn]
        out[i] = anchor_idx[rng.choice(nn, size=U, replace=False)]
    return out, nbr


def rank_one_fit(G, a_cap=0.99):
    """Share coefficients from the off-diagonal Gram, as in the tracker."""
    Uu = G.shape[0]
    mask = ~np.eye(Uu, dtype=bool)
    a = np.sqrt(np.clip(np.abs(G[mask]).reshape(Uu, Uu - 1).mean(axis=1),
                        1e-4, a_cap ** 2))
    for _ in range(20):
        for u in range(Uu):
            others = [v for v in range(Uu) if v != u]
            num = sum(G[u, v] * a[v] for v in others)
            den = sum(a[v] ** 2 for v in others) + 1e-9
            a[u] = np.clip(num / den, 0.0, a_cap)
    return a


def main():
    rng = np.random.default_rng(17)
    obj = torch.load(POOL_PT, map_location="cpu", weights_only=False)
    X = obj.numpy() if torch.is_tensor(obj) else np.asarray(obj)
    pool = EmbeddingPool(X)
    Z = pool.X                                   # centered, unit norm
    idx_train = np.arange(N_TRAIN_POOL)
    idx_eval = np.arange(N_TRAIN_POOL, len(Z))
    rho = 10 ** (SNR_DB / 10)
    notes = [f"pool {Z.shape}, KNN={KNN}, U={U}, SNR={SNR_DB} dB",
             "embeddings transmitted unmodified, no imposed decomposition"]

    # ---- measured affinity statistics -----------------------------------
    tup_cal, _ = build_tuples(Z, idx_train, N_CAL, rng)
    tup_ev, nbr_ev = build_tuples(Z, idx_eval, N_EVAL, rng)
    x_ev = Z[tup_ev]                                   # (N_EVAL, U, D)
    G_ev = np.einsum('nud,nvd->nuv', x_ev, x_ev)
    off = G_ev[:, ~np.eye(U, dtype=bool)].reshape(N_EVAL, -1)
    beta_meas = off.mean(axis=1)
    notes.append(f"measured pairwise affinity: mean {beta_meas.mean():.4f}, "
                 f"sd {beta_meas.std():.4f}, "
                 f"10th pct {np.percentile(beta_meas,10):.3f}, "
                 f"90th pct {np.percentile(beta_meas,90):.3f}")
    print(notes[-1])
    with open(os.path.join(DATA, "e12_affinity.csv"), "w") as f:
        f.write("tuple,beta_measured\n")
        for i, b in enumerate(beta_meas):
            f.write(f"{i},{b}\n")

    # ---- quantize the affinity, which is what feedback would carry ------
    q = np.clip(np.round(beta_meas / 0.05) * 0.05, 0.05, 0.95)
    rows = []
    for d_c in DC_GRID:
        x_cal = Z[tup_cal]
        V_spec, ev = learn_structure_spectral(x_cal, d_c)
        agg = {n: [np.zeros(0), np.zeros(0)] for n in NAMES}
        hits = {n: [] for n in NAMES}
        hits5 = {n: [] for n in NAMES}
        hitsc = {n: [] for n in NAMES}   # scene level, any
                                         # sentence in the anchor
                                         # neighbourhood
        bank = Z[idx_eval]
        pos_of = {g: i for i, g in enumerate(idx_eval)}
        for qb in np.unique(q):
            m = np.where(q == qb)[0]
            if len(m) < 20:
                continue
            xb = x_ev[m]
            Gb = np.einsum('nud,nvd->nuv', xb, xb).mean(axis=0)
            a_hat = rank_one_fit(Gb)
            B = affinity_matrix(a_hat)
            rng_b = np.random.default_rng(4000 + int(qb * 100))
            tilde, h = matched_filter(xb, B, rho, rng_b)
            lm, _ = demux_lmmse(tilde, B, h, rho)
            rng_c = np.random.default_rng(97000 + int(qb * 100))
            est = {
                "OMA": oma_observe(xb, h, rho, rng_c),
                "NOMA": demux_noma_genie(xb, h, rho, rng_c),
                "SR": demux_sr(tilde, B, h),
                "SC": demux_sc(tilde, h),
                "LMMSE": lm,
                "BLMMSE": demux_block_lmmse(tilde, B, h, rho, a_hat, V_spec),
                "DR": demux_dr(tilde, B, h, rho, a_hat, V_spec),
            }
            for n, e in est.items():
                en = e / (np.linalg.norm(e, axis=2, keepdims=True) + 1e-12)
                cos = (en * xb).sum(-1).ravel()
                agg[n][0] = np.concatenate([agg[n][0], cos])
                # top-one retrieval of the transmitted sentence
                qy = en.reshape(-1, D)
                sims = qy @ bank.T
                truth = np.array([pos_of[g] for g in tup_ev[m].ravel()])
                hits[n].append(sims.argmax(axis=1) == truth)
                top5 = np.argpartition(-sims, 5, axis=1)[:, :5]
                hits5[n].append((top5 == truth[:, None]).any(axis=1))
                # scene level: does the top hit belong to the anchor's
                # neighbourhood, that is to a report of the same event
                scene = np.array([[pos_of[g] for g in row]
                                  for row in nbr_ev[m]])
                scene = np.repeat(scene, U, axis=0)
                hitsc[n].append((sims.argmax(axis=1)[:, None]
                                 == scene).any(axis=1))
        r = {}
        for n in NAMES:
            cos = agg[n][0]
            hh = np.concatenate(hits[n])
            h5 = np.concatenate(hits5[n])
            hc = np.concatenate(hitsc[n])
            r[n] = (float(cos.mean()), float((cos < 0.45).mean()),
                    float(hh.mean()), float(h5.mean()), float(hc.mean()))
        rows.append((d_c, r))
        notes.append(f"d_c={d_c:3d}: " + " ".join(
            f"{n} cos={r[n][0]:.3f}/ser={r[n][1]:.3f}/top1={r[n][2]:.3f}"
            f"/top5={r[n][3]:.3f}/scene={r[n][4]:.3f}"
            for n in ("LMMSE", "BLMMSE", "DR", "NOMA")))
        print(notes[-1])

    with open(os.path.join(DATA, "e12_corpus.csv"), "w") as f:
        f.write("d_c," + ",".join(f"{n}_cos,{n}_ser,{n}_top1,{n}_top5,{n}_scene"
                                  for n in NAMES) + "\n")
        for d_c, r in rows:
            f.write(f"{d_c}," + ",".join(
                f"{r[n][0]},{r[n][1]},{r[n][2]},{r[n][3]},{r[n][4]}"
                for n in NAMES) + "\n")
    with open(os.path.join(DATA, "e12_notes.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")
    print("E12 done.")


if __name__ == "__main__":
    main()
