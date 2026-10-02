"""E16: why the decomposition receiver loses the scene metric in E12, and
what recovers it.

In E12 the framework holds the lower semantic error rate at every shared
dimension and the higher scene error rate at every shared dimension. The
scene metric asks only whether the retrieved sentence is a report of the
same event, and there the unstructured LMMSE receiver is far ahead
(0.009 against 0.154 at d_c=32). This experiment separates the possible
causes so the loss can be attributed rather than described.

Three candidate causes.

(a) Noise amplification in the complement. The DR cancels there with
    zero forcing, which inverts the coupling and amplifies noise. If
    this dominates, raising the SNR must close the gap, and replacing
    zero forcing by the minimum mean-square-error filter must also
    close part of it.

(b) Model mismatch. The corpus correlation is diffuse, so the rank-one
    share-coefficient fit and the single global shared basis are both
    approximations, and the receiver is additionally given the affinity
    on a coarse grid. If this dominates, a finer affinity grid helps
    and a higher SNR does not.

(c) The combining branch itself. Combining replaces the shared
    coordinates of every user by one common estimate, which moves each
    estimate toward the centre of the tuple. If this dominates, nothing
    that acts on the complement can recover the scene metric, and the
    loss is intrinsic to the architecture on diffuse correlation.

The variants below hold everything else fixed, so each isolates one
cause. SC is reported alongside as the pure combiner and LMMSE as the
unstructured reference.

Outputs: data/e16_scene_gap.csv, data/e16_notes.txt
"""
import os
import numpy as np
import torch

from semantic_mac import (affinity_matrix, matched_filter, demux_sc,
                          demux_lmmse, demux_dr, learn_structure_spectral,
                          EmbeddingPool)
from exp6_sideinfo import demux_block_lmmse
from exp12_corpus import (build_tuples, rank_one_fit, U, D, KNN,
                          N_CAL, N_TRAIN_POOL, POOL_PT)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)

N_EVAL = 3000
DC_GRID = [32, 128]
SNR_GRID = [20, 30]
QSTEP = {"coarse": 0.05, "fine": 0.01}


def score(e, xb, bank, pos_of, tup_m, nbr_m):
    """Cosine, semantic error rate and scene error rate for one estimate."""
    en = e / (np.linalg.norm(e, axis=2, keepdims=True) + 1e-12)
    cos = (en * xb).sum(-1).ravel()
    sims = en.reshape(-1, D) @ bank.T
    top = sims.argmax(axis=1)
    truth = np.array([pos_of[g] for g in tup_m.ravel()])
    scene = np.repeat(np.array([[pos_of[g] for g in row] for row in nbr_m]),
                      U, axis=0)
    return cos, (top == truth), (top[:, None] == scene).any(axis=1)


def main():
    rng = np.random.default_rng(17)
    obj = torch.load(POOL_PT, map_location="cpu", weights_only=False)
    X = obj.numpy() if torch.is_tensor(obj) else np.asarray(obj)
    pool = EmbeddingPool(X)
    Z = pool.X
    idx_train = np.arange(N_TRAIN_POOL)
    idx_eval = np.arange(N_TRAIN_POOL, len(Z))
    bank = Z[idx_eval]
    pos_of = {g: i for i, g in enumerate(idx_eval)}

    tup_cal, _ = build_tuples(Z, idx_train, N_CAL, rng)
    tup_ev, nbr_ev = build_tuples(Z, idx_eval, N_EVAL, rng)
    x_ev = Z[tup_ev]
    G_ev = np.einsum('nud,nvd->nuv', x_ev, x_ev)
    beta_meas = G_ev[:, ~np.eye(U, dtype=bool)].reshape(N_EVAL, -1).mean(1)

    notes = [f"pool {Z.shape}, KNN={KNN}, U={U}, N_EVAL={N_EVAL}",
             f"measured affinity mean {beta_meas.mean():.4f} "
             f"sd {beta_meas.std():.4f}"]
    print(notes[-1])

    rows = []
    for d_c in DC_GRID:
        V_spec, _ = learn_structure_spectral(Z[tup_cal], d_c)
        for snr_db in SNR_GRID:
            rho = 10 ** (snr_db / 10)
            for gname, step in QSTEP.items():
                q = np.clip(np.round(beta_meas / step) * step, 0.05, 0.95)
                acc = {}
                for qb in np.unique(q):
                    m = np.where(q == qb)[0]
                    if len(m) < 20:
                        continue
                    xb = x_ev[m]
                    Gb = np.einsum('nud,nvd->nuv', xb, xb).mean(axis=0)
                    a_hat = rank_one_fit(Gb)
                    B = affinity_matrix(a_hat)
                    rb = np.random.default_rng(8000 + int(qb * 1000) + snr_db)
                    tilde, h = matched_filter(xb, B, rho, rb)
                    lm, _ = demux_lmmse(tilde, B, h, rho)
                    est = {
                        "LMMSE": lm,
                        "SC": demux_sc(tilde, h),
                        "DR": demux_dr(tilde, B, h, rho, a_hat, V_spec),
                        "DR-MMSE": demux_block_lmmse(tilde, B, h, rho,
                                                     a_hat, V_spec),
                    }
                    for n, e in est.items():
                        c, hh, hs = score(e, xb, bank, pos_of,
                                          tup_ev[m], nbr_ev[m])
                        a0, a1, a2 = acc.setdefault(
                            n, [np.zeros(0), np.zeros(0), np.zeros(0)])
                        acc[n] = [np.concatenate([a0, c]),
                                  np.concatenate([a1, hh]),
                                  np.concatenate([a2, hs])]
                for n, (c, hh, hs) in acc.items():
                    ser = float((c < 0.45).mean())
                    scene_err = float(1.0 - hs.mean())
                    rows.append((d_c, snr_db, gname, n, float(c.mean()),
                                 ser, float(hh.mean()), scene_err))
                    notes.append(
                        f"d_c={d_c:3d} {snr_db}dB grid={gname:6s} "
                        f"{n:8s} cos={c.mean():.3f} ser={ser:.3f} "
                        f"top1={hh.mean():.3f} scene_err={scene_err:.3f}")
                    print(notes[-1])

    with open(os.path.join(DATA, "e16_scene_gap.csv"), "w") as f:
        f.write("d_c,snr,grid,receiver,cos,ser,top1,scene_err\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    with open(os.path.join(DATA, "e16_notes.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")
    print("E16 done.")


if __name__ == "__main__":
    main()
