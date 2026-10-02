"""E18: sensitivity of the channel-in-the-loop adapter to its training
hyperparameters (provenance for lambda_a = 0.2, lambda_o = 1, batch 24,
400 steps, lr 3e-4; writing standard 11.7).

Same calibration protocol as E2 (N_PAIR pairs from the training pool,
spectral initialization), one hyperparameter varied at a time around the
reported setting, and the DR-adapter SER and subspace error measured on
held-out tuples at 20 dB with a fixed evaluation generator. Output:
data/e18_adapter_sensitivity.csv
"""
import math
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp2_structure as e2
from semantic_mac import (affinity_matrix, matched_filter, demux_dr, metrics,
                          random_orthogonal, EmbeddingPool, learn_structure_spectral,
                          subspace_error)

DATA = e2.DATA
GRID = ([("lam_align", v) for v in (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)]
        + [("lam_orth", v) for v in (0.3, 1.0, 3.0)]
        + [("steps", v) for v in (100, 200, 400, 800)]
        + [("batch", v) for v in (12, 24, 48)]
        + [("lr", v) for v in (1e-4, 3e-4, 1e-3)])
BASE = dict(steps=400, batch=24, lr=3e-4, lam_orth=1.0, lam_align=0.2)


def main():
    rng = np.random.default_rng(18)
    obj = torch.load(e2.POOL_PT, map_location="cpu")
    X = obj.numpy() if torch.is_tensor(obj) else np.asarray(obj)
    pool = EmbeddingPool(X)
    a = math.sqrt(e2.BETA) * np.ones(e2.U)
    B = affinity_matrix(a)
    R = random_orthogonal(e2.D, rng)
    V_true = R[:, :e2.DC]
    _, x_cal = e2.gen_clean(pool, e2.N_PAIR, a, R, rng, idx_pool=e2.IDX_TRAIN)
    V_spec, _ = learn_structure_spectral(x_cal, e2.DC)
    Q, _ = np.linalg.qr(np.hstack([V_spec, rng.standard_normal((e2.D, e2.D - e2.DC))]))
    rho = 10 ** (20 / 10)
    rng_e = np.random.default_rng(1800)
    z, x = e2.gen_clean(pool, e2.BATCH_EVAL, a, R, rng_e, idx_pool=e2.IDX_EVAL)
    tilde, h = matched_filter(x, B, rho, rng_e)
    ser_spec = metrics(demux_dr(tilde, B, h, rho, a, V_spec), x)[2]
    rows = [("spectral", "none", 0, subspace_error(V_spec, V_true), ser_spec)]
    print(f"spectral: subspace_err={rows[0][3]:.4f} ser20={ser_spec:.4f}")
    for name, val in GRID:
        kw = dict(BASE); kw[name] = val
        rng_t = np.random.default_rng(1801)          # same channel draws for every setting
        W = e2.train_adapter(x_cal, a, Q, rng_t, seed=0, **kw)
        V = W.T[:, :e2.DC]
        err = subspace_error(V, V_true)
        ser = metrics(demux_dr(tilde, B, h, rho, a, V), x)[2]
        rows.append(("adapter", name, val, err, ser))
        print(f"{name}={val}: subspace_err={err:.4f} ser20={ser:.4f}")
    with open(os.path.join(DATA, "e18_adapter_sensitivity.csv"), "w") as f:
        f.write("stage,param,value,subspace_err,ser_20dB\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    print("E18 done.")


if __name__ == "__main__":
    main()
