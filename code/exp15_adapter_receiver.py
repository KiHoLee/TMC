"""E15: is the structure exposure receiver-agnostic?

The spectral step that recovers the shared subspace never refers to a
receiver, so both the DR and DR-MMSE obtain it in
exactly the same way. The adapter is different, because it is trained by
passing calibration embeddings through a differentiable receiver, and
the manuscript trains it through the DR. This experiment asks whether
that choice matters.

Two adapters are trained from the same spectral initialization and the
same calibration pairs, one through the differentiable DR and one
through a differentiable DR-MMSE. Each receiver is then
evaluated on every basis, so the self pairing and the cross pairing can
be compared.

If the two adapters give each receiver the same performance, the
structure exposure of Section V is receiver-agnostic and can be reported
as a property of the framework rather than of one receiver. If they do
not, the difference is itself worth reporting.

Outputs: data/e15_adapter_receiver.csv, data/e15_notes.txt
"""
import math
import os
import numpy as np
import torch

from semantic_mac import (EmbeddingPool, affinity_matrix, matched_filter,
                          demux_dr, random_orthogonal,
                          learn_structure_spectral, subspace_error, metrics)
from exp6_sideinfo import demux_block_lmmse
from exp2_structure import dr_torch
from exp9_retrieval import sample_with_index, top1

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


def blmmse_torch(tilde, B, h, rho, a, W, d_c):
    """Differentiable DR-MMSE in the adapter frame W.

    Mirrors dr_torch, with the exact estimator on both blocks: the MMSE
    form on the shared block and the joint filter on the complement."""
    Bt, Ut, d = tilde.shape
    sigma2 = 1.0 / rho
    y = tilde @ W.T
    Zs, Zp = y[:, :, :d_c], y[:, :, d_c:]
    Hi = torch.diag_embed(1.0 / h)
    Hm = torch.diag_embed(h)
    Bb = B.unsqueeze(0).expand(Bt, Ut, Ut)
    Gamma = Hi @ Bb @ Hm
    Cn = sigma2 * (Hi @ Bb @ Hi)
    ratio = h.unsqueeze(1) / h.unsqueeze(2)
    gamma = a.view(1, 1, Ut) * Bb * ratio
    gamma = gamma.sum(dim=2) - (a.view(1, Ut)
                                * (Bb.diagonal(dim1=1, dim2=2) - 1.0))
    Cn_inv = torch.linalg.inv(Cn)
    gCg = torch.einsum('bu,buv,bv->b', gamma, Cn_inv, gamma).clamp_min(1e-9)
    c_hat = (torch.einsum('bu,buv,bvk->bk', gamma, Cn_inv, Zs)
             / (gCg + d_c).unsqueeze(-1))
    d_p = d - d_c
    Cp = torch.diag_embed((1.0 - a ** 2).clamp_min(1e-9) / d_p)
    Cp = Cp.unsqueeze(0).expand(Bt, Ut, Ut)
    S = Gamma @ Cp @ Gamma.transpose(1, 2) + Cn
    A = Cp @ Gamma.transpose(1, 2)
    Wp = torch.linalg.solve(S.transpose(1, 2),
                            A.transpose(1, 2)).transpose(1, 2)
    Xp = Wp @ Zp
    z_hat = torch.cat([
        a.view(1, Ut, 1) * c_hat.unsqueeze(1).expand(Bt, Ut, d_c), Xp], dim=2)
    return z_hat @ W


def train_adapter_with(recv, x_cal, a, V_init, rng, steps=400, batch=24,
                       lr=3e-4, lam_orth=1.0, lam_align=0.2, seed=0):
    """Same schedule as the manuscript adapter, with the receiver in the
    loop supplied by the caller."""
    torch.manual_seed(seed)
    W = torch.nn.Parameter(torch.tensor(V_init.T, dtype=torch.float32))
    a_t = torch.tensor(a, dtype=torch.float32)
    B_np = affinity_matrix(a)
    B_t = torch.tensor(B_np, dtype=torch.float32)
    opt = torch.optim.Adam([W], lr=lr)
    x_cal_t = torch.tensor(x_cal, dtype=torch.float32)
    for it in range(steps):
        idx = torch.randint(0, x_cal.shape[0], (batch,))
        x = x_cal_t[idx]
        rho = 10 ** (float(rng.uniform(0, 20)) / 10)
        tilde_np, h_np = matched_filter(x.numpy().astype(np.float64),
                                        B_np, rho, rng)
        tilde = torch.tensor(tilde_np, dtype=torch.float32)
        h = torch.tensor(h_np, dtype=torch.float32)
        x_hat = recv(tilde, B_t, h, rho, a_t, W, DC)
        cosd = 1.0 - torch.nn.functional.cosine_similarity(
            x_hat, x, dim=2).mean()
        y = x @ W.T
        s = torch.nn.functional.normalize(y[:, :, :DC], dim=2)
        align = (s.unsqueeze(1) - s.unsqueeze(2)).pow(2).sum(-1).mean()
        orth = (W @ W.T - torch.eye(D)).pow(2).mean()
        loss = cosd + lam_align * align + lam_orth * orth
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % 200 == 0:
            print(f"    step {it+1}: loss={loss.item():.4f}")
    with torch.no_grad():
        Uo, _, Vo = torch.linalg.svd(W)
        Wo = Uo @ Vo
    return Wo.numpy().astype(np.float64)


def main():
    rng = np.random.default_rng(7)
    obj = torch.load(POOL_PT, map_location="cpu", weights_only=False)
    X = obj.numpy() if torch.is_tensor(obj) else np.asarray(obj)
    pool = EmbeddingPool(X)
    a = math.sqrt(BETA) * np.ones(U)
    B = affinity_matrix(a)
    rho = 10 ** (SNR_DB / 10)
    R = random_orthogonal(D, rng)
    V_true = R[:, :DC]

    # --- verify the differentiable receiver against the numpy one -------
    z_v, _ = sample_with_index(pool, 64, a, rng, IDX_TRAIN)
    x_v = z_v @ R.T
    tl, hh = matched_filter(x_v, B, rho, rng)
    ref = demux_block_lmmse(tl, B, hh, rho, a, V_true)
    with torch.no_grad():
        got = blmmse_torch(torch.tensor(tl, dtype=torch.float64),
                           torch.tensor(B), torch.tensor(hh),
                           rho, torch.tensor(a),
                           torch.tensor(R.T), DC).numpy()
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    notes = [f"differentiable DR-MMSE matches the numpy receiver to "
             f"relative {rel:.2e}"]
    print(notes[0])
    assert rel < 1e-6, "differentiable receiver does not match"

    # --- calibration, one spectral basis, two adapters ------------------
    z_cal, _ = sample_with_index(pool, N_PAIR, a, rng, IDX_TRAIN)
    x_cal = z_cal @ R.T
    V_spec, _ = learn_structure_spectral(x_cal, DC)
    Q, _ = np.linalg.qr(np.hstack([V_spec, rng.standard_normal((D, D - DC))]))
    print("  training adapter through the DR ...")
    W_dr = train_adapter_with(dr_torch, x_cal, a, Q, np.random.default_rng(21))
    print("  training adapter through DR-MMSE ...")
    W_sub = train_adapter_with(blmmse_torch, x_cal, a, Q,
                               np.random.default_rng(21))
    bases = {"spectral": V_spec,
             "adapter(DR)": W_dr.T[:, :DC],
             "adapter(subspace)": W_sub.T[:, :DC],
             "true": V_true}
    for k, V in bases.items():
        notes.append(f"subspace error, {k}: {subspace_error(V, V_true):.4f}")
        print(notes[-1])

    # --- evaluate every receiver on every basis -------------------------
    rng_e = np.random.default_rng(909)
    z, idx = sample_with_index(pool, BATCH_EVAL, a, rng_e, IDX_EVAL)
    x = z @ R.T
    tilde, h = matched_filter(x, B, rho, rng_e)
    bank_p = pool.pca_coords(IDX_EVAL, D - DC)
    pos_of = {g: i for i, g in enumerate(IDX_EVAL)}
    rows = []
    for bname, Vc in bases.items():
        for rname, fn in (("DR", demux_dr), ("Subspace", demux_block_lmmse)):
            e = fn(tilde, B, h, rho, a, Vc)
            cos, _, ser = metrics(e, x)
            zhat = e @ R
            hp = [top1(zhat[:, u, DC:], bank_p,
                       np.array([pos_of[g] for g in idx[:, u + 1]]))
                  for u in range(U)]
            t1 = float(np.mean(np.concatenate(hp)))
            rows.append((bname, rname, cos, ser, t1))
            notes.append(f"basis={bname:18s} receiver={rname:9s} "
                         f"cos={cos:.4f} ser={ser:.4f} top1={t1:.4f}")
            print(notes[-1])

    with open(os.path.join(DATA, "e15_adapter_receiver.csv"), "w") as f:
        f.write("basis,receiver,cos,ser,private_top1\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    with open(os.path.join(DATA, "e15_notes.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")
    print("E15 done.")


if __name__ == "__main__":
    main()
