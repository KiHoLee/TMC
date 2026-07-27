"""E2: embedding-structure optimization on REAL PLM embeddings.

Contents are real BERT AG-News embeddings (8000 x 768, from the published
shared-embedding line); the structured latent is mixed by an unknown random
orthogonal R (the frozen encoder's arbitrary basis).  We compare:

  (i)   LMMSE            -- B-aware optimal linear receiver, no structure
  (ii)  DR + spectral    -- closed-form shared-subspace recovery from N pairs
  (iii) DR + adapter     -- channel-in-the-loop learned linear refinement
  (iv)  DR + oracle      -- true mixing basis (upper bound)

Calibration, adapter training, and subspace/spectrum sweeps draw only from
the first 7000 pool sentences; the performance ladder is evaluated on
tuples drawn from the held-out remaining 1000 sentences.

Outputs: data/e2_subspace_multi.csv, data/e2_spectrum_multi.csv,
data/e2_exponents.txt, data/e2_caliberr.txt, data/e2_ladder.csv
(figures come from replot_all.py only)
"""
import math
import os
import numpy as np
import torch

from semantic_mac import (TAU, EmbeddingPool, affinity_matrix,
                          matched_filter,
                          demux_lmmse, demux_dr, sample_latents_pool,
                          random_orthogonal, learn_structure_spectral,
                          subspace_error, metrics,
                          oma_observe, demux_noma_genie)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "fig")
DATA = os.path.join(HERE, "..", "data")
os.makedirs(FIG, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

# BERT AG-News embedding pool: shipped in data/ for the public release,
# with a fallback to the original location in the paper workspace.
POOL_PT = os.path.join(HERE, "..", "data", "bert_agnews_8000.pt")
if not os.path.exists(POOL_PT):
    POOL_PT = os.path.join(HERE, "..", "..", "5. WCL-DRL",
                           "bert_agnews_8000.pt")
U, D, DC = 4, 768, 128
BETA = 0.5
N_PAIR = 200          # paired calibration samples available to the optimizer
BATCH_EVAL = 4000
N_TRAIN_POOL = 7000   # sentences usable for calibration/adapter training
# evaluation tuples are drawn only from the held-out remainder of the pool
IDX_TRAIN = np.arange(N_TRAIN_POOL)
IDX_EVAL = np.arange(N_TRAIN_POOL, 8000)


def gen_clean(pool, n, a, R, rng, idx_pool=None):
    z = sample_latents_pool(pool, n, U, D, DC, a, rng, idx_pool=idx_pool)
    return z, z @ R.T


# ----------------------------------------------------------------------
#  Learned linear adapter (channel-in-the-loop refinement)
# ----------------------------------------------------------------------
def dr_torch(tilde, B, h, rho, a, W, d_c):
    """Differentiable decomposition receiver in the adapter frame W (d x d).
    Returns ambient-frame estimates W^T z_hat."""
    Bt, Ut, d = tilde.shape
    sigma2 = 1.0 / rho
    y = tilde @ W.T                                   # rotated MF outputs
    Zs, Zp = y[:, :, :d_c], y[:, :, d_c:]
    Hi = torch.diag_embed(1.0 / h)
    Hm = torch.diag_embed(h)
    Bb = B.unsqueeze(0).expand(Bt, Ut, Ut)
    Gamma = Hi @ Bb @ Hm
    Cn = sigma2 * (Hi @ Bb @ Hi)
    ratio = h.unsqueeze(1) / h.unsqueeze(2)           # h_v / h_u
    gamma = a.view(1, 1, Ut) * Bb * ratio
    gamma = gamma.sum(dim=2) - (a.view(1, Ut) * (Bb.diagonal(dim1=1, dim2=2) - 1.0))
    # gamma_u = a_u + sum_{v != u} B_uv a_v h_v/h_u  (diagonal of Bb is 1)
    Cn_inv = torch.linalg.inv(Cn)
    gCg = torch.einsum('bu,buv,bv->b', gamma, Cn_inv, gamma).clamp_min(1e-9)
    c_hat = torch.einsum('bu,buv,bvk->bk', gamma, Cn_inv, Zs) / gCg.unsqueeze(-1)
    shrink_c = (1.0 / d_c) / (1.0 / d_c + 1.0 / gCg)
    c_hat = c_hat * shrink_c.unsqueeze(-1)
    Gp = torch.linalg.solve(Gamma, Zp)
    Binv_uu = torch.linalg.inv(B).diagonal()
    sig_p = (1.0 - a ** 2).clamp_min(1e-9)
    err_p = (d - d_c) * sigma2 * Binv_uu.view(1, Ut) / (h ** 2)
    shrink_p = sig_p.view(1, Ut) / (sig_p.view(1, Ut) + err_p)
    z_hat = torch.cat([
        a.view(1, Ut, 1) * c_hat.unsqueeze(1).expand(Bt, Ut, d_c),
        shrink_p.unsqueeze(-1) * Gp], dim=2)
    return z_hat @ W                                  # back to ambient frame


def train_adapter(x_cal, a, V_init, rng, steps=400, batch=24, lr=3e-4,
                  lam_orth=1.0, lam_align=0.2, seed=0):
    """Refine the full rotation W (init = spectral basis) through the channel.

    x_cal: the SAME N_PAIR calibration embeddings used by the spectral
    estimator (the adapter adds no data cost, as stated in the paper)."""
    torch.manual_seed(seed)
    W = torch.nn.Parameter(torch.tensor(V_init.T, dtype=torch.float32))
    a_t = torch.tensor(a, dtype=torch.float32)
    B_np = affinity_matrix(a)
    B_t = torch.tensor(B_np, dtype=torch.float32)
    opt = torch.optim.Adam([W], lr=lr)
    n_cal = x_cal.shape[0]
    x_cal_t = torch.tensor(x_cal, dtype=torch.float32)
    for it in range(steps):
        idx = torch.randint(0, n_cal, (batch,))
        x = x_cal_t[idx]
        snr_db = float(rng.uniform(0, 20))
        rho = 10 ** (snr_db / 10)
        tilde_np, h_np = matched_filter(x.numpy().astype(np.float64),
                                        B_np, rho, rng)
        tilde = torch.tensor(tilde_np, dtype=torch.float32)
        h = torch.tensor(h_np, dtype=torch.float32)
        x_hat = dr_torch(tilde, B_t, h, rho, a_t, W, DC)
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
        if (it + 1) % 100 == 0:
            print(f"  adapter step {it+1}: loss={loss.item():.4f} "
                  f"cosd={cosd.item():.4f} align={align.item():.4f}")
    with torch.no_grad():
        # re-orthonormalize
        Uo, _, Vo = torch.linalg.svd(W)
        Wo = Uo @ Vo
    return Wo.numpy().astype(np.float64)


def main():
    rng = np.random.default_rng(7)
    obj = torch.load(POOL_PT, map_location="cpu")
    X = obj.numpy() if torch.is_tensor(obj) else np.asarray(obj)
    pool = EmbeddingPool(X)
    print(f"pool: {pool.N} x {pool.d}")
    a = math.sqrt(BETA) * np.ones(U)
    B = affinity_matrix(a)
    R = random_orthogonal(D, rng)
    V_true = R[:, :DC]

    # --- subspace recovery vs N for multiple user counts ------------------
    # Multi-trial averaging; the N grid starts above d_c = 128 because for
    # N < d_c the empirical cross-covariance is rank-deficient in the shared
    # block (at most N of the d_c shared directions are excited), which
    # produces a systematic plateau rather than smooth decay.
    TRIALS = 12
    Ns = [200, 400, 800, 1600, 3200, 6400]
    err_multi = {}
    with open(os.path.join(DATA, "e2_subspace_multi.csv"), "w") as f:
        f.write("U,N,err\n")
        for Um in (2, 4, 8):
            a_u = math.sqrt(BETA) * np.ones(Um)
            err_multi[Um] = []
            for n in Ns:
                errs_t = []
                for t in range(TRIALS):
                    rng_t = np.random.default_rng(42000 + 1000 * Um
                                                  + 10 * n + t)
                    z_u = sample_latents_pool(pool, n, Um, D, DC, a_u, rng_t,
                                              idx_pool=IDX_TRAIN)
                    Vh, _ = learn_structure_spectral(z_u @ R.T, DC)
                    errs_t.append(subspace_error(Vh, V_true))
                e_u = float(np.mean(errs_t))
                err_multi[Um].append(e_u)
                f.write(f"{Um},{n},{e_u}\n")
                print(f"U={Um} N={n:5d} err={e_u:.4f} "
                      f"(std {np.std(errs_t):.4f})")

    # power-law exponents of the subspace-error decay (quoted in the paper)
    with open(os.path.join(DATA, "e2_exponents.txt"), "w") as f:
        for Um in (2, 4, 8):
            slope = np.polyfit(np.log(Ns), np.log(err_multi[Um]), 1)[0]
            f.write(f"U={Um} exponent={slope:.3f}\n")
            print(f"U={Um} power-law exponent {slope:.3f}")

    # --- eigen-spectrum for several calibration sizes (fresh rng) ---------
    with open(os.path.join(DATA, "e2_spectrum_multi.csv"), "w") as f:
        f.write("N,idx,eig\n")
        for n in (100, 400, 1600):
            rng_s = np.random.default_rng(5200 + n)
            z_s = sample_latents_pool(pool, n, U, D, DC, a, rng_s,
                                      idx_pool=IDX_TRAIN)
            _, ev = learn_structure_spectral(z_s @ R.T, DC)
            for i, w in enumerate(ev[:400]):
                f.write(f"{n},{i+1},{w}\n")

    # --- calibration with the paper budget --------------------------------
    _, x_cal = gen_clean(pool, N_PAIR, a, R, rng, idx_pool=IDX_TRAIN)
    V_spec, _ = learn_structure_spectral(x_cal, DC)
    err_spec = subspace_error(V_spec, V_true)
    print(f"spectral (N={N_PAIR}): subspace_err={err_spec:.4f}")

    # full basis for adapter init: complete V_spec to an orthonormal basis
    Q, _ = np.linalg.qr(np.hstack([
        V_spec, rng.standard_normal((D, D - DC))]))
    V_full = Q
    print("training adapter (same calibration pairs as the spectral step) ...")
    W_ad = train_adapter(x_cal, a, V_full, rng)
    err_ad = subspace_error(W_ad.T[:, :DC], V_true)
    print(f"adapter: subspace_err={err_ad:.4f}")
    with open(os.path.join(DATA, "e2_caliberr.txt"), "w") as f:
        f.write(f"spectral_err={err_spec}\nadapter_err={err_ad}\n")

    # --- performance ladder vs SNR (held-out pool sentences) ---------------
    def inst_cos(e_hat, x_true):
        """Per-instance cosine (batch, U) without consuming any RNG."""
        e_n = e_hat / (np.linalg.norm(e_hat, axis=2, keepdims=True) + 1e-12)
        return (e_n * x_true).sum(-1)

    snrs = list(range(0, 21, 4))
    rows = []
    paired = []
    for s in snrs:
        rho = 10 ** (s / 10)
        z, x = gen_clean(pool, BATCH_EVAL, a, R, rng, idx_pool=IDX_EVAL)
        tilde, h = matched_filter(x, B, rho, rng)
        r = {}
        lm, _ = demux_lmmse(tilde, B, h, rho)
        r["LMMSE"] = metrics(lm, x)
        out_spec = demux_dr(tilde, B, h, rho, a, V_spec)
        out_adapt = demux_dr(tilde, B, h, rho, a, W_ad.T[:, :DC])
        r["DR-spec"] = metrics(out_spec, x)
        r["DR-adapt"] = metrics(out_adapt, x)
        r["DR-oracle"] = metrics(demux_dr(tilde, B, h, rho, a, V_true), x)
        rng_c = np.random.default_rng(91000 + s)
        r["OMA"] = metrics(oma_observe(x, h, rho, rng_c), x)
        r["NOMA"] = metrics(demux_noma_genie(x, h, rho, rng_c), x)
        rows.append(r)
        # paired spec-vs-adapter error difference on the SAME channel draws
        err_s = (inst_cos(out_spec, x) < TAU).astype(float).ravel()
        err_a = (inst_cos(out_adapt, x) < TAU).astype(float).ravel()
        d_i = err_s - err_a
        se = float(d_i.std(ddof=1) / math.sqrt(d_i.size))
        paired.append((s, float(err_s.mean()), float(err_a.mean()),
                       float(d_i.mean()), se))
        print(f"snr={s:2d} " + " ".join(
            f"{k}:cos={v[0]:.3f},ser={v[2]:.3f}" for k, v in r.items())
            + f" paired_diff={d_i.mean():+.5f} se={se:.5f}")
    keys = ["OMA", "NOMA", "LMMSE", "DR-spec", "DR-adapt", "DR-oracle"]
    with open(os.path.join(DATA, "e2_ladder.csv"), "w") as f:
        f.write("snr," + ",".join(f"{k}_cos,{k}_nmse,{k}_ser" for k in keys) + "\n")
        for s, r in zip(snrs, rows):
            f.write(f"{s}," + ",".join(
                f"{r[k][0]},{r[k][1]},{r[k][2]}" for k in keys) + "\n")
    with open(os.path.join(DATA, "e2_paired.csv"), "w") as f:
        f.write("snr,ser_spec,ser_adapt,mean_diff,se_diff\n")
        for row in paired:
            f.write(",".join(str(v) for v in row) + "\n")

    # figures are produced only by the canonical replot_all.py (uniform
    # geometry); experiment scripts write CSVs exclusively.
    print("E2 done. Run replot_all.py to regenerate the figures.")


if __name__ == "__main__":
    main()
