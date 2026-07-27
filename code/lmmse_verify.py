"""
Verify Proposition-candidate 1 for the 8th (TMC) paper:

  The B-aware LMMSE receiver
      W*(B,H,sigma^2) = C_x Gamma^T (Gamma C_x Gamma^T + C_n)^{-1}
  (i)   -> Gamma^{-1} (= SR / AA-EDMA demux)      as rho -> inf
  (ii)  -> per-user Wiener (no cross-processing)  as beta -> 0
  (iii) -> coherent combining (SC / MRC)          as beta -> 1
  (iv)  dominates SR and SC at every (beta, rho); strict gap at intermediate beta.

Also simulates an oracle decomposition receiver (ODR) that knows the
shared/private subspace split — the ceiling motivating structured
embedding learning.

Self-contained: model/conventions come from the project's own
semantic_mac.py (matched filter, affinity, metrics), so this script
verifies the same library that produces the paper results.
"""
from __future__ import annotations
import math
import os
import numpy as np

from semantic_mac import affinity_matrix, matched_filter, metrics

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "fig")
os.makedirs(OUT, exist_ok=True)

TAU = 0.45  # SER threshold, as in CL paper


# ----------------------------------------------------------------------
#  Structured embedding generator with a FIXED shared subspace
#  (needed for the ODR ceiling; statistically identical Gram to uwca_core's
#   sample_embeddings: E[<e_u,e_v>] = a_u a_v)
# ----------------------------------------------------------------------
def sample_embeddings_structured(batch, U, d, d_c, a, rng):
    """Shared content c lives in the FIRST d_c coordinates (unit norm there);
    private p_u are mutually orthogonal unit vectors in the remaining d-d_c.
      e_u = a_u * [c; 0] + sqrt(1-a_u^2) * [0; p_u]
    Returns e (batch,U,d), c (batch,d_c), p (batch,U,d-d_c)."""
    a = np.broadcast_to(np.asarray(a, float), (U,))
    e = np.zeros((batch, U, d))
    c_all = np.zeros((batch, d_c))
    p_all = np.zeros((batch, U, d - d_c))
    for b in range(batch):
        c = rng.standard_normal(d_c)
        c /= np.linalg.norm(c)
        G = rng.standard_normal((d - d_c, U))
        Q, _ = np.linalg.qr(G)                      # orthonormal private dirs
        c_all[b] = c
        for u in range(U):
            p_all[b, u] = Q[:, u]
            e[b, u, :d_c] = a[u] * c
            e[b, u, d_c:] = math.sqrt(1.0 - a[u] ** 2) * Q[:, u]
    return e, c_all, p_all


# ----------------------------------------------------------------------
#  Receivers (all consume the matched-filter outputs `tilde`)
# ----------------------------------------------------------------------
def lmmse_matrices(B, h, sigma2, d):
    """Return (W, Eerr) for one sample: W (U,U), Eerr (U,U) per-dim error cov."""
    H = np.diag(h)
    Hi = np.diag(1.0 / h)
    Gamma = Hi @ B @ H
    Cx = B / d
    Cn = sigma2 * (Hi @ B @ Hi)
    S = Gamma @ Cx @ Gamma.T + Cn
    W = Cx @ Gamma.T @ np.linalg.solve(S.T, np.eye(len(h))).T  # Cx Γ^T S^{-1}
    Eerr = Cx - W @ Gamma @ Cx
    return W, Eerr


def demux_lmmse(tilde, B, h, rho):
    """B-aware LMMSE. Returns (e_hat_raw, closed-form per-user MSE avg over batch)."""
    batch, U, d = tilde.shape
    sigma2 = 1.0 / rho
    out = np.empty_like(tilde)
    mse_cf = np.zeros(U)
    for b in range(batch):
        W, Eerr = lmmse_matrices(B, h[b], sigma2, d)
        out[b] = W @ tilde[b]
        mse_cf += d * np.diag(Eerr)
    return out, mse_cf / batch


def demux_sr_raw(tilde, B, h):
    """SR (AA-EDMA) without normalization, for raw-MSE comparison."""
    batch, U, d = tilde.shape
    out = np.empty_like(tilde)
    for b in range(batch):
        H = np.diag(h[b])
        Gamma = np.linalg.inv(H) @ B @ H
        out[b] = np.linalg.solve(Gamma, tilde[b])
    return out


def sr_mse_closed(B, h, rho, d):
    """d * sigma^2 [B^{-1}]_uu / h_u^2, averaged over batch of h."""
    Binv_uu = np.diag(np.linalg.inv(B))
    return (d / rho) * (Binv_uu[None, :] / h ** 2).mean(axis=0)


def demux_sc_mrc(tilde, h):
    """Pure combining (SC endpoint): every user gets the h^2-weighted sum of
    all matched-filter outputs (MRC under the fully-shared hypothesis)."""
    w = h ** 2                                       # (batch,U)
    w = w / w.sum(axis=1, keepdims=True)
    comb = np.einsum('bu,bud->bd', w, tilde)         # (batch,d)
    return np.repeat(comb[:, None, :], tilde.shape[1], axis=1)


def demux_odr(tilde, B, h, rho, a, d_c):
    """Oracle decomposition receiver: knows the fixed shared subspace
    (first d_c coords), a and h.
      shared:  BLUE-combine the U looks at c, then Wiener shrink
      private: SR (Gamma^{-1}) on the complement, then Wiener shrink
      recombine e_hat_u = a_u c_hat + g_hat_u."""
    batch, U, d = tilde.shape
    sigma2 = 1.0 / rho
    a = np.broadcast_to(np.asarray(a, float), (U,))
    out = np.empty_like(tilde)
    for b in range(batch):
        hb = h[b]
        Hi = np.diag(1.0 / hb)
        Gamma = np.diag(1.0 / hb) @ B @ np.diag(hb)
        Cn = sigma2 * (Hi @ B @ Hi)                  # per-dim MF noise cov
        # ---- shared part: z_u = gamma_u * c + n_u on first d_c dims
        gamma = np.array([
            a[u] + sum(B[u, v] * a[v] * hb[v] / hb[u] for v in range(U) if v != u)
            for u in range(U)
        ])
        Z = tilde[b, :, :d_c]                        # (U, d_c)
        Cn_inv = np.linalg.inv(Cn)
        denom = gamma @ Cn_inv @ gamma
        if denom > 1e-12:                            # beta=0 -> no shared part
            c_hat = (gamma @ Cn_inv @ Z) / denom     # BLUE, (d_c,)
            # Wiener shrink: per-dim signal var 1/d_c, BLUE error var 1/denom
            shrink_c = (1.0 / d_c) / (1.0 / d_c + 1.0 / denom)
            c_hat = shrink_c * c_hat
        else:
            c_hat = np.zeros(d_c)
        # ---- private part: SR on the complement
        Gp = np.linalg.solve(Gamma, tilde[b, :, d_c:])   # (U, d-d_c) est of g_u
        Binv = np.linalg.inv(B)
        for u in range(U):
            sig_p = 1.0 - a[u] ** 2                       # ||g_u||^2
            err_p = (d - d_c) * sigma2 * Binv[u, u] / hb[u] ** 2
            shrink_p = sig_p / (sig_p + err_p)
            out[b, u, :d_c] = a[u] * c_hat
            out[b, u, d_c:] = shrink_p * Gp[u]
    return out


# ----------------------------------------------------------------------
#  Evaluation helpers
# ----------------------------------------------------------------------
def eval_all(e, tilde, h, B, rho, a, d_c, with_odr=True):
    """Return dict name -> (nmse_raw, cos, ser) plus closed forms."""
    d = e.shape[2]
    res = {}

    def add(name, e_hat_raw):
        nmse = ((e_hat_raw - e) ** 2).sum(-1).mean()
        e_n = e_hat_raw / (np.linalg.norm(e_hat_raw, axis=2, keepdims=True) + 1e-12)
        cos, _, ser = metrics(e_n, e, tau=TAU)
        res[name] = dict(nmse=float(nmse), cos=cos, ser=ser)

    add("MF (TIN)", tilde.copy())
    add("SR (AA-EDMA)", demux_sr_raw(tilde, B, h))
    add("SC (MRC)", demux_sc_mrc(tilde, h))
    lm, mse_cf = demux_lmmse(tilde, B, h, rho)
    add("LMMSE (proposed)", lm)
    res["LMMSE (proposed)"]["nmse_cf"] = float(mse_cf.mean())
    res["SR (AA-EDMA)"]["nmse_cf"] = float(sr_mse_closed(B, h, rho, d).mean())
    if with_odr:
        add("ODR (oracle)", demux_odr(tilde, B, h, rho, a, d_c))
    return res


def run_sweep(betas, rho, U=4, d=64, d_c=16, batch=3000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for beta in betas:
        a = math.sqrt(beta) * np.ones(U)
        B = affinity_matrix(a)
        e, _, _ = sample_embeddings_structured(batch, U, d, d_c, a, rng)
        tilde, h = matched_filter(e, B, rho, rng, fading=True)
        res = eval_all(e, tilde, h, B, rho, a, d_c)
        rows.append((beta, res))
        lm, sr = res["LMMSE (proposed)"], res["SR (AA-EDMA)"]
        print(f"beta={beta:4.2f} | LMMSE nmse {lm['nmse']:.4f} (cf {lm['nmse_cf']:.4f}) "
              f"cos {lm['cos']:.3f} | SR nmse {sr['nmse']:.4f} (cf {sr['nmse_cf']:.4f}) "
              f"cos {sr['cos']:.3f} | SC cos {res['SC (MRC)']['cos']:.3f} "
              f"| ODR cos {res['ODR (oracle)']['cos']:.3f}")
    return rows


def run_snr_sweep(beta, rhos_db, U=4, d=64, d_c=16, batch=3000, seed=1):
    rng = np.random.default_rng(seed)
    a = math.sqrt(beta) * np.ones(U)
    B = affinity_matrix(a)
    rows = []
    for rdb in rhos_db:
        rho = 10 ** (rdb / 10)
        e, _, _ = sample_embeddings_structured(batch, U, d, d_c, a, rng)
        tilde, h = matched_filter(e, B, rho, rng, fading=True)
        res = eval_all(e, tilde, h, B, rho, a, d_c)
        rows.append((rdb, res))
        print(f"SNR={rdb:3d} dB | " + " | ".join(
            f"{k.split(' ')[0]} cos {v['cos']:.3f} ser {v['ser']:.3f}"
            for k, v in res.items()))
    return rows


# ----------------------------------------------------------------------
#  Endpoint checks (Proposition 1 (i)-(iii))
# ----------------------------------------------------------------------
def endpoint_checks(U=4, d=64, seed=2):
    rng = np.random.default_rng(seed)
    print("\n=== Endpoint checks ===")
    # (i) high SNR: W* -> Gamma^{-1}
    beta = 0.4
    a = math.sqrt(beta) * np.ones(U)
    B = affinity_matrix(a)
    h = np.clip(np.abs((rng.standard_normal(U) + 1j * rng.standard_normal(U)) / math.sqrt(2)), 0.2, None)
    for rdb in (10, 40, 80):
        W, _ = lmmse_matrices(B, h, 10 ** (-rdb / 10), d)
        Ginv = np.linalg.inv(np.diag(1 / h) @ B @ np.diag(h))
        rel = np.linalg.norm(W - Ginv) / np.linalg.norm(Ginv)
        print(f"(i)   rho={rdb:2d} dB : ||W*-Gamma^-1||_F/||Gamma^-1||_F = {rel:.2e}")
    # (ii) beta=0: off-diagonal of W* vanishes, diagonal = Wiener
    W0, _ = lmmse_matrices(np.eye(U), h, 0.1, d)
    off = np.abs(W0 - np.diag(np.diag(W0))).max()
    wiener = h ** 2 / (h ** 2 + d * 0.1)
    diag_err = np.abs(np.diag(W0) - wiener).max()
    print(f"(ii)  beta=0    : max|offdiag W*| = {off:.2e}, max|diag - Wiener| = {diag_err:.2e}")
    # (iii) beta->1: W* row ~ rank-1 combining; compare to SC weights h^2-normalized
    a1 = math.sqrt(0.999) * np.ones(U)
    B1 = affinity_matrix(a1)
    W1, _ = lmmse_matrices(B1, h, 0.1, d)
    r = W1[0] * h  # undo the h_v/h_u structure: effective combining weights on h_v e_v looks
    r = np.abs(r) / np.abs(r).sum()
    mrc = h ** 2 / (h ** 2).sum()
    print(f"(iii) beta=.999 : normalized row-0 weights {np.round(r,3)} vs MRC {np.round(mrc,3)}")


# ----------------------------------------------------------------------
#  Figures
# ----------------------------------------------------------------------
def plot_all(rows_beta, rows_snr, rho_db, beta_mid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["MF (TIN)", "SR (AA-EDMA)", "SC (MRC)", "LMMSE (proposed)", "ODR (oracle)"]
    styles = {"MF (TIN)": ("0.6", ":", "v"),
              "SR (AA-EDMA)": ("tab:red", "--", "s"),
              "SC (MRC)": ("tab:green", "-.", "^"),
              "LMMSE (proposed)": ("tab:blue", "-", "o"),
              "ODR (oracle)": ("k", ":", "d")}

    betas = [b for b, _ in rows_beta]

    # Fig 1: NMSE vs beta + closed-form overlays
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for n in names:
        c, ls, mk = styles[n]
        ax.semilogy(betas, [r[n]["nmse"] for _, r in rows_beta], ls, color=c, marker=mk,
                    ms=4, label=n)
    ax.semilogy(betas, [r["LMMSE (proposed)"]["nmse_cf"] for _, r in rows_beta],
                'x', color="tab:blue", ms=9, mew=2, label="LMMSE closed form")
    ax.semilogy(betas, [r["SR (AA-EDMA)"]["nmse_cf"] for _, r in rows_beta],
                '+', color="tab:red", ms=10, mew=2, label="SR closed form")
    ax.set_xlabel(r"semantic affinity  $\beta$")
    ax.set_ylabel("NMSE")
    ax.set_title(f"NMSE vs affinity (U=4, d=64, SNR={rho_db} dB)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_nmse_vs_beta.png"), dpi=160)

    # Fig 2: cosine + SER vs beta
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for n in names:
        c, ls, mk = styles[n]
        axes[0].plot(betas, [r[n]["cos"] for _, r in rows_beta], ls, color=c, marker=mk, ms=4, label=n)
        axes[1].semilogy(betas, [max(r[n]["ser"], 1e-4) for _, r in rows_beta], ls, color=c, marker=mk, ms=4, label=n)
    axes[0].set_xlabel(r"$\beta$"); axes[0].set_ylabel("mean cosine"); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel(r"$\beta$"); axes[1].set_ylabel(f"SER (tau={TAU})"); axes[1].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Cosine recovery / SER vs affinity (U=4, d=64, SNR={rho_db} dB)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_cos_ser_vs_beta.png"), dpi=160)

    # Fig 3: SNR sweep at intermediate beta
    rhos = [r for r, _ in rows_snr]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for n in names:
        c, ls, mk = styles[n]
        ax.semilogy(rhos, [max(r[n]["ser"], 1e-4) for _, r in rows_snr], ls, color=c, marker=mk, ms=4, label=n)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel(f"SER (tau={TAU})")
    ax.set_title(f"SER vs SNR at intermediate affinity beta={beta_mid}")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_ser_vs_snr.png"), dpi=160)
    print(f"\nFigures saved to {OUT}")


if __name__ == "__main__":
    RHO_DB = 10
    BETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    print(f"=== beta sweep @ {RHO_DB} dB ===")
    rows_beta = run_sweep(BETAS, 10 ** (RHO_DB / 10))
    BETA_MID = 0.4
    print(f"\n=== SNR sweep @ beta={BETA_MID} ===")
    rows_snr = run_snr_sweep(BETA_MID, list(range(0, 21, 4)))
    endpoint_checks()
    plot_all(rows_beta, rows_snr, RHO_DB, BETA_MID)
