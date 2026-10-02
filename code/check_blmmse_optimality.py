"""Optimality check for the subspace-aware (block) LMMSE receiver of E6.

Two facts are verified numerically, both of which the manuscript states.

1. The private branch filter Wp = Cp Gamma^T (Gamma Cp Gamma^T + Cn)^-1 is
   the MMSE filter on the complement block: no random perturbation of it
   lowers the mean-square error.
2. The zero-forcing plus scalar-Wiener filter that the DR uses on the same
   block is within a small margin of Wp, and the margin is reported.

3. The DR shared branch, namely the BLUE followed by the scalar Wiener
   shrinkage, equals the MMSE estimator of the common content exactly, so
   the whole gap between the DR and the block LMMSE lives in the private
   branch.

Output: printed table, stored in data/e6_optimality.txt by the caller.
"""
import sys, math
import numpy as np
sys.path.insert(0, ".")
from semantic_mac import affinity_matrix

from semantic_mac import U_MAIN as U, D_SYN as D, DC_SYN as DC  # Table IV
d_p = D - DC
rng = np.random.default_rng(31337)
for beta, rho_db in ((0.3, 10), (0.5, 10), (0.4, 4)):
    a = math.sqrt(beta)*np.ones(U); B = affinity_matrix(a); rho = 10**(rho_db/10)
    sigma2 = 1.0/rho
    hb = np.clip(np.abs((rng.standard_normal(U)+1j*rng.standard_normal(U))/math.sqrt(2)), 0.2, None)
    Hi = np.diag(1/hb); Gamma = Hi@B@np.diag(hb); Cn = sigma2*(Hi@B@Hi)
    Cp = np.diag((1-a**2)/d_p)
    # draw private parts directly in the complement block
    Nb = 20000
    Xp = rng.standard_normal((Nb, U, d_p))
    Xp /= np.linalg.norm(Xp, axis=2, keepdims=True)
    Xp *= np.sqrt(1-a**2)[None, :, None]
    L = np.linalg.cholesky(Cn)
    nz = rng.standard_normal((Nb, U, d_p))
    Zp = np.einsum('uv,bvd->bud', Gamma, Xp) + np.einsum('uv,bvd->bud', L, nz)
    S = Gamma@Cp@Gamma.T + Cn
    Wp = np.linalg.solve(S.T, (Cp@Gamma.T).T).T
    def mse(W):
        est = np.einsum('uv,bvd->bud', W, Zp)
        return float(((est-Xp)**2).sum(-1).mean())
    base = mse(Wp)
    # zero-forcing + per-user scalar Wiener shrink (what the DR does)
    Binv = np.linalg.inv(B)
    Gi = np.linalg.inv(Gamma)
    sh = np.array([(1-a[u]**2)/((1-a[u]**2)+d_p*sigma2*Binv[u,u]/hb[u]**2) for u in range(U)])
    Wdr = np.diag(sh)@Gi
    # the perturbation check draws from its own generator, seeded after the
    # model is built, so it does not inherit the model's draw
    rng_chk = np.random.default_rng(31338)
    better = sum(1 for _ in range(4000)
                 if mse(Wp + 0.03*np.linalg.norm(Wp)*rng_chk.standard_normal((U,U))/U) < base-1e-12)
    print(f"beta={beta} {rho_db}dB | MMSE filter {base:.5f} | DR filter {mse(Wdr):.5f} "
          f"({100*(mse(Wdr)/base-1):+.2f}%) | perturbations beating MMSE: {better}/4000")

# ---- 3. DR shared branch equals the MMSE shared estimator exactly -------
print()
for beta, rho_db in ((0.3, 10), (0.5, 10), (0.4, 4)):
    a = math.sqrt(beta) * np.ones(U)
    B = affinity_matrix(a)
    sigma2 = 1.0 / 10 ** (rho_db / 10)
    hb = np.clip(np.abs((rng.standard_normal(U) + 1j * rng.standard_normal(U))
                        / math.sqrt(2)), 0.2, None)
    Hi = np.diag(1 / hb)
    Cn = sigma2 * (Hi @ B @ Hi)
    Cn_inv = np.linalg.inv(Cn)
    gamma = np.array([a[u] + sum(B[u, v] * a[v] * hb[v] / hb[u]
                                 for v in range(U) if v != u) for u in range(U)])
    den = float(gamma @ Cn_inv @ gamma)
    Zs = rng.standard_normal((U, DC))          # arbitrary observation
    blue = (gamma @ Cn_inv @ Zs) / den
    dr_shared = blue * ((1 / DC) / ((1 / DC) + (1 / den)))
    mmse_shared = (gamma @ Cn_inv @ Zs) / (den + DC)
    rel = np.linalg.norm(dr_shared - mmse_shared) / np.linalg.norm(mmse_shared)
    print(f"beta={beta} {rho_db}dB | DR shared branch vs MMSE shared branch: "
          f"relative difference {rel:.2e}")
