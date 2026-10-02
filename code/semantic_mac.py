"""
Core library for the TMC paper:
  "Structured Shared-Private Embedding Multiplexing for Semantic Multiple
   Access in Dynamic Mobile Networks"

Builds on the published shared-embedding multiple-access framework
(Lee, Choi, Lee, IEEE JSAC 2026, doi 10.1109/JSAC.2025.3643816).

Pipeline modeled here
---------------------
1. Content model (shared-scene decomposition), latent frame:
       z_u = a_u [c; 0] + sqrt(1-a_u^2) [0; p_u]        (structured latent)
   c in R^{d_c}: shared scene content, p_u in R^{d-d_c}: private content.
   A frozen foundation encoder outputs RAW embeddings x_u = R z_u with an
   unknown orthogonal mixing R (the encoder's arbitrary basis) -- the
   structure is present but hidden in the coordinates.

2. Matched-filter MAC front end (identical to the JSAC/AA-EDMA line):
       tilde_x_u = x_u + sum_{v!=u} beta_uv (h_v/h_u) x_v + n_u,
       Cov(n_u,n_w) = sigma^2 B_uw/(h_u h_w) I_d,  sigma^2 = 1/rho.

3. Receivers: SR (cancel), SC (combine), B-aware LMMSE (optimal linear,
   isotropic prior), and the decomposition receiver DR that knows the
   shared-subspace basis V_c: BLUE-combining on the shared block +
   SR cancellation and Wiener shrinkage on the private complement.

4. Embedding-structure optimization: closed-form spectral recovery of V_c
   from the cross-covariance of paired clean embeddings (GCCA-style), and a
   channel-in-the-loop learned linear adapter refining it.

5. Mobility: time-varying a_u(t) from user trajectories around a scene, and
   a decision-directed EWMA affinity tracker.
"""
from __future__ import annotations
import math
import numpy as np

TAU = 0.45  # SER threshold on cosine, fixed in advance and shared by every scheme

# ----------------------------------------------------------------------
#  Main configuration (Table IV). Every experiment, check and plotting
#  script imports these names instead of keeping its own copy, so a change
#  here moves the whole project at once (writing standard 7.9).
# ----------------------------------------------------------------------
U_MAIN = 4                       # users
D_SYN, DC_SYN = 64, 16           # synthetic contents (theory and mobility)
D_REAL, DC_REAL = 768, 128       # real BERT embeddings
SNR_MAIN_DB = 10                 # main operating point
SNR_E3_DB = 12                   # mobility experiments
K_PILOT, N_PILOT, L_D = 5, 64, 320   # pilot period, pilots per user, embeddings per slot
LAMBDA_EWMA = 0.3
N_PAIRS = 200                    # calibration pairs
MOB = dict(box=50.0, r_scene=32.0, a_max=0.95)   # +-box square, scene radius, peak share

try:                              # GPU by default, CPU fallback, one flag for all scripts
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:               # NumPy-only experiments never need torch
    torch = None
    DEVICE = "cpu"


def set_seed(seed: int):
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if str(DEVICE) == "cuda":
            torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
#  Affinity utilities
# ----------------------------------------------------------------------
def affinity_matrix(a: np.ndarray) -> np.ndarray:
    """B_uv = a_u a_v (u != v), B_uu = 1."""
    a = np.asarray(a, dtype=np.float64)
    B = np.outer(a, a)
    np.fill_diagonal(B, 1.0)
    return B


# ----------------------------------------------------------------------
#  Content generators
# ----------------------------------------------------------------------
def sample_latents_isotropic(batch, U, d, d_c, a, rng):
    """Structured latents with isotropic random contents (unit norm).
    Returns z of shape (batch, U, d): shared block = first d_c coords."""
    a = np.broadcast_to(np.asarray(a, float), (batch, U)) \
        if np.ndim(a) > 1 or np.ndim(a) == 1 else np.full((batch, U), float(a))
    if a.shape != (batch, U):
        a = np.broadcast_to(np.asarray(a, float), (batch, U))
    z = np.zeros((batch, U, d))
    c = rng.standard_normal((batch, d_c))
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    p = rng.standard_normal((batch, U, d - d_c))
    p /= np.linalg.norm(p, axis=2, keepdims=True)
    z[:, :, :d_c] = a[:, :, None] * c[:, None, :]
    z[:, :, d_c:] = np.sqrt(1.0 - a[:, :, None] ** 2) * p
    return z


class EmbeddingPool:
    """Real PLM embedding pool (e.g., BERT AG-News, 8000 x 768).
    Centered + unit-normalized; provides PCA coordinates so that structured
    latents can be built from real semantic content."""

    def __init__(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        self.mu = X.mean(axis=0, keepdims=True)
        Xc = X - self.mu
        Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
        self.X = Xc
        # PCA basis of the (centered, normalized) pool
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.pca = Vt            # (d, d) rows = principal directions
        self.spectrum = S ** 2 / len(Xc)
        self.N, self.d = Xc.shape

    def pca_coords(self, idx, k):
        """Top-k PCA coordinates of pool items idx, renormalized to unit."""
        Y = self.X[idx] @ self.pca[:k].T
        return Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)


def sample_latents_pool(pool: EmbeddingPool, batch, U, d, d_c, a, rng,
                        idx_pool=None):
    """Structured latents whose shared/private contents are REAL embeddings:
    shared c = top-d_c PCA coords of one pool sentence, private p_u =
    top-(d-d_c) PCA coords of distinct other sentences.

    idx_pool: optional index array restricting which pool sentences may be
    drawn (train/holdout partition); None draws from the whole pool."""
    a = np.broadcast_to(np.asarray(a, float), (U,))
    choices = np.arange(pool.N) if idx_pool is None else np.asarray(idx_pool)
    idx = np.array([rng.choice(choices, size=U + 1, replace=False)
                    for _ in range(batch)])
    c = pool.pca_coords(idx[:, 0], d_c)                     # (batch, d_c)
    z = np.zeros((batch, U, d))
    for u in range(U):
        p = pool.pca_coords(idx[:, u + 1], d - d_c)
        z[:, u, :d_c] = a[u] * c
        z[:, u, d_c:] = math.sqrt(1.0 - a[u] ** 2) * p
    return z


def random_orthogonal(d, rng):
    G = rng.standard_normal((d, d))
    Q, Rr = np.linalg.qr(G)
    Q *= np.sign(np.diag(Rr))
    return Q


# ----------------------------------------------------------------------
#  Matched-filter MAC front end (JSAC / AA-EDMA convention)
# ----------------------------------------------------------------------
H_MIN = 0.2   # deep-fade clip; E[1/h^2] diverges under unclipped Rayleigh


def matched_filter(e, B, rho, rng, fading=True, h_min=H_MIN, h_fixed=None):
    """tilde_u = sum_v (h_v/h_u) B_uv e_v + n_u,
    Cov(n_u,n_w) = (sigma^2 B_uw / (h_u h_w)) I_d.  Returns (tilde, h).

    h_min clips deep fades. The clip is required rather than cosmetic,
    because the fading-averaged error laws contain E[1/h_u^2], which
    diverges for unclipped Rayleigh fading. It is applied to the channel
    realization and therefore affects every receiver identically."""
    batch, U, d = e.shape
    sigma2 = 1.0 / rho
    B = np.asarray(B, dtype=np.float64)
    if B.ndim == 2:
        B = np.broadcast_to(B, (batch, U, U))
    if h_fixed is not None:
        h = np.broadcast_to(np.asarray(h_fixed, float), (batch, U)).copy()
    elif fading:
        hc = (rng.standard_normal((batch, U)) +
              1j * rng.standard_normal((batch, U))) / math.sqrt(2)
        h = np.clip(np.abs(hc), h_min, None)
    else:
        h = np.ones((batch, U))
    tilde = np.einsum('buv,bvd,bv,bu->bud', B, e, h, 1.0 / h)
    xi = rng.standard_normal((batch, U, d)) * math.sqrt(sigma2)
    for b in range(batch):
        A = np.linalg.cholesky(B[b])
        tilde[b] += (A @ xi[b]) / h[b][:, None]
    return tilde, h


# ----------------------------------------------------------------------
#  Receivers
# ----------------------------------------------------------------------
def demux_sr(tilde, B, h):
    """Similarity-rejecting closed-form demux: (Gamma^{-1} (x) I) tilde."""
    batch, U, d = tilde.shape
    out = np.empty_like(tilde)
    for b in range(batch):
        Gamma = np.diag(1.0 / h[b]) @ B @ np.diag(h[b])
        out[b] = np.linalg.solve(Gamma, tilde[b])
    return out


def demux_sc(tilde, h):
    """Similarity-combining endpoint: h^2-weighted MRC of all MF outputs."""
    w = h ** 2
    w = w / w.sum(axis=1, keepdims=True)
    comb = np.einsum('bu,bud->bd', w, tilde)
    return np.repeat(comb[:, None, :], tilde.shape[1], axis=1)


def lmmse_matrices(B, h, sigma2, d):
    H = np.diag(h)
    Hi = np.diag(1.0 / h)
    Gamma = Hi @ B @ H
    Cx = B / d
    Cn = sigma2 * (Hi @ B @ Hi)
    S = Gamma @ Cx @ Gamma.T + Cn
    W = np.linalg.solve(S.T, (Cx @ Gamma.T).T).T
    Eerr = Cx - W @ Gamma @ Cx
    return W, Eerr


def demux_lmmse(tilde, B, h, rho):
    """B-aware LMMSE (optimal linear receiver under isotropic prior).
    Returns (estimates, closed-form per-user total MSE averaged over batch)."""
    batch, U, d = tilde.shape
    sigma2 = 1.0 / rho
    out = np.empty_like(tilde)
    mse_cf = np.zeros(U)
    for b in range(batch):
        W, Eerr = lmmse_matrices(B, h[b], sigma2, d)
        out[b] = W @ tilde[b]
        mse_cf += d * np.diag(Eerr)
    return out, mse_cf / batch


def demux_dr(tilde, B, h, rho, a, Vc):
    """Decomposition receiver (proposed).

    Vc: (d, d_c) orthonormal basis of the shared subspace (from the
    embedding-structure optimizer; oracle = true mixing columns).
    Shared block: BLUE-combining of the U looks at the common content,
    followed by Wiener shrinkage.  Private complement: SR cancellation +
    per-user Wiener shrinkage.  Recombine."""
    batch, U, d = tilde.shape
    d_c = Vc.shape[1]
    sigma2 = 1.0 / rho
    a = np.broadcast_to(np.asarray(a, float), (U,))
    Binv = np.linalg.inv(B)
    out = np.empty_like(tilde)
    Zs = tilde @ Vc                          # (batch, U, d_c) shared-block obs
    Zp = tilde - (Zs @ Vc.T)                 # complement part (in ambient frame)
    for b in range(batch):
        hb = h[b]
        Hi = np.diag(1.0 / hb)
        Gamma = Hi @ B @ np.diag(hb)
        Cn = sigma2 * (Hi @ B @ Hi)
        gamma = np.array([
            a[u] + sum(B[u, v] * a[v] * hb[v] / hb[u]
                       for v in range(U) if v != u) for u in range(U)])
        Cn_inv = np.linalg.inv(Cn)
        denom = float(gamma @ Cn_inv @ gamma)
        if denom > 1e-12:
            c_hat = (gamma @ Cn_inv @ Zs[b]) / denom
            c_hat *= (1.0 / d_c) / (1.0 / d_c + 1.0 / denom)
        else:
            c_hat = np.zeros(d_c)
        Gp = np.linalg.solve(Gamma, Zp[b])   # SR on the complement
        for u in range(U):
            sig_p = 1.0 - a[u] ** 2
            err_p = (d - d_c) * sigma2 * Binv[u, u] / hb[u] ** 2
            shrink = sig_p / (sig_p + err_p) if sig_p > 0 else 0.0
            out[b, u] = a[u] * (Vc @ c_hat) + shrink * Gp[u]
    return out


# ----------------------------------------------------------------------
#  Conventional baselines (orthogonal and power-domain multiple access)
# ----------------------------------------------------------------------
def oma_observe(e, h, rho, rng):
    """Conventional orthogonal MA (OFDMA-style): user u is confined to a
    disjoint d/U-dimensional block and decoded only from that block, so it
    never sees cross-user interference but its recovery is capped by the
    1/U-energy subspace (cosine ceiling ~ sqrt(1/U))."""
    batch, U, d = e.shape
    blk = d // U
    sigma = math.sqrt(1.0 / rho)
    out = np.zeros_like(e)
    for u in range(U):
        sl = slice(u * blk, (u + 1) * blk)
        out[:, u, sl] = e[:, u, sl] + \
            rng.standard_normal((batch, blk)) * sigma / h[:, u][:, None]
    return out


def demux_noma_genie(e, h, rho, rng):
    """Genie-aided NOMA-SIC upper bound: every user decoded from an
    interference-free observation e_u + n/h_u at per-user SNR rho
    (perfect cancellation, no error propagation)."""
    batch, U, d = e.shape
    sigma = math.sqrt(1.0 / rho)
    return e + rng.standard_normal(e.shape) * sigma / h[:, :, None]


# ----------------------------------------------------------------------
#  Embedding-structure optimization
# ----------------------------------------------------------------------
def learn_structure_spectral(x_clean, d_c):
    """Closed-form shared-subspace recovery from N paired CLEAN embeddings.

    x_clean: (N, U, d) raw (mixed) embeddings of co-located users.
    The averaged symmetrized cross-covariance has column space equal to the
    shared subspace (private parts are independent and average out).
    Returns Vc_hat (d, d_c), eigenvalues (d,)."""
    N, U, d = x_clean.shape
    M = np.zeros((d, d))
    cnt = 0
    for u in range(U):
        for v in range(u + 1, U):
            C = x_clean[:, u, :].T @ x_clean[:, v, :] / N
            M += C + C.T
            cnt += 2
    M /= cnt
    w, V = np.linalg.eigh(M)
    order = np.argsort(w)[::-1]
    return V[:, order[:d_c]], w[order]


def subspace_error(Vhat, Vtrue):
    """Normalized projection-Frobenius distance in [0,1]."""
    P1 = Vhat @ Vhat.T
    P2 = Vtrue @ Vtrue.T
    k = Vtrue.shape[1]
    return float(np.linalg.norm(P1 - P2) / math.sqrt(2 * k))


def estimate_a_from_clean(x_clean, Vc):
    """a_u estimate from clean paired data: sqrt(mean shared-block energy)."""
    E = np.linalg.norm(x_clean @ Vc, axis=2) ** 2      # (N, U)
    return np.sqrt(np.clip(E.mean(axis=0), 0.0, 1.0))


# ----------------------------------------------------------------------
#  Mobility model and online affinity tracking
# ----------------------------------------------------------------------
def mobility_trajectories(U, T, speed, rng, box=60.0, r_scene=28.0,
                          a_max=0.95, dt=1.0):
    """Random-waypoint trajectories around a scene at the origin.
    Returns a_t of shape (T, U): a_u(t) = a_max * exp(-d_u(t)^2 / (2 r^2))."""
    pos = rng.uniform(-box, box, size=(U, 2))
    wp = rng.uniform(-box, box, size=(U, 2))
    a_t = np.zeros((T, U))
    for t in range(T):
        for u in range(U):
            vec = wp[u] - pos[u]
            dist = np.linalg.norm(vec)
            if dist < speed * dt:
                wp[u] = rng.uniform(-box, box, size=2)
            else:
                pos[u] += (speed * dt) * vec / dist
        d2 = (pos ** 2).sum(axis=1)
        a_t[t] = a_max * np.exp(-d2 / (2 * r_scene ** 2))
    return a_t


def pilot_affinity_obs(e_clean, h, rho, rng, a_cap=0.95):
    """Affinity observation from one orthogonal pilot slot.

    Each user transmits n_p clean embeddings on an interference-free pilot
    resource; the receiver observes e_u + n/h_u.  Off-diagonal Gram entries
    of the pilot observations are unbiased for beta_uv = a_u a_v (independent
    noises), so no bias correction is needed.  The share coefficients are
    then the rank-one alternating-least-squares fit of the off-diagonal
    Gram, which uses all pairs jointly."""
    n_p, U, d = e_clean.shape
    sigma = math.sqrt(1.0 / rho)
    y = e_clean + rng.standard_normal(e_clean.shape) * sigma / h[None, :, None]
    G = np.einsum('bud,bvd->buv', y, y).mean(axis=0)
    # rank-1 least-squares fit of the off-diagonal Gram: G_uv ~ a_u a_v.
    # Alternating least squares; uses all pairs jointly (no small-denominator
    # amplification, robust in low-affinity regimes).
    mask = ~np.eye(U, dtype=bool)
    a = np.sqrt(np.clip(np.abs(G[mask]).reshape(U, U - 1).mean(axis=1),
                        1e-4, a_cap ** 2))
    for _ in range(20):
        for u in range(U):
            others = [v for v in range(U) if v != u]
            num = sum(G[u, v] * a[v] for v in others)
            den = sum(a[v] ** 2 for v in others) + 1e-9
            a[u] = np.clip(num / den, 0.0, a_cap)
    return a


class AffinityTracker:
    """EWMA tracker of the per-user share coefficients, driven by sparse
    orthogonal affinity-pilot observations (every K-th slot)."""

    def __init__(self, U, lam=0.5, a_init=0.3):
        self.a = np.full(U, float(a_init))
        self.lam = lam

    def update(self, obs):
        self.a = (1 - self.lam) * self.a + self.lam * obs
        return self.a.copy()


# ----------------------------------------------------------------------
#  Metrics
# ----------------------------------------------------------------------
def metrics(e_hat, e_true, tau=TAU):
    """(mean cosine, NMSE of raw estimate, SER). e_true unit-norm."""
    nmse = ((e_hat - e_true) ** 2).sum(-1).mean()
    e_n = e_hat / (np.linalg.norm(e_hat, axis=2, keepdims=True) + 1e-12)
    cos = (e_n * e_true).sum(-1)
    return float(cos.mean()), float(nmse), float((cos < tau).mean())
