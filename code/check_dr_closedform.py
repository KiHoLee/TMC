"""Monte Carlo check of the DR error law, Proposition 2 / eq. (drmse).

The fading average of the conditional closed form is computed on channel
draws from a generator of its own, seeded after the E1 model, and compared
with the measured DR NMSE stored in data/e1_beta.csv at every affinity of
the E1 sweep. The agreement level printed here is the one the manuscript
quotes (writing standard 6.6, 6.7). Output: data/e1_dr_cf.csv
"""
import csv
import math
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
U, D, DC = 4, 64, 16          # E1 configuration
RHO_DB = 10
N_DRAWS = 20000
H_MIN = 0.2


def affinity_matrix(a):
    B = np.outer(a, a)
    np.fill_diagonal(B, 1.0)
    return B


def dr_closed_form(a, rho, rng):
    """Fading average of eq. (drmse) over clipped Rayleigh gains."""
    B = affinity_matrix(a)
    Binv = np.linalg.inv(B)
    sigma2 = 1.0 / rho
    tot = np.zeros(U)
    for _ in range(N_DRAWS):
        h = np.clip(np.abs((rng.standard_normal(U) + 1j * rng.standard_normal(U))
                           / math.sqrt(2)), H_MIN, None)
        Hi = np.diag(1.0 / h)
        Cn = sigma2 * (Hi @ B @ Hi)
        gamma = np.array([a[u] + sum(B[u, v] * a[v] * h[v] / h[u]
                                     for v in range(U) if v != u) for u in range(U)])
        denom = float(gamma @ np.linalg.solve(Cn, gamma))
        vs = 1.0 / denom if denom > 1e-12 else np.inf
        for u in range(U):
            Vp = (D - DC) * sigma2 * Binv[u, u] / h[u] ** 2
            shared = a[u] ** 2 * (DC * vs) / (1 + DC * vs) if np.isfinite(vs) else a[u] ** 2
            priv = (1 - a[u] ** 2) * Vp / ((1 - a[u] ** 2) + Vp) if (1 - a[u] ** 2) > 0 else 0.0
            tot[u] += shared + priv
    return float((tot / N_DRAWS).mean())      # ||x_u|| = 1, so MSE = NMSE


def main():
    rows = list(csv.DictReader(open(os.path.join(DATA, "e1_beta.csv"))))
    rng = np.random.default_rng(777001)       # own generator, seeded after the model
    rho = 10 ** (RHO_DB / 10)
    out, worst = [], 0.0
    for r in rows:
        beta = float(r["beta"])
        a = math.sqrt(beta) * np.ones(U)
        cf = dr_closed_form(a, rho, rng)
        mc = float(r["DR_nmse"])
        out.append((beta, mc, cf, abs(mc - cf)))
        worst = max(worst, abs(mc - cf))
        print(f"beta={beta:4.2f} DR_nmse(MC)={mc:.5f} closed form={cf:.5f} |diff|={abs(mc-cf):.2e}")
    with open(os.path.join(DATA, "e1_dr_cf.csv"), "w") as f:
        f.write("beta,DR_nmse_mc,DR_cf,abs_diff\n")
        for row in out:
            f.write(",".join(f"{v:.6g}" for v in row) + "\n")
    sr = [abs(float(r["SR_nmse"]) - float(r["SR_cf"])) / float(r["SR_cf"]) for r in rows]
    print(f"max |DR_nmse - DR_cf| over the sweep: {worst:.2e}")
    print(f"max relative |SR_nmse - SR_cf|/SR_cf: {max(sr):.2%}")


if __name__ == "__main__":
    main()
