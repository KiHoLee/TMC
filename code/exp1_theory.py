"""E1: receiver theory verification (synthetic isotropic contents).

Outputs: data/e1_beta.csv, data/e1_snr.csv, data/e1_endpoints.txt
(figures come from replot_all.py only)
"""
import math
import os
import numpy as np
from semantic_mac import (affinity_matrix, matched_filter, demux_sr, demux_sc,
                          demux_lmmse, demux_dr, lmmse_matrices,
                          sample_latents_isotropic, metrics,
                          oma_observe, demux_noma_genie)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "fig")
DATA = os.path.join(HERE, "..", "data")
os.makedirs(FIG, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

U, D, DC = 4, 64, 16
BATCH = 4000
NAMES = ["OMA", "NOMA", "SR", "SC", "LMMSE", "DR"]


def run_point(beta, rho, rng, conv_seed=0):
    a = math.sqrt(beta) * np.ones(U)
    B = affinity_matrix(a)
    z = sample_latents_isotropic(BATCH, U, D, DC, a, rng)
    tilde, h = matched_filter(z, B, rho, rng)
    Vc = np.eye(D)[:, :DC]
    res = {}
    res["SR"] = metrics(demux_sr(tilde, B, h), z)
    res["SC"] = metrics(demux_sc(tilde, h), z)
    lm, cf = demux_lmmse(tilde, B, h, rho)
    res["LMMSE"] = metrics(lm, z)
    res["LMMSE_cf"] = float(cf.mean())
    # SR closed form: d sigma^2 [B^-1]_uu / h^2 averaged
    Binv = np.diag(np.linalg.inv(B))
    res["SR_cf"] = float((D / rho) * (Binv[None, :] / h ** 2).mean())
    res["DR"] = metrics(demux_dr(tilde, B, h, rho, a, Vc), z)
    # conventional baselines on a dedicated stream (keeps main draws intact)
    rng_c = np.random.default_rng(90000 + conv_seed)
    res["OMA"] = metrics(oma_observe(z, h, rho, rng_c), z)
    res["NOMA"] = metrics(demux_noma_genie(z, h, rho, rng_c), z)
    return res


def main():
    rng = np.random.default_rng(0)
    rho_db = 10
    rho = 10 ** (rho_db / 10)
    betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    rows = []
    for b in betas:
        r = run_point(b, rho, rng, conv_seed=int(b * 100))
        rows.append(r)
        print(f"beta={b:4.2f} " + " ".join(
            f"{n}:cos={r[n][0]:.3f},nmse={r[n][1]:.3f},ser={r[n][2]:.3f}"
            for n in NAMES))
    with open(os.path.join(DATA, "e1_beta.csv"), "w") as f:
        f.write("beta," + ",".join(f"{n}_cos,{n}_nmse,{n}_ser" for n in NAMES)
                + ",LMMSE_cf,SR_cf\n")
        for b, r in zip(betas, rows):
            f.write(f"{b}," + ",".join(
                f"{r[n][0]},{r[n][1]},{r[n][2]}" for n in NAMES)
                + f",{r['LMMSE_cf']},{r['SR_cf']}\n")

    beta_mid = 0.4
    snrs = list(range(0, 21, 4))
    rows_s = []
    for s in snrs:
        r = run_point(beta_mid, 10 ** (s / 10), rng, conv_seed=1000 + s)
        rows_s.append(r)
        print(f"snr={s} " + " ".join(f"{n}:ser={r[n][2]:.3f}" for n in NAMES))
    with open(os.path.join(DATA, "e1_snr.csv"), "w") as f:
        f.write("snr," + ",".join(f"{n}_cos,{n}_nmse,{n}_ser" for n in NAMES) + "\n")
        for s, r in zip(snrs, rows_s):
            f.write(f"{s}," + ",".join(
                f"{r[n][0]},{r[n][1]},{r[n][2]}" for n in NAMES) + "\n")

    # endpoint checks
    lines = []
    a = math.sqrt(0.4) * np.ones(U)
    B = affinity_matrix(a)
    h = np.clip(np.abs((rng.standard_normal(U) + 1j * rng.standard_normal(U))
                       / math.sqrt(2)), 0.2, None)
    for rdb in (10, 40, 80):
        W, _ = lmmse_matrices(B, h, 10 ** (-rdb / 10), D)
        Ginv = np.linalg.inv(np.diag(1 / h) @ B @ np.diag(h))
        rel = np.linalg.norm(W - Ginv) / np.linalg.norm(Ginv)
        lines.append(f"(i) rho={rdb}dB rel_diff_W_vs_Gammainv={rel:.3e}")
    W0, _ = lmmse_matrices(np.eye(U), h, 0.1, D)
    off = np.abs(W0 - np.diag(np.diag(W0))).max()
    wiener = h ** 2 / (h ** 2 + D * 0.1)
    lines.append(f"(ii) beta=0 max_offdiag={off:.3e} "
                 f"max_diag_minus_wiener={np.abs(np.diag(W0)-wiener).max():.3e}")
    with open(os.path.join(DATA, "e1_endpoints.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # figures are produced only by the canonical replot_all.py (uniform
    # geometry); experiment scripts write CSVs exclusively.
    print("E1 done. Run replot_all.py to regenerate the figures.")


if __name__ == "__main__":
    main()
