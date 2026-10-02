"""E5: comparison with a trained user-wise attention receiver.

A learned receiver representative of the end-to-end line (per-user query
attention over the matched-filter outputs, residual skip, unit
normalization) is trained at the operating SNR over uniformly random
affinities, then compared against the closed-form LMMSE and DR receivers
across the affinity sweep. The learned receiver receives no affinity side
information and must infer the coupling from data, which is the standard
setting of the learned line.

Outputs: data/e5_learned.csv, data/e5_train_log.csv
(figures come from replot_all.py only)
"""
import math
import os
import numpy as np
import torch

from semantic_mac import (affinity_matrix, matched_filter,
                          sample_latents_isotropic, demux_lmmse, demux_dr,
                          demux_sr, demux_sc, metrics)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "fig")
DATA = os.path.join(HERE, "..", "data")

from semantic_mac import U_MAIN as U, D_SYN as D, DC_SYN as DC  # Table IV
SNR_DB = 10
RHO = 10 ** (SNR_DB / 10)
STEPS = 3000
BATCH = 64


class UserWiseAttention(torch.nn.Module):
    """Per-user query attention over the U matched-filter outputs."""

    def __init__(self, U, d, dk=16, heads=4):
        super().__init__()
        self.U, self.d, self.dk, self.H = U, d, dk, heads
        self.WK = torch.nn.Linear(d, dk * heads, bias=False)
        self.WV = torch.nn.Linear(d, dk * heads, bias=False)
        self.WO = torch.nn.Linear(dk * heads, d, bias=False)
        self.q = torch.nn.Parameter(torch.randn(U, heads, dk) * 0.1)
        self.log_eta = torch.nn.Parameter(torch.zeros(()))

    def forward(self, tilde):
        B, Uu, d = tilde.shape
        K = self.WK(tilde).view(B, Uu, self.H, self.dk)
        V = self.WV(tilde).view(B, Uu, self.H, self.dk)
        sc = torch.einsum('uhk,bihk->buih', self.q, K) / math.sqrt(self.dk)
        alpha = torch.softmax(torch.exp(self.log_eta) * sc, dim=2)
        ctx = torch.einsum('buih,bihk->buhk', alpha, V).reshape(B, Uu, -1)
        out = self.WO(ctx) + tilde
        return out / (out.norm(dim=2, keepdim=True) + 1e-12)


def gen_batch(rng, batch, beta):
    a = math.sqrt(beta) * np.ones(U)
    B = affinity_matrix(a)
    z = sample_latents_isotropic(batch, U, D, DC, a, rng)
    tilde, h = matched_filter(z, B, RHO, rng)
    return z, tilde, h, a, B


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(21)
    net = UserWiseAttention(U, D)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"learned receiver parameters: {n_par}")
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    train_log = []
    # fixed validation batch at beta=0.5 (dedicated RNG, so the training
    # stream is untouched); evaluated every 100 steps as convergence
    # evidence for the fixed training budget
    rng_v = np.random.default_rng(777)
    z_v, tilde_v, h_v, a_v, B_v = gen_batch(rng_v, 2000, 0.5)
    zv_t = torch.tensor(z_v, dtype=torch.float32)
    tv_t = torch.tensor(tilde_v, dtype=torch.float32)
    val_log = []
    for it in range(STEPS):
        beta = float(rng.uniform(0.05, 0.9))
        z, tilde, h, a, B = gen_batch(rng, BATCH, beta)
        zt = torch.tensor(z, dtype=torch.float32)
        tt = torch.tensor(tilde, dtype=torch.float32)
        out = net(tt)
        loss = (1.0 - (out * zt).sum(-1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % 50 == 0:
            train_log.append((it + 1, float(loss.item())))
        if (it + 1) % 100 == 0:
            with torch.no_grad():
                ov = net(tv_t)
                vloss = float((1.0 - (ov * zv_t).sum(-1)).mean())
            val_log.append((it + 1, vloss))
        if (it + 1) % 500 == 0:
            print(f"  step {it+1}: loss={loss.item():.4f} "
                  f"val={val_log[-1][1]:.4f}")
    with open(os.path.join(DATA, "e5_train_log.csv"), "w") as f:
        f.write("step,loss\n")
        for st, lo in train_log:
            f.write(f"{st},{lo}\n")
    # fixed-beta validation curve (convergence evidence)
    with open(os.path.join(DATA, "e5_valcurve.csv"), "w") as f:
        f.write("step,val_loss\n")
        for st, lo in val_log:
            f.write(f"{st},{lo}\n")

    # evaluation across the affinity sweep
    betas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    Vc = np.eye(D)[:, :DC]
    rows = []
    net.eval()
    rng_e = np.random.default_rng(500)
    for beta in betas:
        z, tilde, h, a, B = gen_batch(rng_e, 4000, beta)
        with torch.no_grad():
            out_l = net(torch.tensor(tilde, dtype=torch.float32)).numpy()
        r = {}
        r["Learned"] = metrics(out_l.astype(np.float64), z)
        lm, _ = demux_lmmse(tilde, B, h, RHO)
        r["LMMSE"] = metrics(lm, z)
        r["DR"] = metrics(demux_dr(tilde, B, h, RHO, a, Vc), z)
        rows.append(r)
        print(f"beta={beta:.2f} " + " ".join(
            f"{k}:cos={v[0]:.3f},ser={v[2]:.3f}" for k, v in r.items()))
    with open(os.path.join(DATA, "e5_learned.csv"), "w") as f:
        keys = ["Learned", "LMMSE", "DR"]
        f.write("beta," + ",".join(f"{k}_cos,{k}_nmse,{k}_ser" for k in keys)
                + "\n")
        for b, r in zip(betas, rows):
            f.write(f"{b}," + ",".join(
                f"{r[k][0]},{r[k][1]},{r[k][2]}" for k in keys) + "\n")

    # figures are produced only by the canonical replot_all.py (uniform
    # geometry); experiment scripts write CSVs exclusively.
    print("E5 done. Run replot_all.py to regenerate the figures.")


if __name__ == "__main__":
    main()
