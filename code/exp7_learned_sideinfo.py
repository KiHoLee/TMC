"""E7: learned attention receiver with matched side information
(Reviewers 1.1 and 3.4).

Reviewer 1 observed that the learned attention receiver of E5 receives no
affinity and no channel gains while the DR receives both, so the original
comparison did not isolate the effect of learning. Reviewer 3 raised the
same point. This experiment adds a side-informed variant that is given
exactly the affinity and the channel gains, trained under the same budget
as the original variant, and reports both.

Receivers compared
------------------
  Learned        per-user query attention over the matched-filter
                 outputs, no side information (the E5 receiver)
  Learned+SI     the same network, with the affinity and the channel
                 gains supplied through a context encoder that modulates
                 the per-user queries, and with explicit affinity and
                 channel-ratio biases on the attention scores
  LMMSE          affinity-aware LMMSE of Proposition 1, filters of the
                 form (F (x) I_d)
  BLMMSE         subspace-aware LMMSE of E6, same side information as DR
  DR             the proposed decomposition receiver
  OMA, NOMA-SIC  conventional references

Both learned variants are trained over several seeds and reported as
mean and standard deviation, so the comparison is not read off a single
training run.

GPU is used when available (AINS server), with a CPU fallback.

Outputs: data/e7_learned_sideinfo.csv, data/e7_train_log.txt
"""
import math
import os
import numpy as np
import torch

from semantic_mac import (affinity_matrix, matched_filter, demux_lmmse,
                          demux_dr, sample_latents_isotropic, metrics,
                          oma_observe, demux_noma_genie)
from exp6_sideinfo import demux_block_lmmse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)

from semantic_mac import DEVICE  # CUDA when available, CPU fallback

from semantic_mac import U_MAIN as U, D_SYN as D, DC_SYN as DC  # Table IV
SNR_DB = 10
RHO = 10 ** (SNR_DB / 10)
STEPS = 3000
BATCH = 64
SEEDS = [0, 1, 2, 3, 4]
BETAS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
EVAL_BATCH = 40000   # 1.6e5 per-user decisions, so an SER of
                     # 1e-4 is resolved and no curve is clipped


class UserWiseAttention(torch.nn.Module):
    """Per-user query attention over the U matched-filter outputs.

    With side_info=True the per-user queries are modulated by a context
    encoder reading the affinity and the channel gains, and the attention
    scores carry an explicit affinity bias and channel-ratio bias, so the
    network is given the same quantities the DR uses."""

    def __init__(self, U, d, dk=16, heads=4, side_info=False):
        super().__init__()
        self.U, self.d, self.dk, self.H = U, d, dk, heads
        self.side_info = side_info
        self.WK = torch.nn.Linear(d, dk * heads, bias=False)
        self.WV = torch.nn.Linear(d, dk * heads, bias=False)
        self.WO = torch.nn.Linear(dk * heads, d, bias=False)
        self.q = torch.nn.Parameter(torch.randn(U, heads, dk) * 0.1)
        self.log_eta = torch.nn.Parameter(torch.zeros(()))
        if side_info:
            # context: affinity plus the per-user gain and the mean gain
            self.ctx = torch.nn.Sequential(
                torch.nn.Linear(3, 32), torch.nn.ReLU(),
                torch.nn.Linear(32, heads * dk))
            self.b_aff = torch.nn.Parameter(torch.zeros(()))
            self.b_gain = torch.nn.Parameter(torch.zeros(()))

    def forward(self, tilde, beta=None, h=None):
        B, Uu, d = tilde.shape
        K = self.WK(tilde).view(B, Uu, self.H, self.dk)
        V = self.WV(tilde).view(B, Uu, self.H, self.dk)
        q = self.q.unsqueeze(0).expand(B, -1, -1, -1)          # (B,U,H,dk)
        sc = torch.einsum('buhk,bihk->buih', q, K) / math.sqrt(self.dk)
        if self.side_info:
            hn = h / (h.mean(dim=1, keepdim=True) + 1e-9)      # (B,U)
            ctx_in = torch.stack(
                [beta.unsqueeze(1).expand(-1, Uu), hn,
                 hn.mean(dim=1, keepdim=True).expand(-1, Uu)], dim=2)
            dq = self.ctx(ctx_in).view(B, Uu, self.H, self.dk)
            sc = sc + torch.einsum('buhk,bihk->buih', dq, K) / math.sqrt(self.dk)
            off = 1.0 - torch.eye(Uu, device=tilde.device)
            aff = beta.view(B, 1, 1, 1) * off.view(1, Uu, Uu, 1)
            ratio = torch.log(h.view(B, 1, Uu, 1) / h.view(B, Uu, 1, 1) + 1e-9)
            sc = sc + self.b_aff * aff + self.b_gain * ratio
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


def train(side_info, seed, log):
    torch.manual_seed(seed)
    rng = np.random.default_rng(1000 + seed)
    net = UserWiseAttention(U, D, side_info=side_info).to(DEVICE)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for it in range(STEPS):
        beta = float(rng.uniform(0.05, 0.9))
        z, tilde, h, _, _ = gen_batch(rng, BATCH, beta)
        zt = torch.tensor(z, dtype=torch.float32, device=DEVICE)
        tt = torch.tensor(tilde, dtype=torch.float32, device=DEVICE)
        ht = torch.tensor(h, dtype=torch.float32, device=DEVICE)
        bt = torch.full((BATCH,), beta, dtype=torch.float32, device=DEVICE)
        out = net(tt, bt, ht) if side_info else net(tt)
        loss = (1.0 - (out * zt).sum(-1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % 1000 == 0:
            log.append(f"  side_info={side_info} seed={seed} step={it+1} "
                       f"loss={loss.item():.4f}")
    return net, n_par


def main():
    log = [f"device={DEVICE}"]
    print(log[0])
    nets = {}
    for si in (False, True):
        for seed in SEEDS:
            net, n_par = train(si, seed, log)
            net.eval()
            nets[(si, seed)] = net
            msg = (f"trained side_info={si} seed={seed} params={n_par}")
            log.append(msg)
            print(msg)

    Vc = np.eye(D)[:, :DC]
    rows = []
    for beta in BETAS:
        rng_e = np.random.default_rng(500 + int(beta * 1000))
        z, tilde, h, a, B = gen_batch(rng_e, EVAL_BATCH, beta)
        tt = torch.tensor(tilde, dtype=torch.float32, device=DEVICE)
        ht = torch.tensor(h, dtype=torch.float32, device=DEVICE)
        bt = torch.full((EVAL_BATCH,), beta, dtype=torch.float32, device=DEVICE)
        r = {}
        for si, name in ((False, "Learned"), (True, "LearnedSI")):
            sers, coss = [], []
            for seed in SEEDS:
                with torch.no_grad():
                    o = nets[(si, seed)](tt, bt, ht) if si else nets[(si, seed)](tt)
                c, _, s = metrics(o.cpu().numpy().astype(np.float64), z)
                sers.append(s)
                coss.append(c)
            r[name] = (float(np.mean(coss)), float(np.std(coss)),
                       float(np.mean(sers)), float(np.std(sers)))
        lm, _ = demux_lmmse(tilde, B, h, RHO)
        for name, est in (("LMMSE", lm),
                          ("BLMMSE", demux_block_lmmse(tilde, B, h, RHO, a, Vc)),
                          ("DR", demux_dr(tilde, B, h, RHO, a, Vc))):
            c, _, s = metrics(est, z)
            r[name] = (c, 0.0, s, 0.0)
        rng_c = np.random.default_rng(70000 + int(beta * 100))
        for name, est in (("OMA", oma_observe(z, h, RHO, rng_c)),
                          ("NOMA", demux_noma_genie(z, h, RHO, rng_c))):
            c, _, s = metrics(est, z)
            r[name] = (c, 0.0, s, 0.0)
        rows.append(r)
        print(f"beta={beta:.2f} " + " ".join(
            f"{k}:ser={v[2]:.4f}" for k, v in r.items()))

    keys = ["OMA", "NOMA", "LMMSE", "Learned", "LearnedSI", "BLMMSE", "DR"]
    with open(os.path.join(DATA, "e7_learned_sideinfo.csv"), "w") as f:
        f.write("beta," + ",".join(
            f"{k}_cos,{k}_cos_sd,{k}_ser,{k}_ser_sd" for k in keys) + "\n")
        for b, r in zip(BETAS, rows):
            f.write(f"{b}," + ",".join(
                f"{r[k][0]},{r[k][1]},{r[k][2]},{r[k][3]}" for k in keys) + "\n")
    with open(os.path.join(DATA, "e7_train_log.txt"), "w") as f:
        f.write("\n".join(log) + "\n")
    print("E7 done.")


if __name__ == "__main__":
    main()
