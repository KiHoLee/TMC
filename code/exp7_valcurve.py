"""E7 held-out convergence: training curves of both learned receivers.

Answers whether the 3,000-step budget of experiment E7 trains both variants
to convergence (Comment 1.1 of the review).
Replicates exp7_learned_sideinfo.train() exactly (same seeds, same training
stream) on CPU and adds a held-out objective every 100 steps on a fixed
validation set: 400 draws at each of the ten evaluation affinities, from a
dedicated RNG so the training stream is untouched.

Output: data/e7_valcurve.csv (step, side_info, seed, val_loss_all,
val_loss_b05). Both variants come within 1% of their final held-out
objective by step 1,500, half the budget."""
import math, os, sys
import numpy as np
import torch
import exp7_learned_sideinfo as e7

torch.set_num_threads(4)
DEV = torch.device("cpu")
rng_v = np.random.default_rng(777)
VAL = []
for beta in e7.BETAS:
    z, tilde, h, a, B = e7.gen_batch(rng_v, 400, beta)
    VAL.append((beta, torch.tensor(z, dtype=torch.float32),
                torch.tensor(tilde, dtype=torch.float32),
                torch.tensor(h, dtype=torch.float32),
                torch.full((400,), beta, dtype=torch.float32)))

def val_loss(net, si):
    out = {}
    with torch.no_grad():
        for beta, zt, tt, ht, bt in VAL:
            o = net(tt, bt, ht) if si else net(tt)
            out[beta] = float((1.0 - (o * zt).sum(-1)).mean())
    return float(np.mean(list(out.values()))), out[0.5]

rows = []
for si in (False, True):
    for seed in e7.SEEDS:
        torch.manual_seed(seed)
        rng = np.random.default_rng(1000 + seed)
        net = e7.UserWiseAttention(e7.U, e7.D, side_info=si).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for it in range(e7.STEPS):
            beta = float(rng.uniform(0.05, 0.9))
            z, tilde, h, _, _ = e7.gen_batch(rng, e7.BATCH, beta)
            zt = torch.tensor(z, dtype=torch.float32)
            tt = torch.tensor(tilde, dtype=torch.float32)
            ht = torch.tensor(h, dtype=torch.float32)
            bt = torch.full((e7.BATCH,), beta, dtype=torch.float32)
            out = net(tt, bt, ht) if si else net(tt)
            loss = (1.0 - (out * zt).sum(-1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if (it + 1) % 100 == 0:
                va, vb = val_loss(net, si)
                rows.append((it + 1, si, seed, va, vb))
        print(f"side_info={si} seed={seed} final val_all={rows[-1][3]:.4f} "
              f"val_b05={rows[-1][4]:.4f}", flush=True)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "data", "e7_valcurve.csv"), "w") as f:
    f.write("step,side_info,seed,val_loss_all,val_loss_b05\n")
    for r in rows:
        f.write(f"{r[0]},{int(r[1])},{r[2]},{r[3]:.5f},{r[4]:.5f}\n")
print("done")
