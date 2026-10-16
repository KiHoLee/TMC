"""Runtimes quoted in the last paragraph of Section VII-A (Comment 5.10).

Measures single-threaded CPU times of the spectral calibration, the adapter
training and experiment E1. The GPU figure (five-seed learned comparison on
an RTX A4500) is not measured here. NOTE: running it re-executes E1 and the
E2 calibration, which rewrite their data files with identical contents.

Output: stdout, saved as data/runtimes.txt.
Run:  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 runtime_check.py
"""
import math, os, platform, sys, time, io, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
torch.set_num_threads(1)
import exp2_structure as e2, exp1_theory as e1
from semantic_mac import EmbeddingPool, affinity_matrix, random_orthogonal, learn_structure_spectral

print("machine:", platform.processor(), platform.platform(), "threads: torch", torch.get_num_threads(),
      "OMP", os.environ.get("OMP_NUM_THREADS"), "| numpy", np.__version__, "torch", torch.__version__)
rng = np.random.default_rng(7)
obj = torch.load(e2.POOL_PT, map_location="cpu", weights_only=False)
X = obj.numpy() if torch.is_tensor(obj) else np.asarray(obj)
pool = EmbeddingPool(X)
a = math.sqrt(e2.BETA) * np.ones(e2.U); R = random_orthogonal(e2.D, rng)
_, x_cal = e2.gen_clean(pool, e2.N_PAIR, a, R, rng, idx_pool=e2.IDX_TRAIN)
t = time.perf_counter(); V_spec, _ = learn_structure_spectral(x_cal, e2.DC); t_spec = time.perf_counter() - t
print(f"spectral calibration, N={e2.N_PAIR}, d={e2.D}: {t_spec:.3f} s")
Q, _ = np.linalg.qr(np.hstack([V_spec, rng.standard_normal((e2.D, e2.D - e2.DC))]))
t = time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()):
    e2.train_adapter(x_cal, a, Q, rng)
t_ad = time.perf_counter() - t
print(f"adapter training, 400 steps, batch 24: {t_ad:.2f} s")
t = time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()):
    e1.main()
t_e1 = time.perf_counter() - t
print(f"experiment E1 (all closed-form receivers, 18 points x 4000 draws): {t_e1:.1f} s")
