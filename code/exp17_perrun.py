"""E17 per-run values: the scene-traversal part of the pilot-budget sweep.

Repeats the traversal part of exp17_pilot_budget.py with the same seeds and
prints the tracked, static and genie SER of every run, so that the
run-by-run comparison quoted in the letter (Comment 1.2) can be reproduced.

Output: stdout, saved as data/e17_perrun.txt."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp3_mobility as e3
NP_GRID = [8, 16, 32, 64]
REPS_TS = 5
a_ts = e3.drive_through_profile(e3.T)
for n_p in NP_GRID:
    e3.N_PILOT = n_p
    t0 = time.time()
    tr, ge, st = [], [], []
    for r in range(REPS_TS):
        x = e3.run_trace(a_ts, np.random.default_rng(3 + r))
        tr.append(x["DR-tracked"][1]); ge.append(x["DR-genie"][1]); st.append(x["DR-static"][1])
    tr, ge, st = map(np.array, (tr, ge, st))
    print(f"n_p={n_p:3d} overhead={e3.U*n_p/(e3.K_PILOT*e3.BATCH):.2%}  ({time.time()-t0:.0f}s)")
    print("   tracked per run:", np.round(tr, 4), "mean", round(tr.mean(), 4), "sd", round(tr.std(ddof=1), 4))
    print("   static  per run:", np.round(st, 4), "mean", round(st.mean(), 4), "sd", round(st.std(ddof=1), 4))
    print("   genie   per run:", np.round(ge, 4), "mean", round(ge.mean(), 4))
    d = st - tr
    print("   paired static-tracked:", np.round(d, 4), "mean", round(d.mean(), 4), "sd", round(d.std(ddof=1), 4),
          "wins(tracked<static)", int((d > 0).sum()), "/", REPS_TS, flush=True)
