"""E17: pilot budget that the tracker actually needs, counted per user.

Each user sends its pilots on its own interference-free resource, so the
pilot cost of one pilot slot is U * n_p embedding slots, not n_p. With the
E3 setting (n_p = 64 per user, K = 5, B_d = 320, U = 4) that is
256 / 1600 = 16 percent. This experiment reruns the E3 scene traversal and
speed sweep with the same seeds at several per-user pilot counts, so the
overhead can be set from measured tracking accuracy rather than assumed.

Outputs: data/e17_pilot_budget.csv, data/e17_notes.txt
"""
import os
import numpy as np

import exp3_mobility as e3

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

NP_GRID = [8, 16, 32, 64]
SPEEDS = [0.5, 1.0, 2.0, 4.0, 8.0]
REPS_TS, REPS_SP = 5, 8
B_D = e3.BATCH


def overhead(n_p):
    return e3.U * n_p / (e3.K_PILOT * B_D)


def main():
    rows, notes = [], []
    a_ts = e3.drive_through_profile(e3.T)
    for n_p in NP_GRID:
        e3.N_PILOT = n_p
        ts = [e3.run_trace(a_ts, np.random.default_rng(3 + r)) for r in range(REPS_TS)]
        ts_tr = float(np.mean([x["DR-tracked"][1] for x in ts]))
        ts_ge = float(np.mean([x["DR-genie"][1] for x in ts]))
        ts_st = float(np.mean([x["DR-static"][1] for x in ts]))
        gaps = []
        for si, sp in enumerate(SPEEDS):
            acc = []
            for rep in range(REPS_SP):
                rng = np.random.default_rng(1000 + 100 * si + rep)
                a_tr = e3.mobility_trajectories(e3.U, e3.T, sp, rng, **e3.MOB)
                r = e3.run_trace(a_tr, rng)
                acc.append(r["DR-tracked"][1] - r["DR-genie"][1])
            gaps.append(float(np.mean(acc)))
        rows.append((n_p, overhead(n_p), ts_tr, ts_ge, ts_st,
                     (ts_tr - ts_ge) / ts_ge, *gaps))
        notes.append(f"n_p={n_p:3d} per user  overhead={overhead(n_p):6.2%}  "
                     f"traversal SER tracked={ts_tr:.4f} genie={ts_ge:.4f} "
                     f"static={ts_st:.4f} (tracked {100*(ts_tr-ts_ge)/ts_ge:+.1f}% vs genie)  "
                     f"speed gaps " + " ".join(f"{g:.3f}" for g in gaps))
        print(notes[-1])
    with open(os.path.join(DATA, "e17_pilot_budget.csv"), "w") as f:
        f.write("n_p_per_user,overhead,ts_tracked,ts_genie,ts_static,ts_rel_gap,"
                + ",".join(f"gap_speed_{s}" for s in SPEEDS) + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    with open(os.path.join(DATA, "e17_notes.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")
    print("E17 done.")


if __name__ == "__main__":
    main()
