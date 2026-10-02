"""Complete overhead accounting (Reviewers 1.2 and 4.4).

Reviewer 1 observed that the reported 4 percent counts affinity pilots
only, and asked whether the full-dimensional pilot embeddings, the
coefficient feedback, and the mask updates are included. Reviewer 4 asked
for the same accounting. This script computes every contribution from the
configuration of Table V so that the figure quoted in the manuscript can
be reproduced rather than asserted.

Contributions
  pilots      each user sends n_p clean embeddings on orthogonal resources
              every K-th slot, within a coherence block of B_d embedding
              slots. This is the dominant term and the one the original
              manuscript reported.
  feedback    the receiver returns U share coefficients per update. The
              accuracy needed is set by Remark 1, which shows the receiver
              is flat in the share coefficient over an interval of +-0.2,
              so a coarse uniform quantizer suffices.
  masks       zero over the air. The pseudorandom seed is fixed and every
              node regenerates its mask locally from the fed-back
              coefficients, so only the coefficients travel.
  calibration one-time rather than recurring, amortized over the interval
              between encoder changes.

Output: data/overhead_accounting.txt
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)

# configuration of Table V
from semantic_mac import U_MAIN as U, D_REAL, D_SYN, N_PILOT as N_P, K_PILOT as K, L_D as B_D, N_PAIRS as N_CAL
D = D_REAL              # embedding dimension; the d=64 case is printed as well
BITS_PER_COEF = 4       # 16 levels over [0,1], step 0.0625, well inside
                        # the +-0.2 flat region established by Remark 1
BITS_PER_DIM = 2        # control-channel efficiency assumed for feedback
RECAL_BLOCKS = 10 ** 4  # coherence blocks between encoder recalibrations


def main():
    lines = []
    # --- recurring uplink cost, in embedding slots per K blocks ----------
    slots_per_window = K * B_D
    # every user sends its n_p pilots on its own interference-free
    # resource, so one pilot slot costs U * n_p embedding slots
    pilot_slots = U * N_P
    pilot_frac = pilot_slots / slots_per_window

    # --- recurring feedback, converted to equivalent embedding slots -----
    fb_bits = U * BITS_PER_COEF
    fb_dims = fb_bits / BITS_PER_DIM
    fb_slots = fb_dims / D
    fb_frac = fb_slots / slots_per_window
    fb_frac_syn = (fb_dims / D_SYN) / slots_per_window   # same term at d = 64

    mask_frac = 0.0

    # --- one-time calibration, amortized --------------------------------
    cal_slots = N_CAL * U
    cal_frac = cal_slots / (RECAL_BLOCKS * B_D)

    total_recurring = pilot_frac + fb_frac + mask_frac

    lines.append("Configuration: "
                 f"U={U}, d={D}, n_p={N_P}, K={K}, B_d={B_D}, "
                 f"N_cal={N_CAL}, {BITS_PER_COEF} bits per coefficient")
    lines.append("")
    lines.append(f"{'contribution':<34}{'cost':>22}{'fraction':>14}")
    lines.append("-" * 70)
    lines.append(f"{'affinity pilots':<34}"
                 f"{f'{pilot_slots} of {slots_per_window} slots':>22}"
                 f"{pilot_frac:>13.4%}")
    lines.append(f"{'share-coefficient feedback':<34}"
                 f"{f'{fb_bits} bits per {K} slots':>22}"
                 f"{fb_frac:>13.6%}")
    lines.append(f"{'mask refresh':<34}"
                 f"{'regenerated locally':>22}"
                 f"{mask_frac:>13.4%}")
    lines.append("-" * 70)
    lines.append(f"{'total recurring':<34}{'':>22}{total_recurring:>13.4%}")
    lines.append("")
    lines.append(f"{'calibration (one time)':<34}"
                 f"{f'{cal_slots} slots':>22}"
                 f"{cal_frac:>13.6%}"
                 f"   amortized over {RECAL_BLOCKS} coherence blocks")
    lines.append("")
    lines.append(f"At d={D_SYN} the feedback term is {fb_frac_syn:.4%} of the slot budget.")
    lines.append("The feedback term is smaller than the pilot term by a "
                 f"factor of {pilot_frac / fb_frac:.0f}, so the recurring "
                 "total rounds to the same")
    lines.append("figure as the pilot term alone at the reported precision. "
                 "The mask refresh is")
    lines.append("free over the air because the seed never changes and only "
                 "the coefficients travel.")

    txt = "\n".join(lines)
    with open(os.path.join(DATA, "overhead_accounting.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
