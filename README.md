# Structured Shared-Private Embedding Multiplexing: Simulation Code and Data

Reproducibility package for "Semantic Multiple Access via Structured Shared-Private
Embeddings in Mobile Networks" (Ki-Ho Lee, Hyun-Ho Choi, Jung-Ryun Lee; under review
at IEEE Transactions on Mobile Computing, major revision 2026). Every figure and every
number in the paper is regenerated from the files here.

## Layout
- `code/semantic_mac.py` — shared library: embedding model, masks, matched-filter front
  end, receivers (SimR, SimC, LMMSE, DR, DR-MMSE), spectral recovery, affinity tracker,
  conventional references (OMA, NOMA-SIC), and the main configuration block
  (`U_MAIN`, `D_SYN/DC_SYN`, `D_REAL/DC_REAL`, SNRs, pilot parameters, `TAU`, `H_MIN`, `MOB`).
- `code/exp*.py` — one script per experiment; each writes its CSV to `data/`.
- `code/replot_all.py` — regenerates every result figure in `fig/` from `data/` only.
- `code/check_*.py`, `code/lmmse_verify.py` — closed-form and optimality checks.
- `code/overhead_accounting.py` — the pilot/feedback/calibration overhead accounting.
- `data/` — raw results (CSV) and notes behind every figure and table.
- `fig/` — figure PDFs as printed.

## Figure to script to data
| Figure | Script | Data |
|---|---|---|
| Fig. 1 | `fig1_system_r1.tex` (TikZ) | — |
| Fig. 2 | `exp1_theory.py` | `e1_beta.csv` (+ `e1_dr_cf.csv` from `check_dr_closedform.py`) |
| Fig. 3 | `exp1_theory.py` | `e1_snr.csv` |
| Figs. 4–6 | `exp2_structure.py` | `e2_spectrum_multi.csv`, `e2_subspace_multi.csv`, `e2_ladder.csv` |
| Figs. 7–8 | `exp3_mobility.py` | `e3_timeseries.csv`, `e3_speed.csv` |
| Fig. 9 | `exp4_mismatch.py`, `exp4_lmmse_ref.py` | `e4_mismatch.csv`, `e4_lmmse_ref.csv` |
| Fig. 10 | `exp7_learned_sideinfo.py` | `e7_learned_sideinfo.csv` |
| Fig. 11 | `exp8_partial_sharing.py` | `e8_partial_sharing.csv` |
| Fig. 12 | `exp12_corpus.py` (DR-MMSE curve: `exp16_scene_gap.py`) | `e12_corpus.csv`, `e16_scene_gap.csv` |
| Fig. 13 | `exp11_impairments.py` | `e11_imbalance.csv` |
| Table V | `exp9_retrieval.py` | `e9_retrieval.csv` |
| Table VI | `exp11_impairments.py` | `e11_phase.csv`, `e11_gain.csv` |
| E10 (text) | `exp10_eval_choices.py` | `e10_threshold_sweep.csv`, `e10_clip_sweep.csv` |
| E17 (text) | `exp17_pilot_budget.py` | `e17_pilot_budget.csv` |

## Requirements and conventions
- Python 3.10, NumPy 2.2, matplotlib 3.9; PyTorch 2.5 (CPU) for E2/E5/E15, PyTorch 2.10+cu128
  on an NVIDIA RTX A4500 for E7. `semantic_mac.DEVICE` selects CUDA when available.
- Seeds are fixed inside every script (`np.random.default_rng(<constant>)`,
  `torch.manual_seed`); E7 trains over seeds 0–4.
- Embeddings are unit norm; the per-user SNR is rho = 1/sigma^2 referenced to a
  unit-norm embedding; block-fading Rayleigh gains with E[h^2] = 1 are clipped at
  h_min = 0.2; the SER threshold is tau = 0.45 on the cosine recovery.
- The BERT embeddings (`data/bert_agnews_8000.pt`, 8000 x 768, bert-base-uncased mean
  pooled, AG-News) are provided for the real-embedding experiments.

Run `python3 code/exp1_theory.py` ... `python3 code/exp17_pilot_budget.py` to regenerate
`data/`, then `python3 code/replot_all.py` to regenerate `fig/`.
