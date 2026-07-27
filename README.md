# Structured Shared–Private Embedding Multiplexing

Simulation code and data for the manuscript

> K.-H. Lee, H.-H. Choi, and J.-R. Lee, "Structured Shared–Private
> Embedding Multiplexing for Semantic Multiple Access in Dynamic Mobile
> Networks," submitted to *IEEE Transactions on Mobile Computing*, 2026.

Builds on the published shared-embedding multiple-access framework
(Lee, Choi, Lee, *IEEE JSAC*, vol. 44, 2026, doi 10.1109/JSAC.2025.3643816).

## Layout

- `code/semantic_mac.py` — core library (content model, matched-filter
  front end, SR/SC/LMMSE/DR receivers, spectral structure recovery,
  mobility model, affinity tracker)
- `code/exp1_theory.py` — E1: receiver theory validation (affinity and
  SNR sweeps, closed-form overlays) → `data/e1_*.csv`
- `code/exp2_structure.py` — E2: embedding-structure optimization on
  real BERT embeddings (spectral recovery, learned adapter, held-out
  evaluation ladder, paired spec-vs-adapter confidence intervals) →
  `data/e2_*.csv`, `data/e2_exponents.txt`
- `code/exp3_mobility.py` — E3: time-varying affinity tracking under
  mobility (scene traversal and speed sweep) → `data/e3_*.csv`
- `code/exp4_mismatch.py` — E4: robustness to affinity estimation
  error → `data/e4_mismatch.csv`
- `code/exp5_learned.py` — E5: comparison with a trained user-wise
  attention receiver → `data/e5_learned.csv`, `data/e5_train_log.csv`,
  `data/e5_valcurve.csv` (fixed-affinity validation curve)
- `code/check_mask_realization.py` — realized Haar-mask front end vs
  the expected cross-Gram model → `data/e_mask_check.csv`
- `code/replot_all.py` — the single canonical figure generator; reads
  only `data/*.csv` and writes every paper figure with a uniform
  canvas geometry
- `code/lmmse_verify.py` — early derivation-check prototype for the
  affinity-aware LMMSE proposition (predecessor of E1)
- `data/bert_agnews_8000.pt` — frozen `bert-base-uncased` mean-pooled
  embeddings of 8,000 AG News sentences (768-dim), the real-content
  pool used by E2

## Reproduction

Requirements: Python 3.10+, `numpy`, `torch` (CPU is sufficient),
`matplotlib`.

```bash
cd code
python exp1_theory.py      # E1  (minutes)
python exp2_structure.py   # E2  (about an hour on CPU)
python exp3_mobility.py    # E3  (about an hour on CPU)
python exp4_mismatch.py    # E4  (minutes)
python exp5_learned.py     # E5  (minutes)
python replot_all.py       # regenerate every figure from data/*.csv
```

Every experiment fixes its random seeds, experiment scripts write CSVs
only, and `replot_all.py` is the only script that produces figures, so
each figure in the paper is regenerable from the shipped CSVs without
rerunning the experiments.

## License

MIT — see `LICENSE`.
