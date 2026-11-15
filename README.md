# Reproducible code for "When Credit Propagation Helps in On-Chain Fraud Detection: A Regime Analysis and Low-False-Positive Decision Rule"

**Archived release:** v1.1.0 — https://doi.org/10.5281/zenodo.22874572
(concept DOI covering all versions). Cite v1.1.0 for the corrected evaluation.

This repository reproduces every table and figure of the manuscript:

- the four-regime structure analysis (laundering clusters, mixer, dispersed
  phishing, cross-time extrapolation);
- the D-Credit gated fusion results and the learned gate weights;
- the three ablations (supervision gap, operator choice, gate-selection metric);
- the interpretability case study and the runtime figures.

All experiments run on a single NVIDIA RTX 2060 (12 GB). The multi-seed runs use
**10 independent seed draws** and report **95% confidence intervals**.

## Repository layout

```
repro/
  exp/        experiment scripts (data preparation, baselines, D-Credit, multi-seed)
  figures/    scripts that regenerate Fig. 1-3
  analysis/   compute_ci.py -> statistics.md (mean, SD, 95% CI)
  results/    reference outputs (JSON) produced by the scripts
  data/       place the datasets here (see data/README.md; not distributed)
```

## Environment

```bash
pip install -r requirements.txt
```

Verified with Python 3.12, torch 2.5.1+cu121, torch-geometric 2.8.0.post1,
xgboost 3.4.1, scikit-learn 1.8.0, numpy 1.26.4, scipy 1.17.1, pandas 3.0.3,
matplotlib 3.10.9. CPU-only runs work but are slower; pass `--device cpu` to the
PyTorch scripts.

## Data

The datasets are public and are **not** redistributed here. See `data/README.md`
for the exact sources and the directory layout the scripts expect:

```
data/elliptic/processed/elliptic.pt          # Elliptic (Bitcoin), from Kaggle
data/bert4eth/raw/*.csv                      # BERT4ETH (Ethereum), from GitHub
data/bert4eth/processed/nodes.npz, edges.npz # built by preprocess_bert4eth.py
results/eth_protocol.npz                     # built by prep_protocol.py
```

## Reproduction commands

Data preparation (once):

```bash
python exp/preprocess_bert4eth.py     # CSV -> nodes.npz / edges.npz
python exp/prep_protocol.py           # temporal split -> results/eth_protocol.npz
```

Multi-seed evaluation of the ETH **transductive all-time** regimes (mixer Table 3,
phishing Table 4, gated rule Table 6 of the manuscript; one seed draw shared by
every method inside a trial; final metrics exclude both the training seeds and the
validation nodes used to select the gate):

```bash
python exp/multiseed_eval.py --label 2 --fractions 0.1,0.5 --trials 10   # mixer
python exp/multiseed_eval.py --label 1 --fractions 0.1,0.5 --trials 10   # phishing
python exp/sage_same_seed.py --label 2 --fractions 0.1,0.5 --trials 10 --epochs 200
python exp/sage_same_seed.py --label 1 --fractions 0.1,0.5 --trials 10 --epochs 200
```

Elliptic t=29 snapshot (Table 2, laundering clusters):

```bash
python exp/elliptic_t29_multiseed.py --trials 10 --fractions 0.1,0.5
```

Cross-time regime and full-label baselines (Table 5 and §4.4):

```bash
python exp/baselines_propagation.py        # LP / PPR / APPNP under the official split
python exp/baselines_xgb_official.py       # XGBoost, full and local feature sets
python exp/train_baselines_gnn.py          # GCN / GraphSAGE / GAT / GIN / GCNII
```

Operator and gate ablations, interpretability case study (Sections 6.4-6.6):

```bash
python exp/draft_support.py                # direction ablation + t29 fusion + case study
python exp/d_credit_v01.py --label 2       # supervision-gap variant (CE + consistency)
```

Statistics (mean, SD, 95% CI over the 10 trials):

```bash
python analysis/compute_ci.py              # -> analysis/statistics.md
python analysis/paired_contrasts.py        # -> analysis/paired_contrasts.{md,json}
python analysis/interpretability_audit.py  # -> analysis/interpretability_audit.{md,json}
```

Figures:

```bash
python figures/fig1_regime_comparison.py
python figures/fig2_dcredit_architecture.py
python figures/fig3_gate_weight.py
```

## Protocol notes

- **Same seeds for every method.** Inside a trial, all methods receive one seed
  draw (the same positives and negatives), so rows of a table are directly
  comparable; the reported intervals are Student-t intervals over 10 draws.
- **Gate learned on validation data only.** The fusion weight `w` is selected on
  the official Ethereum validation period, which is disjoint from the evaluation
  nodes. `multiseed_eval.py` records both the AUC-selected and the
  Recall@FPR=1%-selected `w`, so the gate-selection analysis (Section 5.3) can be
  checked directly.
- **Strict evaluation isolation.** For every trial the evaluation mask excludes
  both the training seeds and the validation nodes. Two assertions abort a run on
  any overlap, and each output JSON stores an `evaluation_protocol` block with
  `seed_nodes_used_for_scoring: 0` and `validation_nodes_used_for_scoring: 0`.
- **Transductive Ethereum protocol.** Features and graph edges are built from the
  complete 2017–2022 snapshot (blocks up to 14,297,730). The first-appearance split
  only determines which labeled nodes are scored; it does not make the features
  future-blind. The ETH results are therefore retrospective and transductive.
- **Same-seed graph baseline.** `exp/sage_same_seed.py` trains GraphSAGE with the
  identical seed draws, graph and validation-only early stopping, and the same
  evaluation masks.
- **Metrics.** AUC-ROC, average precision with the fraud class as positive, and
  Recall@FPR=1% (threshold chosen for a 1% false-positive rate).

## License and contact

Code is released under the MIT License (see `LICENSE`). Elliptic and BERT4ETH
data remain under the terms of their original providers.

Junfei Huang, School of Computer Science, Beijing University of Posts and
Telecommunications — huangjunfei@bupt.edu.cn
