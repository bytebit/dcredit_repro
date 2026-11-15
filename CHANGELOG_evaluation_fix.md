# CHANGELOG — evaluation isolation and protocol fix (2026-11-11)

## Old behaviour

1. **Mixer evaluation reused validation nodes.** `exp/multiseed_eval.py` built the
   evaluation set as "all candidates minus training seeds" for the Tornado regime.
   The validation nodes used to select the gate weight were therefore scored
   again in the reported metrics, contradicting the manuscript statement that the
   gate is never selected on evaluation nodes.
2. **Ethereum protocol wording was inconsistent with the code.** Node features
   (in/out degree and in/out amount) and graph edges were built from all
   transactions in the 2017–2022 snapshot, while the manuscript described the
   ETH experiments as "same window" and as a temporal split.
3. **No same-seed graph baseline.** GNN baselines were only evaluated under the
   full-label Elliptic setting.
4. **No paired uncertainty.** Gate advantages were argued from marginal
   confidence intervals rather than per-trial paired differences.
5. **Interpretability claim rested on three illustrative paths.**

## Corrected behaviour

1. The evaluation mask now excludes **both** training seeds and validation nodes;
   two assertions abort the run on any overlap, and every output JSON records
   `evaluation_protocol` (`seed_nodes_used_for_scoring: 0`,
   `validation_nodes_used_for_scoring: 0`,
   `evaluation_nodes_exclude_validation: true`).
2. **Protocol decision: transductive, retrospective all-time graph (Option A).**
   The first-appearance split determines which labeled nodes are scored; features
   and edges use the complete 2017–2022 snapshot (blocks up to 14,297,730) and may
   include a node's own evaluation period. All ETH wording was rewritten
   accordingly; "same-window" and temporal-holdout language was removed.
3. A **GraphSAGE** baseline was trained under the identical seed draws, graph,
   validation early stopping and evaluation masks (10 trials per setting).
4. **Paired contrasts** (gate − LP, gate − XGBoost) with Student-t CIs and Holm
   correction were added for Recall@FPR=1% (primary family) and the AUC metrics
   (secondary families).
5. The path claim was replaced by a **systematic audit** of the top-100
   highest-credit held-out accounts (retrieval rate, median path length, paths via
   normal/licit nodes); no analyst-utility claim is made.

## Files changed

- `exp/multiseed_eval.py` — strict isolation, assertions, audit metadata,
  `--out-dir`.
- `exp/sage_same_seed.py` — new GraphSAGE baseline.
- `analysis/paired_contrasts.py` — new paired contrast analysis.
- `analysis/interpretability_audit.py` — new systematic path audit.
- `analysis/compute_ci.py` — GraphSAGE rows, transductive labels.
- `figures/fig1_regime_comparison.py` — regenerated from corrected JSON.
- `paper_v2.md` — corrected protocol language, corrected tables, narrowed claims.
- `build_docx_v2.py` — rebuilt manuscript.

## Experiments rerun

- Mixer (label 2) and phishing (label 1): 10 trials × fractions {0.1, 0.5},
  seed base 700, run twice (run 1 and run 2) for the reproduction check.
- GraphSAGE: same protocol, 200 epochs, 10 trials × 2 fractions × 2 regimes,
  run twice.
- Elliptic laundering (t=29) and cross-time results are unchanged and were not
  rerun.

## Old versus new headline metrics (Tornado, AUC / Recall@FPR=1%)

| Method / seeds | Before rework | After rework |
| --- | --- | --- |
| LP, 10% | 0.903 / 0.607 | **0.881 / 0.578** |
| XGBoost, 10% | 0.873 / 0.266 | **0.882 / 0.321** |
| Gated rule, 10% | 0.909 / 0.680 | **0.888 / 0.676** |
| LP, 50% | 0.916 / 0.704 | **0.889 / 0.665** |
| XGBoost, 50% | 0.905 / 0.325 | **0.910 / 0.388** |
| Gated rule, 50% | 0.920 / 0.770 | **0.895 / 0.748** |

New same-seed GraphSAGE: 0.933 / 0.528 (10%) and 0.937 / 0.548 (50%).
Phishing results are numerically unchanged (the evaluation mask was already
restricted to test-period nodes).

## Headline interpretation after correction

- The **low-FPR advantage of propagation and of the gated rule survives**: the
  gate gains +0.099 [+0.068, +0.130] over LP and +0.355 [+0.330, +0.381] over
  XGBoost at 10% seeds, and +0.083 [+0.069, +0.097] / +0.360 [+0.339, +0.380] at
  50% seeds (paired t-tests, Holm-adjusted p ≤ 0.001).
- The previous statement that the gated rule **beats both channels on AUC does
  not survive**: its AUC is 0.888/0.895 versus 0.882/0.910 for XGBoost, and a
  same-seed GraphSAGE baseline reaches 0.933/0.937.
- Path backtracking is **systematically retrievable** (100/100 top-credit accounts
  in the audited trial, median 1 hop, 3% of paths via normal/licit nodes), but
  human utility is not established.

## Results that could not be reproduced

None: the second independent run reproduced every metric; see
`verification_report.md`.
