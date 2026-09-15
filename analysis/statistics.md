# Multi-seed statistics (mean with 95% confidence interval)

All intervals are Student-t intervals over independent seed draws. Within a trial every method sees the same seed set.

### ETH same-window, label=Tornado (mixer) (n=10 trials, seed base 700)


**Seed fraction 0.1**

| Method | AUC-ROC (95% CI) | AUC-PR (95% CI) | Recall@FPR=1% (95% CI) |
|---|---|---|---|
| lp | 0.903 ± 0.002 | 0.879 ± 0.010 | 0.607 ± 0.042 |
| ppr | 0.875 ± 0.008 | 0.817 ± 0.022 | 0.346 ± 0.080 |
| xgb | 0.873 ± 0.004 | 0.757 ± 0.006 | 0.266 ± 0.014 |
| mlp | 0.808 ± 0.006 | 0.599 ± 0.008 | 0.086 ± 0.003 |
| appnp | 0.790 ± 0.002 | 0.586 ± 0.022 | 0.042 ± 0.019 |
| gate_fusion | 0.909 ± 0.002 | 0.891 ± 0.007 | 0.680 ± 0.034 |

Learned gate weight w: by validation AUC 0.94 ± 0.02; by validation Recall@FPR=1% 0.98 ± 0.03; the two criteria select the same w in 3/10 trials.

**Seed fraction 0.5**

| Method | AUC-ROC (95% CI) | AUC-PR (95% CI) | Recall@FPR=1% (95% CI) |
|---|---|---|---|
| lp | 0.916 ± 0.003 | 0.907 ± 0.003 | 0.704 ± 0.010 |
| ppr | 0.905 ± 0.003 | 0.894 ± 0.004 | 0.657 ± 0.023 |
| xgb | 0.905 ± 0.003 | 0.812 ± 0.005 | 0.325 ± 0.018 |
| mlp | 0.810 ± 0.005 | 0.600 ± 0.006 | 0.085 ± 0.007 |
| appnp | 0.786 ± 0.006 | 0.584 ± 0.031 | 0.045 ± 0.020 |
| gate_fusion | 0.920 ± 0.003 | 0.916 ± 0.003 | 0.770 ± 0.010 |

Learned gate weight w: by validation AUC 0.93 ± 0.03; by validation Recall@FPR=1% 0.95 ± 0.04; the two criteria select the same w in 4/10 trials.

### ETH same-window, label=Phishing (n=10 trials, seed base 700)


**Seed fraction 0.1**

| Method | AUC-ROC (95% CI) | AUC-PR (95% CI) | Recall@FPR=1% (95% CI) |
|---|---|---|---|
| lp | 0.338 ± 0.005 | 0.492 ± 0.007 | 0.085 ± 0.008 |
| ppr | 0.316 ± 0.004 | 0.452 ± 0.007 | 0.053 ± 0.007 |
| xgb | 0.772 ± 0.006 | 0.674 ± 0.007 | 0.070 ± 0.009 |
| mlp | 0.727 ± 0.009 | 0.635 ± 0.008 | 0.039 ± 0.005 |
| appnp | 0.718 ± 0.007 | 0.623 ± 0.007 | 0.021 ± 0.005 |
| gate_fusion | 0.771 ± 0.009 | 0.685 ± 0.006 | 0.091 ± 0.012 |

Learned gate weight w: by validation AUC 0.45 ± 0.24; by validation Recall@FPR=1% 0.76 ± 0.10; the two criteria select the same w in 1/10 trials.

**Seed fraction 0.5**

| Method | AUC-ROC (95% CI) | AUC-PR (95% CI) | Recall@FPR=1% (95% CI) |
|---|---|---|---|
| lp | 0.370 ± 0.006 | 0.544 ± 0.003 | 0.143 ± 0.006 |
| ppr | 0.349 ± 0.005 | 0.512 ± 0.003 | 0.112 ± 0.008 |
| xgb | 0.798 ± 0.005 | 0.705 ± 0.005 | 0.111 ± 0.005 |
| mlp | 0.739 ± 0.006 | 0.642 ± 0.006 | 0.044 ± 0.004 |
| appnp | 0.729 ± 0.006 | 0.629 ± 0.006 | 0.025 ± 0.005 |
| gate_fusion | 0.805 ± 0.006 | 0.725 ± 0.005 | 0.154 ± 0.007 |

Learned gate weight w: by validation AUC 0.52 ± 0.07; by validation Recall@FPR=1% 0.77 ± 0.10; the two criteria select the same w in 0/10 trials.

### Elliptic t=29 snapshot (n=10 trials, seed base 700)


**Seed fraction 0.1**

| Method | AUC-ROC (95% CI) | Recall@FPR=1% (95% CI) |
|---|---|---|
| lp | 0.895 ± 0.019 | 0.470 ± 0.023 |
| xgb | 0.916 ± 0.009 | 0.168 ± 0.065 |
| gate_fusion | 0.910 ± 0.019 | 0.243 ± 0.097 |

Gate weight selected on the seed-internal split: by AUC 0.00 ± 0.00; by AP 0.20 ± 0.40.

**Seed fraction 0.5**

| Method | AUC-ROC (95% CI) | Recall@FPR=1% (95% CI) |
|---|---|---|
| lp | 0.958 ± 0.005 | 0.664 ± 0.032 |
| xgb | 0.952 ± 0.007 | 0.185 ± 0.090 |
| gate_fusion | 0.955 ± 0.007 | 0.342 ± 0.197 |

Gate weight selected on the seed-internal split: by AUC 0.00 ± 0.00; by AP 0.30 ± 0.46.
