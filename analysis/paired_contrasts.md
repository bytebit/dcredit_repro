# Paired gate contrasts

Paired within-trial differences (gate minus baseline) over 10 independent seed draws. Intervals are Student-t 95% CIs of the paired differences; p values come from two-sided paired t-tests, Holm-adjusted within each metric family.

Primary family: Recall@FPR=1%. AUC-ROC and AUC-PR are secondary.

## Primary: Recall@FPR=1%

| Comparison | Metric | Mean paired diff | 95% CI | Holm p | n |
|---|---|---|---|---|---|
| gate - LP, Tornado, 10 seeds | Recall@FPR=1% | +0.099 | [+0.068, +0.130] | 0.0002 | 10 |
| gate - XGBoost, Tornado, 10 seeds | Recall@FPR=1% | +0.355 | [+0.330, +0.381] | 0.0000 | 10 |
| gate - LP, Tornado, 50 seeds | Recall@FPR=1% | +0.083 | [+0.069, +0.097] | 0.0000 | 10 |
| gate - XGBoost, Tornado, 50 seeds | Recall@FPR=1% | +0.360 | [+0.339, +0.380] | 0.0000 | 10 |
| gate - LP, Phishing, 10 seeds | Recall@FPR=1% | +0.006 | [-0.005, +0.017] | 0.2714 | 10 |
| gate - XGBoost, Phishing, 10 seeds | Recall@FPR=1% | +0.021 | [+0.012, +0.031] | 0.0021 | 10 |
| gate - LP, Phishing, 50 seeds | Recall@FPR=1% | +0.011 | [+0.005, +0.016] | 0.0036 | 10 |
| gate - XGBoost, Phishing, 50 seeds | Recall@FPR=1% | +0.042 | [+0.035, +0.050] | 0.0000 | 10 |

## Secondary: AUC metrics

| Comparison | Metric | Mean paired diff | 95% CI | Holm p | n |
|---|---|---|---|---|---|
| gate - LP, Tornado, 10 seeds | AUC-ROC | +0.007 | [+0.006, +0.008] | 0.0000 | 10 |
| gate - XGBoost, Tornado, 10 seeds | AUC-ROC | +0.005 | [+0.002, +0.009] | 0.0283 | 10 |
| gate - LP, Tornado, 10 seeds | AUC-PR | +0.019 | [+0.015, +0.024] | 0.0000 | 10 |
| gate - XGBoost, Tornado, 10 seeds | AUC-PR | +0.115 | [+0.106, +0.123] | 0.0000 | 10 |
| gate - LP, Tornado, 50 seeds | AUC-ROC | +0.005 | [+0.005, +0.006] | 0.0000 | 10 |
| gate - XGBoost, Tornado, 50 seeds | AUC-ROC | -0.015 | [-0.020, -0.010] | 0.0005 | 10 |
| gate - LP, Tornado, 50 seeds | AUC-PR | +0.014 | [+0.012, +0.017] | 0.0000 | 10 |
| gate - XGBoost, Tornado, 50 seeds | AUC-PR | +0.086 | [+0.081, +0.091] | 0.0000 | 10 |
| gate - LP, Phishing, 10 seeds | AUC-ROC | +0.433 | [+0.423, +0.443] | 0.0000 | 10 |
| gate - XGBoost, Phishing, 10 seeds | AUC-ROC | -0.001 | [-0.006, +0.005] | 0.7752 | 10 |
| gate - LP, Phishing, 10 seeds | AUC-PR | +0.193 | [+0.184, +0.202] | 0.0000 | 10 |
| gate - XGBoost, Phishing, 10 seeds | AUC-PR | +0.010 | [+0.006, +0.014] | 0.0010 | 10 |
| gate - LP, Phishing, 50 seeds | AUC-ROC | +0.434 | [+0.425, +0.444] | 0.0000 | 10 |
| gate - XGBoost, Phishing, 50 seeds | AUC-ROC | +0.007 | [+0.001, +0.013] | 0.0572 | 10 |
| gate - LP, Phishing, 50 seeds | AUC-PR | +0.182 | [+0.177, +0.187] | 0.0000 | 10 |
| gate - XGBoost, Phishing, 50 seeds | AUC-PR | +0.021 | [+0.018, +0.023] | 0.0000 | 10 |

## Ten paired differences

- **gate - LP, Tornado, 10 seeds** (AUC-ROC): +0.005, +0.006, +0.006, +0.006, +0.006, +0.007, +0.009, +0.006, +0.009, +0.006
- **gate - XGBoost, Tornado, 10 seeds** (AUC-ROC): +0.006, +0.012, +0.003, +0.013, +0.007, +0.002, +0.002, +0.004, +0.006, -0.004
- **gate - LP, Tornado, 10 seeds** (AUC-PR): +0.014, +0.015, +0.019, +0.017, +0.018, +0.018, +0.032, +0.015, +0.029, +0.017
- **gate - XGBoost, Tornado, 10 seeds** (AUC-PR): +0.123, +0.129, +0.096, +0.128, +0.129, +0.109, +0.103, +0.117, +0.107, +0.108
- **gate - LP, Tornado, 10 seeds** (Recall@FPR=1%): +0.100, +0.075, +0.131, +0.127, +0.056, +0.064, +0.187, +0.039, +0.107, +0.098
- **gate - XGBoost, Tornado, 10 seeds** (Recall@FPR=1%): +0.415, +0.353, +0.302, +0.387, +0.347, +0.379, +0.299, +0.374, +0.337, +0.361
- **gate - LP, Tornado, 50 seeds** (AUC-ROC): +0.004, +0.005, +0.004, +0.006, +0.005, +0.006, +0.005, +0.004, +0.005, +0.005
- **gate - XGBoost, Tornado, 50 seeds** (AUC-ROC): -0.010, -0.015, -0.002, -0.015, -0.025, -0.028, -0.013, -0.017, -0.016, -0.011
- **gate - LP, Tornado, 50 seeds** (AUC-PR): +0.012, +0.012, +0.011, +0.023, +0.014, +0.019, +0.013, +0.015, +0.013, +0.011
- **gate - XGBoost, Tornado, 50 seeds** (AUC-PR): +0.085, +0.090, +0.087, +0.100, +0.081, +0.074, +0.090, +0.085, +0.085, +0.084
- **gate - LP, Tornado, 50 seeds** (Recall@FPR=1%): +0.101, +0.076, +0.082, +0.111, +0.060, +0.106, +0.075, +0.076, +0.049, +0.093
- **gate - XGBoost, Tornado, 50 seeds** (Recall@FPR=1%): +0.374, +0.421, +0.358, +0.390, +0.357, +0.325, +0.353, +0.331, +0.339, +0.349
- **gate - LP, Phishing, 10 seeds** (AUC-ROC): +0.428, +0.444, +0.429, +0.411, +0.426, +0.454, +0.422, +0.423, +0.438, +0.452
- **gate - XGBoost, Phishing, 10 seeds** (AUC-ROC): +0.004, +0.005, -0.015, -0.001, -0.004, +0.003, -0.012, +0.004, +0.002, +0.007
- **gate - LP, Phishing, 10 seeds** (AUC-PR): +0.190, +0.201, +0.205, +0.172, +0.193, +0.200, +0.189, +0.174, +0.196, +0.209
- **gate - XGBoost, Phishing, 10 seeds** (AUC-PR): +0.020, +0.008, +0.004, +0.017, +0.016, +0.005, +0.007, +0.011, +0.004, +0.011
- **gate - LP, Phishing, 10 seeds** (Recall@FPR=1%): +0.013, +0.008, +0.022, -0.004, +0.010, -0.007, +0.032, -0.024, -0.002, +0.011
- **gate - XGBoost, Phishing, 10 seeds** (Recall@FPR=1%): +0.042, +0.013, +0.026, +0.024, +0.030, +0.003, +0.034, +0.026, -0.000, +0.016
- **gate - LP, Phishing, 50 seeds** (AUC-ROC): +0.420, +0.432, +0.419, +0.443, +0.440, +0.429, +0.456, +0.428, +0.427, +0.451
- **gate - XGBoost, Phishing, 50 seeds** (AUC-ROC): +0.017, +0.015, -0.006, +0.010, +0.011, +0.004, +0.017, -0.007, +0.006, +0.002
- **gate - LP, Phishing, 50 seeds** (AUC-PR): +0.174, +0.172, +0.178, +0.186, +0.183, +0.180, +0.192, +0.180, +0.180, +0.192
- **gate - XGBoost, Phishing, 50 seeds** (AUC-PR): +0.025, +0.021, +0.015, +0.025, +0.022, +0.019, +0.024, +0.015, +0.023, +0.017
- **gate - LP, Phishing, 50 seeds** (Recall@FPR=1%): +0.000, -0.002, +0.020, +0.016, +0.004, +0.009, +0.018, +0.016, +0.013, +0.012
- **gate - XGBoost, Phishing, 50 seeds** (Recall@FPR=1%): +0.029, +0.025, +0.050, +0.047, +0.049, +0.053, +0.034, +0.052, +0.049, +0.037
