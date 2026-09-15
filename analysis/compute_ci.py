# -*- coding: utf-8 -*-
"""Summarise multi-seed runs: mean, SD and 95% confidence interval.

Reads the raw per-trial JSON files produced by
`exp/multiseed_eval.py` (ETH regimes) and `exp/elliptic_t29_multiseed.py`
(Elliptic t=29 snapshot), and writes `analysis/statistics.md`.

The 95% CI is the Student-t interval, mean +/- t(0.975, n-1) * SD / sqrt(n).
"""
import io
import json
import os

import numpy as np
from scipy import stats as st

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")
OUT = os.path.join(BASE, "analysis", "statistics.md")

METRICS = ("auc_roc", "auc_pr", "recall_at_fpr1")
METHODS = ("lp", "ppr", "xgb", "mlp", "appnp", "gate_fusion")


def ci95(values):
    v = np.asarray(values, dtype=float)
    n = len(v)
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    half = float(st.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
    return mean, sd, half, n


def fmt(values):
    mean, sd, half, n = ci95(values)
    return f"{mean:.3f} ± {half:.3f}", f"{sd:.3f}", n


def eth_block(path, label, lines):
    d = json.load(io.open(path, encoding="utf-8"))
    meta = d["meta"]
    lines.append(f"\n### ETH same-window, label={label} "
                 f"(n={meta['trials']} trials, seed base {meta['seed_base']})\n")
    for frac, res in d["results"].items():
        lines.append(f"\n**Seed fraction {frac}**\n")
        lines.append("| Method | AUC-ROC (95% CI) | AUC-PR (95% CI) | "
                     "Recall@FPR=1% (95% CI) |")
        lines.append("|---|---|---|---|")
        for m in METHODS:
            if m not in res:
                continue
            cells = []
            for metric in METRICS:
                txt, _, _ = fmt([r[metric] for r in res[m]])
                cells.append(txt)
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        if "gate_w_auc" in res:
            wa, sa, _, na = ci95(res["gate_w_auc"])
            wr, sr, _, nr = ci95(res["gate_w_recall"])
            agree = sum(1 for a, b in zip(res["gate_w_auc"], res["gate_w_recall"])
                        if abs(a - b) < 1e-9)
            lines.append(f"\nLearned gate weight w: by validation AUC "
                         f"{wa:.2f} ± {sa:.2f}; by validation Recall@FPR=1% "
                         f"{wr:.2f} ± {sr:.2f}; the two criteria select the same "
                         f"w in {agree}/{na} trials.")


def elliptic_block(path, lines):
    d = json.load(io.open(path, encoding="utf-8"))
    meta = d["meta"]
    lines.append(f"\n### Elliptic t={meta['time_step']} snapshot "
                 f"(n={meta['trials']} trials, seed base {meta['seed_base']})\n")
    for frac, res in d["results"].items():
        lines.append(f"\n**Seed fraction {frac}**\n")
        lines.append("| Method | AUC-ROC (95% CI) | Recall@FPR=1% (95% CI) |")
        lines.append("|---|---|---|")
        for m in ("lp", "xgb", "gate_fusion"):
            if m not in res:
                continue
            a, _, _ = fmt([r["auc_roc"] for r in res[m]])
            r_, _, _ = fmt([r["recall_at_fpr1"] for r in res[m]])
            lines.append(f"| {m} | {a} | {r_} |")
        if "learned_w" in res:
            wa = [w[0] for w in res["learned_w"]]
            wn = [w[1] for w in res["learned_w"]]
            lines.append(f"\nGate weight selected on the seed-internal split: "
                         f"by AUC {np.mean(wa):.2f} ± {np.std(wa):.2f}; "
                         f"by AP {np.mean(wn):.2f} ± {np.std(wn):.2f}.")


def main():
    lines = ["# Multi-seed statistics (mean with 95% confidence interval)",
             "",
             "All intervals are Student-t intervals over independent seed draws. "
             "Within a trial every method sees the same seed set."]
    for name, label in (("multiseed_label2.json", 2), ("multiseed_label1.json", 1)):
        p = os.path.join(RES, name)
        if os.path.exists(p):
            eth_block(p, "Tornado (mixer)" if label == 2 else "Phishing", lines)
    p = os.path.join(RES, "elliptic_t29_multiseed.json")
    if os.path.exists(p):
        elliptic_block(p, lines)
    text = "\n".join(lines) + "\n"
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
