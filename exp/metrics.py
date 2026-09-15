# -*- coding: utf-8 -*-
"""共享指标工具：illicit=正类 口径"""

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def recall_at_fpr(y_metric, score, fpr_target=0.01):
    """正类=illicit(1)，负类=licit(0)；licit 误报率 ≤ fpr_target 时对 illicit 的召回。"""
    neg = y_metric == 0
    pos = y_metric == 1
    n_neg = int(neg.sum())
    n_pos = int(pos.sum())
    order = np.argsort(-score, kind="stable")
    hit_neg = 0
    for rank, idx in enumerate(order):
        if y_metric[idx] == 0:
            hit_neg += 1
        if hit_neg / n_neg >= fpr_target:
            k = rank + 1
            return float((y_metric[order[:k]] == 1).sum() / max(n_pos, 1)), k
    return float((y_metric[order] == 1).sum() / max(n_pos, 1)), len(order)


def val_tuned_f1(y_val, score_val, y_te, score_te):
    cands = np.unique(np.concatenate([[0.0], score_val, [1.0]]))
    best_t, best_f1 = 0.5, -1.0
    for thr in cands:
        f1 = f1_score(y_val, (score_val >= thr).astype(int), average="macro")
        if f1 > best_f1:
            best_t, best_f1 = thr, f1
    return best_t, best_f1, f1_score(y_te, (score_te >= best_t).astype(int), average="macro")


def evaluate(y_te, score_te, y_val=None, score_val=None):
    res = {
        "auc_roc": float(roc_auc_score(y_te, score_te)),
        "auc_pr_illicit": float(average_precision_score(y_te, score_te)),
        "recall_at_fpr1": float(recall_at_fpr(y_te, score_te, 0.01)[0]),
    }
    if y_val is not None and score_val is not None:
        thr, val_f1, te_f1 = val_tuned_f1(y_val, score_val, y_te, score_te)
        res["thr"] = float(thr)
        res["val_macro_f1"] = float(val_f1)
        res["test_macro_f1"] = float(te_f1)
    return res
