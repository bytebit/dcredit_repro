# -*- coding: utf-8 -*-
"""Week1 - XGBoost 基线：官方时间划分 + 特征泄漏对照（全 166 vs local 94）

指标口径：illicit=正类（y_metric: illicit=1, licit=0；score=P(illicit)=1-P(licit)）
  - Macro-F1（验证集选阈值）、AUC-ROC、AP（fraud AP）、Recall@FPR=1%（1% licit 被误标时对 illicit 的召回）
"""
import json
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
import xgboost as xgb

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "elliptic", "processed", "elliptic.pt")
PROTO = os.path.join(BASE, "results", "elliptic_protocol.pt")
OUT = os.path.join(BASE, "results")


def recall_at_fpr(y_metric, score, fpr_target=0.01):
    """正类=illicit(1)，负类=licit(0)；把 licit 误报率压到 fpr_target 时对 illicit 的召回。"""
    neg = y_metric == 0
    pos = y_metric == 1
    n_neg = int(neg.sum())
    order = np.argsort(-score, kind="stable")
    hit_neg = 0
    for rank, idx in enumerate(order):
        if y_metric[idx] == 0:
            hit_neg += 1
        if hit_neg / n_neg >= fpr_target:
            k = rank + 1
            return float((y_metric[order[:k]] == 1).sum() / max(pos.sum(), 1)), k
    return float((y_metric[order] == 1).sum() / max(pos.sum(), 1)), len(order)


def best_threshold_f1(y_val, score_val, y_te, score_te):
    cands = np.unique(score_val)
    cands = np.concatenate([[0.0], cands, [1.0]])
    best_t, best_f1 = 0.5, -1.0
    for thr in cands:
        f1 = f1_score(y_val, (score_val >= thr).astype(int), average="macro")
        if f1 > best_f1:
            best_t, best_f1 = thr, f1
    pred = (score_te >= best_t).astype(int)
    return best_t, best_f1, f1_score(y_te, pred, average="macro")


def run_variant(name, cols, X, y, train_mask, val_mask, test_mask):
    Xtr = X[train_mask][:, cols]
    Xva = X[val_mask][:, cols]
    Xte = X[test_mask][:, cols]
    ytr = y[train_mask]
    yva = y[val_mask]
    yte = y[test_mask]

    scale_pos = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=0,
        n_jobs=-1,
    )
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

    # 指标方向：illicit=正类
    pva = 1.0 - model.predict_proba(Xva)[:, 1]
    pte = 1.0 - model.predict_proba(Xte)[:, 1]
    yva_m = (yva == 0).astype(int)
    yte_m = (yte == 0).astype(int)
    thr, val_f1, te_f1 = best_threshold_f1(yva_m, pva, yte_m, pte)
    rec, k = recall_at_fpr(yte_m, pte, 0.01)
    res = {
        "variant": name,
        "n_features": len(cols),
        "thr": float(thr),
        "val_macro_f1": float(val_f1),
        "test_macro_f1": float(te_f1),
        "auc_roc": float(roc_auc_score(yte_m, pte)),
        "auc_pr_illicit": float(average_precision_score(yte_m, pte)),
        "recall_at_fpr1": rec,
        "n_ranked_for_fpr1": k,
    }
    print(f"[{name}] {json.dumps(res, ensure_ascii=False)}")
    return res


def main():
    d = torch.load(DATA, weights_only=False)
    X = d["x"].numpy()
    y = d["y"].numpy()
    proto = torch.load(PROTO, weights_only=False)
    train_mask, val_mask, test_mask = (
        proto["train_mask"].numpy(),
        proto["val_mask"].numpy(),
        proto["test_mask"].numpy(),
    )

    results = [
        run_variant("full_166", list(range(166)), X, y, train_mask, val_mask, test_mask),
        run_variant("local_only_94", list(range(94)), X, y, train_mask, val_mask, test_mask),
    ]
    with open(os.path.join(OUT, "xgb_official_split.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved ->", os.path.join(OUT, "xgb_official_split.json"))


if __name__ == "__main__":
    main()
