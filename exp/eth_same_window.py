# -*- coding: utf-8 -*-
"""ETH 同窗口检测（D-Credit 的正确评测场景）

设定：测试期账户内，随机抽 frac 的 phishing（+normal）作种子，
评测能否找到同期其余 phishing（held-out phish vs normal）。
模型：LP / PPR（图结构）/ XGBoost（同种子特征基线）

--label 2：正类改为 tornado（混币器）账户，候选集为全图 tornado vs normal
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb

from metrics import evaluate, recall_at_fpr, val_tuned_f1

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "bert4eth", "processed")
PROTO = os.path.join(BASE, "results", "eth_protocol.npz")
OUT = os.path.join(BASE, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", default="0.1,0.2,0.5")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--label", type=int, default=1, help="正类标签：1=phishing, 2=tornado")
    args = ap.parse_args()

    nd = np.load(os.path.join(PROC, "nodes.npz"), allow_pickle=True)
    ed = np.load(os.path.join(PROC, "edges.npz"))
    proto = np.load(PROTO)
    labels = nd["label"].astype(int)
    feat = nd["feat"].astype(np.float32)
    edges = ed["edges"]
    test_mask = proto["test_mask"]
    n = len(labels)
    y = np.full(n, -1, dtype=int)
    y[labels == 1] = 1
    y[labels == 0] = 0
    if args.label == 1:
        cand_mask = test_mask & (y != -1)
        pos_candidates = np.where((labels == args.label) & cand_mask)[0]
        y_pos = y
    else:
        cand_mask = (labels == args.label) | (labels == 0)
        pos_candidates = np.where((labels == args.label) & cand_mask)[0]
        y_pos = (labels == args.label).astype(int)
    neg_candidates = np.where((labels == 0) & cand_mask)[0]

    src = edges[:, 0].astype(int)
    dst = edges[:, 1].astype(int)
    src_t = torch.from_numpy(src).long()
    dst_t = torch.from_numpy(dst).long()
    n_e = len(src)
    deg_u = np.bincount(np.concatenate([src, dst]), minlength=n).astype(float)
    deg_u[deg_u == 0] = 1.0
    deg_t = torch.from_numpy(deg_u).float()
    lv = edges[:, 3].astype(float)
    w = lv / lv.max()
    outdeg = np.bincount(src, minlength=n).astype(float)
    indeg = np.bincount(dst, minlength=n).astype(float)
    outdeg[outdeg == 0] = 1.0
    indeg[indeg == 0] = 1.0

    report = {}
    for frac in [float(x) for x in args.fractions.split(",")]:
        rows = {"lp": [], "ppr": [], "xgb": []}
        for trial in range(args.trials):
            rng = np.random.default_rng(100 + trial)
            seed_p = rng.choice(pos_candidates, size=max(1, int(len(pos_candidates) * frac)), replace=False)
            seed_n = rng.choice(neg_candidates, size=max(1, int(len(neg_candidates) * frac)), replace=False)
            seed = np.concatenate([seed_p, seed_n])
            hold = np.setdiff1d(np.arange(n), seed)
            hold = hold[cand_mask[hold]]
            y_hold = (y_pos[hold] == 1).astype(int) if args.label == 1 else y_pos[hold]

            # LP（金额加权双向扩散 + 种子夹紧）
            F = torch.zeros(n, 2)
            F[seed_p, 0] = 1.0
            F[seed_n, 1] = 1.0
            onehot = F.clone()
            for _ in range(100):
                msg = (F[src_t] + F[dst_t]) / 2.0
                new = torch.zeros(n, 2)
                idx_s = src_t.unsqueeze(1).expand(-1, 2)
                idx_d = dst_t.unsqueeze(1).expand(-1, 2)
                new.scatter_add_(0, idx_s, msg / deg_t[src_t].unsqueeze(1))
                new.scatter_add_(0, idx_d, msg / deg_t[dst_t].unsqueeze(1))
                new[seed] = onehot[seed]
                if torch.norm(new - F).item() < 1e-10:
                    F = new
                    break
                F = new
            s_hold = F[hold, 0].numpy()
            rows["lp"].append({"auc_roc": float(roc_auc_score(y_hold, s_hold)),
                               "auc_pr": float(average_precision_score(y_hold, s_hold)),
                               "recall_at_fpr1": float(recall_at_fpr(y_hold, s_hold)[0])})

            # PPR（双向，金额加权）
            s = torch.zeros(n)
            s[seed_p] = 1.0 / len(seed_p)
            c = s.clone()
            for _ in range(100):
                msg = (c[src_t] + c[dst_t]) / 2.0
                cn = torch.zeros(n)
                cn.scatter_add_(0, src_t, msg / deg_t[src_t])
                cn.scatter_add_(0, dst_t, msg / deg_t[dst_t])
                c = 0.85 * cn + 0.15 * s
            s_hold = c[hold].numpy()
            rows["ppr"].append({"auc_roc": float(roc_auc_score(y_hold, s_hold)),
                                "auc_pr": float(average_precision_score(y_hold, s_hold)),
                                "recall_at_fpr1": float(recall_at_fpr(y_hold, s_hold)[0])})

            # XGBoost（同种子）
            ytr = (y_pos[seed] == 1).astype(int) if args.label == 1 else y_pos[seed]
            m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                  subsample=0.9, colsample_bytree=0.8,
                                  scale_pos_weight=(ytr == 0).sum() / max((ytr == 1).sum(), 1),
                                  eval_metric="aucpr", tree_method="hist", random_state=0, n_jobs=-1)
            m.fit(feat[seed], ytr)
            p = m.predict_proba(feat[hold])[:, 1]
            rows["xgb"].append({"auc_roc": float(roc_auc_score(y_hold, p)),
                                "auc_pr": float(average_precision_score(y_hold, p)),
                                "recall_at_fpr1": float(recall_at_fpr(y_hold, p)[0])})

        report[frac] = {}
        for name in rows:
            agg = {}
            for metric in rows[name][0]:
                vals = [r[metric] for r in rows[name]]
                agg[metric] = [round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)]
            report[frac][name] = agg
        print(f"frac={frac}: " + json.dumps(report[frac], ensure_ascii=False))

    with open(os.path.join(OUT, "eth_same_window.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("saved ->", os.path.join(OUT, "eth_same_window.json"))


if __name__ == "__main__":
    main()
