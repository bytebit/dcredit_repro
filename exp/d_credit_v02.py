# -*- coding: utf-8 -*-
"""D-Credit v0.2：可学习信用×行为融合（M4 的实证形态）

score = w · z(LP信用) + (1-w) · z(XGB行为特征分)，w 在验证集上学习。
对照：LP / PPR / XGB / LR 融合（种子训练，含交互项）。
预期：Tornado 体制 w→高（结构主导），Phishing 体制 w→低（特征主导），融合全面 ≥ 单通道。
"""
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb

from metrics import recall_at_fpr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "bert4eth", "processed")
PROTO = os.path.join(BASE, "results", "eth_protocol.npz")
OUT = os.path.join(BASE, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", default="0.1,0.5")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--label", type=int, default=2)
    args = ap.parse_args()

    nd = np.load(os.path.join(PROC, "nodes.npz"), allow_pickle=True)
    ed = np.load(os.path.join(PROC, "edges.npz"))
    proto = np.load(PROTO)
    labels = nd["label"].astype(int)
    feat = nd["feat"].astype(np.float32)
    edges = ed["edges"]
    test_mask = proto["test_mask"]
    val_mask = proto["val_mask"]
    n = len(labels)
    y = np.full(n, -1, dtype=int)
    y[labels == 1] = 1
    y[labels == 0] = 0
    test_mask = test_mask & (y != -1)
    src = edges[:, 0].astype(int)
    dst = edges[:, 1].astype(int)
    deg_u = np.bincount(np.concatenate([src, dst]), minlength=n).astype(float)
    deg_u[deg_u == 0] = 1.0

    if args.label == 1:
        cand_mask = test_mask & (y != -1)
        y_pos = y
        val_cand = val_mask & ((labels == 1) | (labels == 0))
    else:
        cand_mask = (labels == args.label) | (labels == 0)
        y_pos = (labels == args.label).astype(int)
        val_cand = cand_mask
    pos_cand = np.where((y_pos == 1) & cand_mask)[0]
    neg_cand = np.where((y_pos == 0) & cand_mask)[0]
    val_idx = np.where(val_cand)[0]
    print(f"label={args.label}: pos_cand={len(pos_cand)} neg_cand={len(neg_cand)}")

    def lp_scores(seed_p, seed_n):
        Fm = np.zeros((n, 2))
        Fm[seed_p, 0] = 1.0
        Fm[seed_n, 1] = 1.0
        one = Fm.copy()
        seed = np.concatenate([seed_p, seed_n])
        for _ in range(100):
            msg = (Fm[src] + Fm[dst]) / 2.0
            new = np.zeros_like(Fm)
            np.add.at(new, src, msg / deg_u[src, None])
            np.add.at(new, dst, msg / deg_u[dst, None])
            new[seed] = one[seed]
            if np.abs(new - Fm).max() < 1e-10:
                Fm = new
                break
            Fm = new
        return Fm[:, 0]

    report = {}
    for frac in [float(x) for x in args.fractions.split(",")]:
        rows = {"lp": [], "ppr": [], "xgb": [], "gate_fusion": [], "lr_fusion": []}
        ws = []
        for trial in range(args.trials):
            rng = np.random.default_rng(500 + trial)
            seed_p = rng.choice(pos_cand, size=max(1, int(len(pos_cand) * frac)), replace=False)
            seed_n = rng.choice(neg_cand, size=max(1, int(len(neg_cand) * frac)), replace=False)
            seed = np.concatenate([seed_p, seed_n])
            hold = np.setdiff1d(np.arange(n), seed)
            hold = hold[cand_mask[hold]]
            y_hold = y_pos[hold]
            ytr = y_pos[seed]
            lp = lp_scores(seed_p, seed_n)

            s = np.zeros(n)
            s[seed_p] = 1.0 / len(seed_p)
            c = s.copy()
            for _ in range(100):
                msg = (c[src] + c[dst]) / 2.0
                cn = np.zeros(n)
                np.add.at(cn, src, msg / deg_u[src])
                np.add.at(cn, dst, msg / deg_u[dst])
                c = 0.85 * cn + 0.15 * s

            xm = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                   subsample=0.9, colsample_bytree=0.8,
                                   scale_pos_weight=(ytr == 0).sum() / max((ytr == 1).sum(), 1),
                                   eval_metric="aucpr", tree_method="hist", random_state=0, n_jobs=-1)
            xm.fit(feat[seed], ytr)
            xp = xm.predict_proba(feat)[:, 1]

            def met(sc):
                return {
                    "auc_roc": float(roc_auc_score(y_hold, sc)),
                    "auc_pr": float(average_precision_score(y_hold, sc)),
                    "recall_at_fpr1": float(recall_at_fpr(y_hold, sc)[0]),
                }

            rows["lp"].append(met(lp[hold]))
            rows["ppr"].append(met(c[hold]))
            rows["xgb"].append(met(xp[hold]))

            # 线性门控融合（w 在验证集上选）
            zv = lambda v: (v - v.mean()) / (v.std() + 1e-9)
            yv = y_pos[val_idx]
            lpv, xpv = zv(lp)[val_idx], zv(xp)[val_idx]
            best_w, best_auc = 0.0, -1.0
            for w in np.arange(0.0, 1.001, 0.05):
                auc = roc_auc_score(yv, w * lpv + (1 - w) * xpv)
                if auc > best_auc:
                    best_auc, best_w = auc, w
            sc = best_w * zv(lp) + (1 - best_w) * zv(xp)
            rows["gate_fusion"].append(met(sc[hold]))
            ws.append(round(float(best_w), 2))

            # LR 融合（种子训练，交互项）
            zl = zv(lp)[seed]
            zx = zv(xp)[seed]
            Xlr = np.stack([zl, zx, zl * zx], axis=1)
            lr = LogisticRegression(max_iter=1000)
            lr.fit(Xlr, ytr)
            zl_h = zv(lp)[hold]
            zx_h = zv(xp)[hold]
            Xlr_h = np.stack([zl_h, zx_h, zl_h * zx_h], axis=1)
            rows["lr_fusion"].append(met(lr.predict_proba(Xlr_h)[:, 1]))

        report[frac] = {}
        for name in rows:
            agg = {}
            for metric in rows[name][0]:
                vals = [r[metric] for r in rows[name]]
                agg[metric] = [round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)]
            report[frac][name] = agg
        report[frac]["learned_gate_w"] = ws
        print(f"frac={frac}: " + json.dumps(report[frac], ensure_ascii=False))

    with open(os.path.join(OUT, f"dcredit_v02_label{args.label}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("saved ->", os.path.join(OUT, f"dcredit_v02_label{args.label}.json"))


if __name__ == "__main__":
    main()
