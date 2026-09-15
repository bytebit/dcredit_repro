# -*- coding: utf-8 -*-
"""Week2 - 快照内传播评测（Elliptic 结构修正后的正确实验）

背景：Elliptic 234,355 条边 100% 连接同一时间步内交易 → 官方时间划分下训练/测试无图连接，
跨时间传播结构性无信号。本脚本在"单时间步快照"内评测传播方法：
给定快照内少量 illicit 种子（与 licit 种子），能否通过图传播找到同快照内其余 illicit。

模型：LP（软标签夹紧传播）/ PPR（illicit 种子个性化 PageRank）/ XGBoost（同种子特征基线）
指标（illicit=正类）：AUC-ROC、fraud AP、Recall@FPR=1%、Top-k 召回（k=持有 illicit 数）
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb

from metrics import recall_at_fpr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "elliptic", "processed", "elliptic.pt")
OUT = os.path.join(BASE, "results")


def build_snapshot(y, t, ei, k):
    """取时间步 k 的快照：节点、特征、边；返回节点索引（原图）、邻接、标签。"""
    nodes = np.where(t == k)[0]
    node_set = set(nodes.tolist())
    u, v = ei
    keep = np.array([u[i] in node_set and v[i] in node_set for i in range(len(u))], dtype=bool)
    return nodes, np.stack([u[keep], v[keep]])


def operator_from_edges(ei_sub, n_local):
    """有向 in-degree 对称归一化算子（本地索引），资金流方向 u→v。"""
    u, v = ei_sub
    deg = np.bincount(v, minlength=n_local).astype(np.float64)
    deg[deg == 0] = 1.0
    vals = deg[u] ** -0.5 * deg[v] ** -0.5
    return torch.sparse_coo_tensor(torch.from_numpy(np.stack([v, u])), torch.from_numpy(vals),
                                   (n_local, n_local)).coalesce().float()


def label_propagation(S, seed_ill, seed_lic, n, n_iter=200):
    F = torch.zeros(n, 2)
    F[seed_ill, 0] = 1.0
    F[seed_lic, 1] = 1.0
    onehot = F.clone()
    seed = np.concatenate([seed_ill, seed_lic])
    for _ in range(n_iter):
        new = torch.sparse.mm(S, F)
        new[seed] = onehot[seed]
        if torch.norm(new - F).item() < 1e-10:
            F = new
            break
        F = new
    return F[:, 0].numpy()


def ppr(S, seed_ill, n, alpha=0.15, n_iter=500):
    s = torch.zeros(n)
    s[seed_ill] = 1.0 / max(len(seed_ill), 1)
    c = s.clone()
    for _ in range(n_iter):
        new = (1.0 - alpha) * torch.sparse.mm(S, c.unsqueeze(1)).squeeze(1) + alpha * s
        if torch.norm(new - c).item() < 1e-12:
            c = new
            break
        c = new
    return c.numpy()


def xgb_score(X, y_loc, seed_idx, holdout_idx):
    Xtr = X[seed_idx]
    ytr = (y_loc[seed_idx] == 0).astype(int)
    scale = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          subsample=0.9, colsample_bytree=0.8,
                          scale_pos_weight=scale, eval_metric="aucpr",
                          tree_method="hist", random_state=0, n_jobs=-1)
    m.fit(Xtr, ytr)
    return m.predict_proba(X[holdout_idx])[:, 1]


def topk_recall(y_hold, score, k):
    order = np.argsort(-score, kind="stable")[:k]
    n_pos = int((y_hold == 1).sum())
    return float((y_hold[order] == 1).sum() / max(n_pos, 1))


def run_snapshot(X, y, t, ei, k, frac, trial, seed_rng):
    nodes, ei_sub = build_snapshot(y, t, ei, k)
    n = len(nodes)
    idx = {int(node): i for i, node in enumerate(nodes)}
    u, v = ei_sub
    u_l = np.array([idx[int(a)] for a in u])
    v_l = np.array([idx[int(b)] for b in v])
    y_loc = y[nodes]
    ill = np.where(y_loc == 0)[0]
    lic = np.where(y_loc == 1)[0]
    n_ill_seed = max(1, int(round(len(ill) * frac)))
    n_lic_seed = max(1, int(round(len(lic) * frac)))
    rng = np.random.default_rng(seed_rng * 1000 + trial)
    seed_ill = rng.choice(ill, size=n_ill_seed, replace=False)
    seed_lic = rng.choice(lic, size=n_lic_seed, replace=False)
    seed = np.concatenate([seed_ill, seed_lic])
    hold = np.setdiff1d(np.arange(n), seed)
    y_hold = (y_loc[hold] == 0).astype(int)

    S = operator_from_edges(np.stack([u_l, v_l]), n)
    res = {}

    sc = label_propagation(S, seed_ill, seed_lic, n)
    s_hold = sc[hold]
    res["lp"] = {
        "auc_roc": float(roc_auc_score(y_hold, s_hold)),
        "auc_pr": float(average_precision_score(y_hold, s_hold)),
        "recall_at_fpr1": float(recall_at_fpr(y_hold, s_hold)[0]),
        "topk_recall": topk_recall(y_hold, s_hold, int((y_hold == 1).sum())),
    }

    sc = ppr(S, seed_ill, n)
    s_hold = sc[hold]
    res["ppr"] = {
        "auc_roc": float(roc_auc_score(y_hold, s_hold)),
        "auc_pr": float(average_precision_score(y_hold, s_hold)),
        "recall_at_fpr1": float(recall_at_fpr(y_hold, s_hold)[0]),
        "topk_recall": topk_recall(y_hold, s_hold, int((y_hold == 1).sum())),
    }

    s_hold = xgb_score(X[nodes], y_loc, seed, hold)
    res["xgb"] = {
        "auc_roc": float(roc_auc_score(y_hold, s_hold)),
        "auc_pr": float(average_precision_score(y_hold, s_hold)),
        "recall_at_fpr1": float(recall_at_fpr(y_hold, s_hold)[0]),
        "topk_recall": topk_recall(y_hold, s_hold, int((y_hold == 1).sum())),
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time_steps", default="13,29,32,42")
    ap.add_argument("--fractions", default="0.1,0.2,0.5")
    ap.add_argument("--trials", type=int, default=5)
    args = ap.parse_args()

    d = torch.load(DATA, weights_only=False)
    X, y, t, ei = d["x"].numpy(), d["y"].numpy(), d["time_step"].numpy(), d["edge_index"].numpy()
    steps = [int(x) for x in args.time_steps.split(",")]
    fracs = [float(x) for x in args.fractions.split(",")]

    report = {}
    for k in steps:
        report[k] = {}
        for frac in fracs:
            rows = {"lp": [], "ppr": [], "xgb": []}
            for trial in range(args.trials):
                r = run_snapshot(X, y, t, ei, k, frac, trial, seed_rng=7)
                for name in rows:
                    rows[name].append(r[name])
            report[k][frac] = {}
            for name in rows:
                agg = {}
                for metric in rows[name][0]:
                    vals = [rr[metric] for rr in rows[name]]
                    agg[metric] = [round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)]
                report[k][frac][name] = agg
            print(f"t{k} frac={frac}: " + json.dumps(report[k][frac], ensure_ascii=False))

    with open(os.path.join(OUT, "snapshot_propagation.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("saved ->", os.path.join(OUT, "snapshot_propagation.json"))


if __name__ == "__main__":
    main()
