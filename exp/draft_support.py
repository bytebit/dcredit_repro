# -*- coding: utf-8 -*-
"""论文初稿支撑实验

1) Elliptic t29 门控融合（跨数据集一致性）：LP 信用 + XGB 特征，w 在种子内 20% 验证集选择
2) 方向消融（E6）：无向对称 vs 有向出 vs 有向入（Tornado / Elliptic t29）
3) 可解释案例：Tornado 高信用 held-out 账户回溯到最近种子的路径
"""
import json
import os
from collections import deque

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb

from metrics import recall_at_fpr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results")


def zscore(v):
    return (v - v.mean()) / (v.std() + 1e-9)


def lp_undirected(c0, src, dst, deg_u, n, n_iter=100):
    c = c0.copy()
    seed_idx = np.where(c0 != 0)[0]
    one = c0.copy()
    for _ in range(n_iter):
        msg = (c[src] + c[dst]) / 2.0
        new = np.zeros(n)
        np.add.at(new, src, msg / deg_u[src])
        np.add.at(new, dst, msg / deg_u[dst])
        new[seed_idx] = one[seed_idx]
        if np.abs(new - c).max() < 1e-10:
            c = new
            break
        c = new
    return c


def lp_directed(c0, src, dst, deg, n, direction, n_iter=100):
    c = c0.copy()
    seed_idx = np.where(c0 != 0)[0]
    one = c0.copy()
    for _ in range(n_iter):
        if direction == "out":  # 资金沿 src→dst：dst 从 src 收
            new = np.zeros(n)
            np.add.at(new, dst, c[src] / deg[src])
        else:  # in：src 从 dst 收
            new = np.zeros(n)
            np.add.at(new, src, c[dst] / deg[dst])
        new[seed_idx] = one[seed_idx]
        if np.abs(new - c).max() < 1e-10:
            c = new
            break
        c = new
    return c


def ellip_t29_fusion():
    d = torch.load(os.path.join(BASE, "data", "elliptic", "processed", "elliptic.pt"), weights_only=False)
    X, y, t, ei = d["x"].numpy(), d["y"].numpy(), d["time_step"].numpy(), d["edge_index"].numpy()
    k = 29
    nodes = np.where(t == k)[0]
    node_set = set(nodes.tolist())
    u, v = ei
    keep = np.array([u[i] in node_set and v[i] in node_set for i in range(len(u))], dtype=bool)
    u_l, v_l = u[keep], v[keep]
    idx = {int(x): i for i, x in enumerate(nodes)}
    us = np.array([idx[int(a)] for a in u_l])
    vs = np.array([idx[int(b)] for b in v_l])
    n = len(nodes)
    y_loc = y[nodes]
    ill = np.where(y_loc == 0)[0]
    lic = np.where(y_loc == 1)[0]
    deg_u = np.bincount(np.concatenate([us, vs]), minlength=n).astype(float)
    deg_u[deg_u == 0] = 1.0
    outdeg = np.bincount(us, minlength=n).astype(float)
    indeg = np.bincount(vs, minlength=n).astype(float)
    outdeg[outdeg == 0] = 1.0
    indeg[indeg == 0] = 1.0

    report = {}
    for frac in [0.1, 0.5]:
        rows = {"lp": [], "xgb": [], "gate_fusion": []}
        ws = []
        for trial in range(3):
            rng = np.random.default_rng(700 + trial)
            sp = rng.choice(ill, size=max(1, int(len(ill) * frac)), replace=False)
            sn = rng.choice(lic, size=max(1, int(len(lic) * frac)), replace=False)
            seed = np.concatenate([sp, sn])
            hold = np.setdiff1d(np.arange(n), seed)
            y_hold = (y_loc[hold] == 0).astype(int)
            ytr = (y_loc[seed] == 0).astype(int)

            c0 = np.zeros(n)
            c0[sp] = 1.0 / len(sp)
            lp = lp_undirected(c0, us, vs, deg_u, n)
            xm = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                   subsample=0.9, colsample_bytree=0.8,
                                   scale_pos_weight=(ytr == 0).sum() / max((ytr == 1).sum(), 1),
                                   eval_metric="aucpr", tree_method="hist", random_state=0, n_jobs=-1)
            xm.fit(X[nodes][seed], ytr)
            xp = xm.predict_proba(X[nodes])[:, 1]

            def met(sc):
                return {
                    "auc_roc": float(roc_auc_score(y_hold, sc)),
                    "recall_at_fpr1": float(recall_at_fpr(y_hold, sc)[0]),
                }

            rows["lp"].append(met(lp[hold]))
            rows["xgb"].append(met(xp[hold]))
            # 门控：种子内 20% 验证，分别按 AUC / AP 选 w
            perm = rng.permutation(len(seed))
            val_s = seed[perm[: max(1, len(seed) // 5)]]
            yv = (y_loc[val_s] == 0).astype(int)
            lpv, xpv = zscore(lp)[val_s], zscore(xp)[val_s]
            best_w_auc, best_auc = 0.0, -1.0
            best_w_ap, best_ap = 0.0, -1.0
            for w in np.arange(0.0, 1.001, 0.05):
                sc_v = w * lpv + (1 - w) * xpv
                auc = roc_auc_score(yv, sc_v)
                ap = average_precision_score(yv, sc_v)
                if auc > best_auc:
                    best_auc, best_w_auc = auc, w
                if ap > best_ap:
                    best_ap, best_w_ap = ap, w
            sc = best_w_ap * zscore(lp) + (1 - best_w_ap) * zscore(xp)
            rows["gate_fusion"].append(met(sc[hold]))
            ws.append([round(float(best_w_auc), 2), round(float(best_w_ap), 2)])
        report[frac] = {}
        for name in rows:
            report[frac][name] = {
                kk: [round(float(np.mean([r[kk] for r in rows[name]])), 4),
                     round(float(np.std([r[kk] for r in rows[name]])), 4)]
                for kk in rows[name][0]
            }
        report[frac]["learned_w"] = ws
        print("ellip_t29", frac, json.dumps(report[frac], ensure_ascii=False))
    return report


def direction_ablation():
    """Tornado 上无向/有向出/有向入 LP 对比。"""
    nd = np.load(os.path.join(BASE, "data", "bert4eth", "processed", "nodes.npz"), allow_pickle=True)
    ed = np.load(os.path.join(BASE, "data", "bert4eth", "processed", "edges.npz"))
    labels = nd["label"].astype(int)
    edges = ed["edges"]
    n = len(labels)
    cand = (labels == 2) | (labels == 0)
    ypos = (labels == 2).astype(int)
    pos = np.where((labels == 2) & cand)[0]
    neg = np.where((labels == 0) & cand)[0]
    src = edges[:, 0].astype(int)
    dst = edges[:, 1].astype(int)
    deg_u = np.bincount(np.concatenate([src, dst]), minlength=n).astype(float)
    deg_u[deg_u == 0] = 1.0
    outdeg = np.bincount(src, minlength=n).astype(float)
    indeg = np.bincount(dst, minlength=n).astype(float)
    outdeg[outdeg == 0] = 1.0
    indeg[indeg == 0] = 1.0
    rng = np.random.default_rng(800)
    sp = rng.choice(pos, size=278, replace=False)
    sn = rng.choice(neg, size=697, replace=False)
    seed = np.concatenate([sp, sn])
    hold = np.setdiff1d(np.arange(n), seed)
    hold = hold[cand[hold]]
    yh = ypos[hold]
    c0 = np.zeros(n)
    c0[sp] = 1.0 / len(sp)
    res = {}
    for name, sc in [
        ("undirected", lp_undirected(c0, src, dst, deg_u, n)),
        ("directed_out", lp_directed(c0, src, dst, outdeg, n, "out")),
        ("directed_in", lp_directed(c0, src, dst, indeg, n, "in")),
    ]:
        res[name] = {
            "auc": round(float(roc_auc_score(yh, sc[hold])), 4),
            "rec@fpr1": round(float(recall_at_fpr(yh, sc[hold])[0]), 4),
        }
    print("direction_ablation(tornado):", json.dumps(res, ensure_ascii=False))
    return res


def case_study():
    """Tornado 可解释案例：高信用 held-out 账户 → 最近种子路径。"""
    nd = np.load(os.path.join(BASE, "data", "bert4eth", "processed", "nodes.npz"), allow_pickle=True)
    ed = np.load(os.path.join(BASE, "data", "bert4eth", "processed", "edges.npz"))
    labels = nd["label"].astype(int)
    edges = ed["edges"]
    n = len(labels)
    cand = (labels == 2) | (labels == 0)
    pos = np.where((labels == 2) & cand)[0]
    neg = np.where((labels == 0) & cand)[0]
    src = edges[:, 0].astype(int)
    dst = edges[:, 1].astype(int)
    deg_u = np.bincount(np.concatenate([src, dst]), minlength=n).astype(float)
    deg_u[deg_u == 0] = 1.0
    rng = np.random.default_rng(800)
    sp = rng.choice(pos, size=278, replace=False)
    sn = rng.choice(neg, size=697, replace=False)
    seed = np.concatenate([sp, sn])
    c0 = np.zeros(n)
    c0[sp] = 1.0 / len(sp)
    lp = lp_undirected(c0, src, dst, deg_u, n)
    hold = np.setdiff1d(np.arange(n), seed)
    hold = hold[cand[hold]]
    tornado_hold = hold[labels[hold] == 2]
    top = tornado_hold[np.argsort(-lp[tornado_hold])[:3]]
    # BFS 无向最近种子
    adj = [[] for _ in range(n)]
    for a, b in zip(src, dst):
        adj[a].append(b)
        adj[b].append(a)
    seed_set = set(sp.tolist())
    out = []
    for node in top:
        parent = {node: -1}
        q = deque([node])
        found = None
        while q and found is None:
            x = q.popleft()
            for nb in adj[x]:
                if nb in parent:
                    continue
                parent[nb] = x
                if nb in seed_set:
                    found = nb
                    break
                q.append(nb)
        path = []
        cur = found
        while cur != -1:
            path.append(cur)
            cur = parent.get(cur, -1)
        addr = nd["address"]
        out.append({
            "account": str(addr[node])[:12],
            "credit": round(float(lp[node]), 6),
            "path_len": len(path),
            "path": [str(addr[p])[:10] for p in path],
        })
    print("case_study:", json.dumps(out, ensure_ascii=False))
    return out


if __name__ == "__main__":
    results = {
        "ellip_t29_fusion": ellip_t29_fusion(),
        "direction_ablation": direction_ablation(),
        "case_study": case_study(),
    }
    with open(os.path.join(OUT, "draft_support.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved ->", os.path.join(OUT, "draft_support.json"))
