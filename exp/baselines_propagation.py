# -*- coding: utf-8 -*-
"""Week2 - 传播基线：LP / PPR / APPNP（官方划分）

统一传播算子：与 PyG GCNConv 一致的有向 in-degree 对称归一化 S = D_in^{-1/2} A D_in^{-1/2}
（边 u→v 表示资金由 u 流向 v，v 聚合来自 u 的信用）。

口径：illicit=正类；种子 = 训练集 illicit（与 GNN 半监督设置一致，val/test 标签不可见）。
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import evaluate

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "elliptic", "processed", "elliptic.pt")
PROTO = os.path.join(BASE, "results", "elliptic_protocol.pt")
OUT = os.path.join(BASE, "results")


def build_operator(edge_index, n):
    """有向 in-degree 对称归一化 S（与 GCNConv 一致）。
    边 u→v 表示资金由 u 流向 v；v 聚合来自 u 的信用，故算子取 (v,u) 方向，
    (S @ F)[v] = sum_u S[v,u] F[u]。
    """
    u, v = edge_index
    deg_in = torch.bincount(v, minlength=n).float().clamp(min=1.0)
    vals = deg_in[u].pow(-0.5) * deg_in[v].pow(-0.5)
    return torch.sparse_coo_tensor(torch.stack([v, u]), vals, (n, n)).coalesce()


def load():
    d = torch.load(DATA, weights_only=False)
    proto = torch.load(PROTO, weights_only=False)
    return (
        d["x"],
        d["y"],
        d["edge_index"],
        proto["train_mask"].numpy(),
        proto["val_mask"].numpy(),
        proto["test_mask"].numpy(),
    )


def label_propagation(S, y, train_mask, n_iter=100, tol=1e-8):
    """LP：从训练标注出发迭代软标签，训练节点每轮夹紧。"""
    n = y.shape[0]
    F = torch.zeros(n, 2)
    F[(y == 0) & train_mask, 0] = 1.0
    F[(y == 1) & train_mask, 1] = 1.0
    onehot = F.clone()
    for _ in range(n_iter):
        new = torch.sparse.mm(S, F)
        new[train_mask] = onehot[train_mask]
        if torch.norm(new - F).item() < tol:
            F = new
            break
        F = new
    return F[:, 0].numpy()


def ppr(S, y, train_mask, alpha=0.15, n_iter=200, tol=1e-10):
    """PPR：从训练 illicit 种子（均匀分布）出发的个性化 PageRank。"""
    n = y.shape[0]
    seeds = np.where((y == 0) & train_mask)[0]
    s = torch.zeros(n)
    s[seeds] = 1.0 / max(len(seeds), 1)
    c = s.clone()
    for _ in range(n_iter):
        new = (1.0 - alpha) * torch.sparse.mm(S, c.unsqueeze(1)).squeeze(1) + alpha * s
        if torch.norm(new - c).item() < tol:
            c = new
            break
        c = new
    return c.numpy()


class APPNP(nn.Module):
    def __init__(self, in_dim, hidden, alpha=0.15, K=10, dropout=0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, 2)
        self.alpha = alpha
        self.K = K
        self.dropout = dropout

    def forward(self, x, S):
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin2(h)
        z = h
        for _ in range(self.K):
            z = (1.0 - self.alpha) * torch.sparse.mm(S, z) + self.alpha * h
        return z


def train_appnp(S, x, y, train_mask, val_mask, test_mask, device, hidden=64, lr=1e-3, wd=5e-4,
                epochs=600, patience=80, seed=0, alpha=0.15, K=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = APPNP(x.size(1), hidden, alpha=alpha, K=K).to(device)
    n_pos = (y[train_mask] == 1).sum().float()
    n_neg = (y[train_mask] == 0).sum().float()
    class_w = torch.tensor([1.0, (n_neg / n_pos).item()], device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val, best_state, bad = -1.0, None, 0
    yv = (y[val_mask] == 0).cpu().numpy().astype(int)
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(x, S)
        loss = F.cross_entropy(logits[train_mask], y[train_mask], weight=class_w)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.softmax(model(x, S), dim=-1)
        score_v = (1.0 - pv[val_mask, 1]).cpu().numpy()
        val = float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(yv, score_v))
        if val > best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(x, S), dim=-1)
    return (1.0 - p[val_mask, 1]).cpu().numpy(), (1.0 - p[test_mask, 1]).cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y, edge_index, train_mask, val_mask, test_mask = load()
    n = x.shape[0]
    S = build_operator(edge_index, n)
    y_metric = (y.numpy() == 0).astype(int)
    yv_m, yte_m = y_metric[val_mask], y_metric[test_mask]

    results = {}

    # LP
    t0 = time.time()
    sc_lp = label_propagation(S, y.numpy(), train_mask)
    results["lp"] = evaluate(yte_m, sc_lp[test_mask], yv_m, sc_lp[val_mask])
    results["lp"]["seconds"] = round(time.time() - t0, 2)
    print("[lp]", json.dumps(results["lp"], ensure_ascii=False))

    # PPR
    t0 = time.time()
    sc_ppr = ppr(S, y.numpy(), train_mask)
    results["ppr"] = evaluate(yte_m, sc_ppr[test_mask], yv_m, sc_ppr[val_mask])
    results["ppr"]["seconds"] = round(time.time() - t0, 2)
    print("[ppr]", json.dumps(results["ppr"], ensure_ascii=False))

    # APPNP
    t0 = time.time()
    xs, ys, eis = x.to(device), y.to(device), S.to(device)
    trm = torch.from_numpy(train_mask).to(device)
    vam = torch.from_numpy(val_mask).to(device)
    tem = torch.from_numpy(test_mask).to(device)
    pv, pte = train_appnp(eis, xs, ys, trm, vam, tem, device)
    results["appnp"] = evaluate(yte_m, pte, yv_m, pv)
    results["appnp"]["seconds"] = round(time.time() - t0, 2)
    print("[appnp]", json.dumps(results["appnp"], ensure_ascii=False))

    with open(os.path.join(OUT, "propagation_baselines.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved ->", os.path.join(OUT, "propagation_baselines.json"))


if __name__ == "__main__":
    main()
