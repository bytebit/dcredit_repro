# -*- coding: utf-8 -*-
"""D-Credit v0（BERT4ETH 账户图）

核心：从训练期钓鱼种子出发的方向感知可学习信用传播
  c_{t+1} = (1-α) [ γ_out·S_out·c + γ_in·S_in·c ] + α·s
  S_out/S_in：金额加权（(log1p val)^β）的行归一化传播算子（出/入方向）
信用分数与节点特征拼接 → MLP 分类（phishing vs normal）

同场基线：XGBoost（特征）、PPR 分数 + MLP、LP（标签传播）
指标（illicit=phishing 正类）：AUC-ROC / AP / Recall@FPR=1%，另报"可达子集"指标
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb

from metrics import evaluate, recall_at_fpr, val_tuned_f1

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "bert4eth", "processed")
PROTO = os.path.join(BASE, "results", "eth_protocol.npz")
OUT = os.path.join(BASE, "results")


def load():
    nd = np.load(os.path.join(PROC, "nodes.npz"), allow_pickle=True)
    ed = np.load(os.path.join(PROC, "edges.npz"))
    proto = np.load(PROTO)
    return nd, ed, proto


def build_edge_data(edges, n):
    """返回边数据：src/dst、每边归一化金额权重 w、方向 base 权重（行归一化）。"""
    src = edges[:, 0].astype(np.int64)
    dst = edges[:, 1].astype(np.int64)
    lv = edges[:, 3].astype(np.float64)
    outdeg = np.bincount(src, minlength=n).astype(np.float64)
    indeg = np.bincount(dst, minlength=n).astype(np.float64)
    outdeg[outdeg == 0] = 1.0
    indeg[indeg == 0] = 1.0
    w = lv / max(lv.max(), 1e-9)  # 归一化金额权重（每边）
    return (
        torch.from_numpy(src).long(),
        torch.from_numpy(dst).long(),
        torch.from_numpy(w).float(),
        torch.from_numpy(1.0 / outdeg[src]).float(),   # 沿资金方向：dst ← src
        torch.from_numpy(1.0 / indeg[dst]).float(),    # 反向：src ← dst
    )


class Dcredit(nn.Module):
    def __init__(self, in_dim, hidden=64, K=6):
        super().__init__()
        self.logit_alpha = nn.Parameter(torch.tensor(0.0))
        self.logit_gamma_out = nn.Parameter(torch.tensor(0.0))
        self.logit_gamma_in = nn.Parameter(torch.tensor(0.0))
        self.log_beta = nn.Parameter(torch.tensor(0.0))  # 金额指数（0 表示无金额加权）
        self.K = K
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + 1, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, 2),
        )

    def forward(self, feat, src, dst, w, w_out, w_in, seed):
        alpha = torch.sigmoid(self.logit_alpha)
        g = torch.softmax(torch.stack([self.logit_gamma_out, self.logit_gamma_in]), dim=0)
        g_out, g_in = g[0], g[1]
        beta = torch.exp(self.log_beta)
        wb = w.pow(beta)
        c = seed.clone()
        for _ in range(self.K):
            c_out = torch.zeros_like(c)
            c_out.scatter_add_(0, dst, w_out * wb * c[src])
            c_in = torch.zeros_like(c)
            c_in.scatter_add_(0, src, w_in * wb * c[dst])
            c = (1.0 - alpha) * (g_out * c_out + g_in * c_in) + alpha * seed
        cred = torch.log1p(c)
        cred = cred / (cred.max() + 1e-8)
        return self.mlp(torch.cat([feat, cred.unsqueeze(1)], dim=-1)), cred


def train_model(model, feat, seed, src, dst, w, w_out, w_in, y_t, mask_t, mask_v, device,
                epochs=400, patience=60, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_pos = (y_t[mask_t] == 1).sum().float()
    n_neg = (y_t[mask_t] == 0).sum().float()
    cw = torch.tensor([1.0, (n_neg / max(n_pos, 1)).item()], device=device)
    yv = y_t[mask_v].cpu().numpy()
    best_val, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits, _ = model(feat, src, dst, w, w_out, w_in, seed)
        loss = F.cross_entropy(logits[mask_t], y_t[mask_t], weight=cw)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.softmax(model(feat, src, dst, w, w_out, w_in, seed)[0], dim=-1)
        sc = (pv[mask_v, 1]).cpu().numpy()
        val = roc_auc_score(yv, sc)
        if val > best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=400)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    patience = 60

    nd, ed, proto = load()
    labels = nd["label"].astype(int)
    feat = nd["feat"].astype(np.float32)
    edges = ed["edges"]
    train_mask = proto["train_mask"]
    val_mask = proto["val_mask"]
    test_mask = proto["test_mask"]
    n = len(labels)

    # 二分类：phishing=1, normal=0（其余不参与监督）
    y = np.full(n, -1, dtype=np.int64)
    y[labels == 1] = 1
    y[labels == 0] = 0
    sup = y != -1
    train_mask = train_mask & sup
    val_mask = val_mask & sup
    test_mask = test_mask & sup
    y_t = torch.from_numpy(y).long().to(device)
    trm = torch.from_numpy(train_mask).to(device)
    vam = torch.from_numpy(val_mask).to(device)
    tem = torch.from_numpy(test_mask).to(device)

    src, dst, w, w_out, w_in = build_edge_data(edges, n)
    src, dst, w, w_out, w_in = (x.to(device) for x in (src, dst, w, w_out, w_in))

    seed = torch.zeros(n, device=device)
    seeds = np.where((labels == 1) & train_mask)[0]
    seed[seeds] = 1.0 / max(len(seeds), 1)

    feat_t = torch.from_numpy(feat).to(device)
    y_te = (y[test_mask] == 1).astype(int)
    y_va = (y[val_mask] == 1).astype(int)
    results = {}

    # ---- D-Credit ----
    torch.manual_seed(0)
    model = Dcredit(feat.shape[1]).to(device)
    t0 = time.time()
    best_val = train_model(model, feat_t, seed, src, dst, w, w_out, w_in, y_t, trm, vam, device,
                           epochs=args.epochs)
    model.eval()
    with torch.no_grad():
        logits, cred = model(feat_t, src, dst, w, w_out, w_in, seed)
    pte = torch.softmax(logits, dim=-1)[tem, 1].cpu().numpy()
    pva = torch.softmax(logits, dim=-1)[vam, 1].cpu().numpy()
    res = evaluate(y_te, pte, y_va, pva)
    res["val_auc"] = float(best_val)
    res["seconds"] = round(time.time() - t0, 1)
    results["dcredit"] = res
    print("[dcredit]", json.dumps(res, ensure_ascii=False))

    # ---- PPR 固定分数 + MLP（用 PPR 分数替换可学习信用） ----
    def ppr_seeds(S, seed_idx, alpha=0.15, n_iter=500):
        s = torch.zeros(n)
        s[seed_idx] = 1.0 / max(len(seed_idx), 1)
        c = s.clone()
        for _ in range(n_iter):
            new = (1.0 - alpha) * torch.sparse.mm(S, c.unsqueeze(1)).squeeze(1) + alpha * s
            if torch.norm(new - c).item() < 1e-12:
                c = new
                break
            c = new
        return c.numpy()

    # 无向 PPR 用 scatter 版（CPU）
    def ppr_undirected(src_c, dst_c, seed_idx, alpha=0.15, n_iter=500):
        deg = torch.zeros(n)
        deg.scatter_add_(0, src_c, torch.ones_like(src_c).float())
        deg.scatter_add_(0, dst_c, torch.ones_like(dst_c).float())
        deg = deg.clamp(min=1.0)
        s = torch.zeros(n)
        s[seed_idx] = 1.0 / max(len(seed_idx), 1)
        c = s.clone()
        for _ in range(n_iter):
            msg = (c[src_c] + c[dst_c]) / 2.0
            c_new = torch.zeros(n)
            c_new.scatter_add_(0, src_c, msg / deg[src_c])
            c_new.scatter_add_(0, dst_c, msg / deg[dst_c])
            c = (1.0 - alpha) * c_new + alpha * s
        return c.numpy()

    c_ppr = ppr_undirected(src.cpu(), dst.cpu(), np.where((labels == 1) & train_mask)[0], alpha=0.15)
    c_ppr = np.log1p(c_ppr)
    c_ppr = c_ppr / (c_ppr.max() + 1e-8)
    feat_ppr = torch.from_numpy(np.concatenate([feat, c_ppr.reshape(-1, 1)], axis=1).astype(np.float32)).to(device)
    torch.manual_seed(0)
    model2 = Dcredit(feat.shape[1] + 1).to(device)
    # 用固定 PPR 分数：直接训练 MLP（credit 分支设为常数）
    model2.mlp = nn.Sequential(
        nn.Linear(feat.shape[1] + 1, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2),
    ).to(device)
    opt = torch.optim.Adam(model2.parameters(), lr=1e-3)
    n_pos = (y_t[trm] == 1).sum().float()
    n_neg = (y_t[trm] == 0).sum().float()
    cw = torch.tensor([1.0, (n_neg / max(n_pos, 1)).item()], device=device)
    best_val2, best_state2, bad = -1.0, None, 0
    for ep in range(args.epochs):
        model2.train()
        opt.zero_grad()
        logits2 = model2.mlp(feat_ppr)
        loss = F.cross_entropy(logits2[trm], y_t[trm], weight=cw)
        loss.backward()
        opt.step()
        model2.eval()
        with torch.no_grad():
            pv = torch.softmax(model2.mlp(feat_ppr), dim=-1)
        val = roc_auc_score(y_va, pv[vam, 1].cpu().numpy())
        if val > best_val2:
            best_val2 = val
            best_state2 = {k: v.detach().cpu().clone() for k, v in model2.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model2.load_state_dict(best_state2)
    model2.eval()
    with torch.no_grad():
        pte2 = torch.softmax(model2.mlp(feat_ppr), dim=-1)[tem, 1].cpu().numpy()
        pva2 = torch.softmax(model2.mlp(feat_ppr), dim=-1)[vam, 1].cpu().numpy()
    res2 = evaluate(y_te, pte2, y_va, pva2)
    results["mlp_ppr"] = res2
    print("[mlp+ppr]", json.dumps(res2, ensure_ascii=False))

    # ---- MLP 仅特征 ----
    torch.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(feat.shape[1], 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2)).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    best_val3, best_state3, bad = -1.0, None, 0
    for ep in range(args.epochs):
        mlp.train()
        opt.zero_grad()
        logits3 = mlp(feat_t)
        loss = F.cross_entropy(logits3[trm], y_t[trm], weight=cw)
        loss.backward()
        opt.step()
        mlp.eval()
        with torch.no_grad():
            pv = torch.softmax(mlp(feat_t), dim=-1)
        val = roc_auc_score(y_va, pv[vam, 1].cpu().numpy())
        if val > best_val3:
            best_val3 = val
            best_state3 = {k: v.detach().cpu().clone() for k, v in mlp.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    mlp.load_state_dict(best_state3)
    mlp.eval()
    with torch.no_grad():
        pte3 = torch.softmax(mlp(feat_t), dim=-1)[tem, 1].cpu().numpy()
        pva3 = torch.softmax(mlp(feat_t), dim=-1)[vam, 1].cpu().numpy()
    res3 = evaluate(y_te, pte3, y_va, pva3)
    results["mlp_feat"] = res3
    print("[mlp_feat]", json.dumps(res3, ensure_ascii=False))

    # ---- XGBoost（特征） ----
    xm = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9,
                           colsample_bytree=0.8, scale_pos_weight=(y[train_mask] == 0).sum() / max((y[train_mask] == 1).sum(), 1),
                           eval_metric="aucpr", tree_method="hist", random_state=0, n_jobs=-1)
    xm.fit(feat[train_mask], (y[train_mask] == 1).astype(int))
    pte4 = xm.predict_proba(feat[test_mask])[:, 1]
    pva4 = xm.predict_proba(feat[val_mask])[:, 1]
    res4 = evaluate(y_te, pte4, y_va, pva4)
    results["xgb"] = res4
    print("[xgb]", json.dumps(res4, ensure_ascii=False))

    with open(os.path.join(OUT, "dcredit_eth.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved ->", os.path.join(OUT, "dcredit_eth.json"))


if __name__ == "__main__":
    main()
