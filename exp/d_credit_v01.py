# -*- coding: utf-8 -*-
"""D-Credit v0.1：修正版（Tornado / Phishing 同窗口验证）

修正点（对应 v0 三条失败模式）：
  1) 算子族选择：γ = softmax(无向对称, 有向出, 有向入)；
  2) 独立信用门控 w_c（初始 0 → 从特征模型起步，可学习打开/关闭信用）；
  3) 图平滑一致性辅助损失（分类概率沿边平滑，λ_edge），弥补 CE 监督缺口。

对照：LP / PPR / XGBoost / MLP 特征 / APPNP（预测传播，CE 端到端）。
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb

from metrics import recall_at_fpr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "bert4eth", "processed")
PROTO = os.path.join(BASE, "results", "eth_protocol.npz")
OUT = os.path.join(BASE, "results")


class DCreditV01(nn.Module):
    def __init__(self, in_dim, hidden=48, K=5):
        super().__init__()
        self.logit_alpha = nn.Parameter(torch.tensor(-2.0))  # α≈0.12 起步
        self.logit_gamma = nn.Parameter(torch.zeros(3))      # 无向/有向出/有向入
        self.log_beta = nn.Parameter(torch.tensor(0.0))
        self.w_credit = nn.Parameter(torch.tensor(0.0))      # 信用门控，初始 0
        self.K = K
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, 2),
        )

    def propagate(self, src, dst, w_u_s, w_u_d, w_o, w_i, wb, seed):
        alpha = torch.sigmoid(self.logit_alpha)
        g = torch.softmax(self.logit_gamma, dim=0)  # (无向, 出, 入)
        c = seed.clone()
        for _ in range(self.K):
            # 无向对称
            msg = (c[src] + c[dst]) / 2.0
            cu_s = torch.zeros_like(c)
            cu_s.scatter_add_(0, src, w_u_s * wb * msg)
            cu_d = torch.zeros_like(c)
            cu_d.scatter_add_(0, dst, w_u_d * wb * msg)
            # 有向出/入
            co = torch.zeros_like(c)
            co.scatter_add_(0, dst, w_o * wb * c[src])
            ci = torch.zeros_like(c)
            ci.scatter_add_(0, src, w_i * wb * c[dst])
            c = (1.0 - alpha) * (g[0] * (cu_s + cu_d) + g[1] * co + g[2] * ci) + alpha * seed
        return c

    def forward(self, feat, src, dst, w_u_s, w_u_d, w_o, w_i, wb, seed):
        c = self.propagate(src, dst, w_u_s, w_u_d, w_o, w_i, wb, seed)
        cred = torch.log1p(c)
        cred = cred / (cred.max() + 1e-8)
        logits = self.mlp(feat)
        logits[:, 1] = logits[:, 1] + self.w_credit * cred
        return logits, c

    def gate_values(self):
        alpha = torch.sigmoid(self.logit_alpha).item()
        g = torch.softmax(self.logit_gamma, dim=0).tolist()
        return {
            "alpha": round(alpha, 3),
            "gamma_undirected": round(g[0], 3),
            "gamma_out": round(g[1], 3),
            "gamma_in": round(g[2], 3),
            "beta": round(float(torch.exp(self.log_beta).item()), 3),
            "w_credit": round(float(self.w_credit.item()), 3),
        }


class APPNP(nn.Module):
    def __init__(self, in_dim, hidden=48, K=10, alpha=0.15):
        super().__init__()
        self.K = K
        self.alpha = alpha
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, 2),
        )

    def forward(self, feat, src, dst, w_u_s, w_u_d):
        z = self.mlp(feat)
        z0 = z
        for _ in range(self.K):
            msg = (z[src] + z[dst]) / 2.0
            new = torch.zeros_like(z)
            idx_s = src.unsqueeze(1).expand(-1, 2)
            idx_d = dst.unsqueeze(1).expand(-1, 2)
            new.scatter_add_(0, idx_s, w_u_s.unsqueeze(1) * msg)
            new.scatter_add_(0, idx_d, w_u_d.unsqueeze(1) * msg)
            z = (1.0 - self.alpha) * new + self.alpha * z0
        return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", default="0.1,0.5")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--label", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--lambda_edge", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

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
    test_mask = test_mask & (y != -1)

    if args.label == 1:
        cand_mask = test_mask & (y != -1)
        y_pos = y
    else:
        cand_mask = (labels == args.label) | (labels == 0)
        y_pos = (labels == args.label).astype(int)
    pos_cand = np.where((y_pos == 1) & cand_mask)[0]
    neg_cand = np.where((y_pos == 0) & cand_mask)[0]
    print(f"label={args.label}: pos_cand={len(pos_cand)} neg_cand={len(neg_cand)}")

    src = edges[:, 0].astype(int)
    dst = edges[:, 1].astype(int)
    src_t = torch.from_numpy(src).long()
    dst_t = torch.from_numpy(dst).long()
    deg_u = np.bincount(np.concatenate([src, dst]), minlength=n).astype(float)
    deg_u[deg_u == 0] = 1.0
    deg_t = torch.from_numpy(deg_u).float()
    outdeg = np.bincount(src, minlength=n).astype(float)
    indeg = np.bincount(dst, minlength=n).astype(float)
    outdeg[outdeg == 0] = 1.0
    indeg[indeg == 0] = 1.0
    lv = edges[:, 3].astype(float)
    w = lv / lv.max()

    w_u_s = (1.0 / (2.0 * deg_u[src])).astype(np.float32)
    w_u_d = (1.0 / (2.0 * deg_u[dst])).astype(np.float32)
    w_o = (1.0 / outdeg[src]).astype(np.float32)
    w_i = (1.0 / indeg[dst]).astype(np.float32)
    wb = torch.from_numpy(w.astype(np.float32)).to(device)
    src_d, dst_d = src_t.to(device), dst_t.to(device)
    wu_s_d, wu_d_d = torch.from_numpy(w_u_s).to(device), torch.from_numpy(w_u_d).to(device)
    wo_d, wi_d = torch.from_numpy(w_o).to(device), torch.from_numpy(w_i).to(device)
    feat_t = torch.from_numpy(feat).float().to(device)

    val_nodes = np.where(proto["val_mask"] & cand_mask)[0]
    report = {}
    for frac in [float(x) for x in args.fractions.split(",")]:
        rows = {"lp": [], "ppr": [], "xgb": [], "mlp": [], "appnp": [], "dcredit": []}
        gates = []
        for trial in range(args.trials):
            rng = np.random.default_rng(400 + trial)
            seed_p = rng.choice(pos_cand, size=max(1, int(len(pos_cand) * frac)), replace=False)
            seed_n = rng.choice(neg_cand, size=max(1, int(len(neg_cand) * frac)), replace=False)
            seed = np.concatenate([seed_p, seed_n])
            hold = np.setdiff1d(np.arange(n), seed)
            hold = hold[cand_mask[hold]]
            y_hold = y_pos[hold]
            val_v = np.setdiff1d(val_nodes, seed)

            def metrics(sc):
                return {
                    "auc_roc": float(roc_auc_score(y_hold, sc)),
                    "auc_pr": float(average_precision_score(y_hold, sc)),
                    "recall_at_fpr1": float(recall_at_fpr(y_hold, sc)[0]),
                }

            # LP / PPR
            Fm = torch.zeros(n, 2)
            Fm[seed_p, 0] = 1.0
            Fm[seed_n, 1] = 1.0
            onehot = Fm.clone()
            for _ in range(100):
                msg = (Fm[src_t] + Fm[dst_t]) / 2.0
                new = torch.zeros(n, 2)
                idx_s = src_t.unsqueeze(1).expand(-1, 2)
                idx_d = dst_t.unsqueeze(1).expand(-1, 2)
                new.scatter_add_(0, idx_s, msg / deg_t[src_t].unsqueeze(1))
                new.scatter_add_(0, idx_d, msg / deg_t[dst_t].unsqueeze(1))
                new[seed] = onehot[seed]
                if torch.norm(new - Fm).item() < 1e-10:
                    Fm = new
                    break
                Fm = new
            rows["lp"].append(metrics(Fm[hold, 0].numpy()))

            s = torch.zeros(n)
            s[seed_p] = 1.0 / len(seed_p)
            c = s.clone()
            for _ in range(100):
                msg = (c[src_t] + c[dst_t]) / 2.0
                cn = torch.zeros(n)
                cn.scatter_add_(0, src_t, msg / deg_t[src_t])
                cn.scatter_add_(0, dst_t, msg / deg_t[dst_t])
                c = 0.85 * cn + 0.15 * s
            rows["ppr"].append(metrics(c[hold].numpy()))

            ytr = y_pos[seed]
            xm = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                   subsample=0.9, colsample_bytree=0.8,
                                   scale_pos_weight=(ytr == 0).sum() / max((ytr == 1).sum(), 1),
                                   eval_metric="aucpr", tree_method="hist", random_state=0, n_jobs=-1)
            xm.fit(feat[seed], ytr)
            rows["xgb"].append(metrics(xm.predict_proba(feat[hold])[:, 1]))

            cw = torch.tensor([1.0, (ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float, device=device)
            st_t = torch.from_numpy(seed).long().to(device)
            yt_t = torch.from_numpy(ytr).long().to(device)
            sv_t = torch.from_numpy(val_v).long().to(device)
            yv_t = torch.from_numpy(y_pos[val_v]).long().to(device)
            seed_vec = torch.zeros(n, device=device)
            seed_vec[torch.from_numpy(seed_p).long().to(device)] = 1.0 / max(len(seed_p), 1)

            def fit_and_eval(model, forward_fn):
                opt = torch.optim.Adam(model.parameters(), lr=1e-3)
                best_va, best_state, bad = -1.0, None, 0
                for ep in range(args.epochs):
                    model.train()
                    opt.zero_grad()
                    logits = forward_fn(model)
                    loss = F.cross_entropy(logits[st_t], yt_t, weight=cw)
                    if args.lambda_edge > 0:
                        p1 = torch.sigmoid(logits[:, 1])
                        diff = p1[src_d] - p1[dst_d]
                        loss = loss + args.lambda_edge * diff.pow(2).mean()
                    loss.backward()
                    opt.step()
                    model.eval()
                    with torch.no_grad():
                        pv = torch.softmax(forward_fn(model), dim=-1)
                    if len(val_v) and (yv_t == 1).sum() and (yv_t == 0).sum():
                        auc_v = roc_auc_score(y_pos[val_v], pv[sv_t, 1].cpu().numpy())
                    else:
                        auc_v = 0.5
                    if auc_v > best_va:
                        best_va = auc_v
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        bad = 0
                    else:
                        bad += 1
                        if bad >= 50:
                            break
                model.load_state_dict(best_state)
                model.eval()
                with torch.no_grad():
                    p = torch.softmax(forward_fn(model), dim=-1)[:, 1].cpu().numpy()
                return p

            # MLP 特征
            torch.manual_seed(0)
            mlp = nn.Sequential(nn.Linear(feat.shape[1], 48), nn.ReLU(), nn.Dropout(0.3),
                                nn.Linear(48, 2)).to(device)
            rows["mlp"].append(metrics(fit_and_eval(mlp, lambda m: m(feat_t))[hold]))

            # APPNP（预测传播）
            torch.manual_seed(0)
            appnp = APPNP(feat.shape[1]).to(device)
            rows["appnp"].append(metrics(fit_and_eval(appnp, lambda m: m(feat_t, src_d, dst_d, wu_s_d, wu_d_d))[hold]))

            # D-Credit v0.1
            torch.manual_seed(0)
            dc = DCreditV01(feat.shape[1]).to(device)
            p_dc = fit_and_eval(dc, lambda m: m(feat_t, src_d, dst_d, wu_s_d, wu_d_d, wo_d, wi_d, wb, seed_vec)[0])
            rows["dcredit"].append(metrics(p_dc[hold]))
            gates.append(dc.gate_values())

        report[frac] = {}
        for name in rows:
            agg = {}
            for metric in rows[name][0]:
                vals = [r[metric] for r in rows[name]]
                agg[metric] = [round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)]
            report[frac][name] = agg
        report[frac]["learned_gates"] = gates
        print(f"frac={frac}: " + json.dumps(report[frac], ensure_ascii=False))

    with open(os.path.join(OUT, f"dcredit_v01_label{args.label}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("saved ->", os.path.join(OUT, f"dcredit_v01_label{args.label}.json"))


if __name__ == "__main__":
    main()
