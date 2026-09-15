# -*- coding: utf-8 -*-
"""D-Credit v0 同窗口验证（Tornado / Phishing）

设定：候选集内随机抽 frac 正类（+normal）种子 → 学习端到端信用传播（α/γ/β 门控）
     → 在 held-out 上评测。对照：LP / PPR / XGBoost（特征）/ MLP（特征）。
关键验证：Tornado 体制学习门控能否 ≥ LP 的 Recall@FPR=1%≈0.70；
          Phishing 体制能否学到 γ→0（≈MLP 特征性能，不掉点）。
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

from metrics import recall_at_fpr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "bert4eth", "processed")
PROTO = os.path.join(BASE, "results", "eth_protocol.npz")
OUT = os.path.join(BASE, "results")


class Dcredit(nn.Module):
    def __init__(self, in_dim, hidden=48, K=5):
        super().__init__()
        self.logit_alpha = nn.Parameter(torch.tensor(0.0))
        self.logit_gamma_out = nn.Parameter(torch.tensor(0.0))
        self.logit_gamma_in = nn.Parameter(torch.tensor(0.0))
        self.log_beta = nn.Parameter(torch.tensor(0.0))
        self.K = K
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + 1, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, 2),
        )

    def propagate(self, src, dst, w_out, w_in, wb, seed):
        alpha = torch.sigmoid(self.logit_alpha)
        g = torch.softmax(torch.stack([self.logit_gamma_out, self.logit_gamma_in]), dim=0)
        c = seed.clone()
        for _ in range(self.K):
            c_out = torch.zeros_like(c)
            c_out.scatter_add_(0, dst, w_out * wb * c[src])
            c_in = torch.zeros_like(c)
            c_in.scatter_add_(0, src, w_in * wb * c[dst])
            c = (1.0 - alpha) * (g[0] * c_out + g[1] * c_in) + alpha * seed
        return c

    def forward(self, feat, src, dst, w_out, w_in, wb, seed):
        c = self.propagate(src, dst, w_out, w_in, wb, seed)
        cred = torch.log1p(c)
        cred = cred / (cred.max() + 1e-8)
        return self.mlp(torch.cat([feat, cred.unsqueeze(1)], dim=-1)), c

    def gate_values(self):
        alpha = torch.sigmoid(self.logit_alpha).item()
        g = torch.softmax(torch.stack([self.logit_gamma_out, self.logit_gamma_in]), dim=0).tolist()
        beta = torch.exp(self.log_beta).item()
        return {"alpha": round(alpha, 3), "gamma_out": round(g[0], 3),
                "gamma_in": round(g[1], 3), "beta": round(beta, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", default="0.1,0.2,0.5")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--label", type=int, default=2, help="正类：1=phishing, 2=tornado")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--aux_lambda", type=float, default=1.0,
                    help="信用-分类一致性损失权重（0=关闭）")
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
    lv = edges[:, 3].astype(float)
    w = lv / lv.max()
    outdeg = np.bincount(src, minlength=n).astype(float)
    indeg = np.bincount(dst, minlength=n).astype(float)
    outdeg[outdeg == 0] = 1.0
    indeg[indeg == 0] = 1.0
    w_out = torch.from_numpy((1.0 / outdeg[src]).astype(np.float32)).to(device)
    w_in = torch.from_numpy((1.0 / indeg[dst]).astype(np.float32)).to(device)
    wb = torch.from_numpy(w.astype(np.float32)).to(device)
    src_d, dst_d = src_t.to(device), dst_t.to(device)
    feat_t = torch.from_numpy(feat).float().to(device)

    report = {}
    for frac in [float(x) for x in args.fractions.split(",")]:
        rows = {"lp": [], "ppr": [], "xgb": [], "mlp": [], "dcredit": []}
        gates = []
        for trial in range(args.trials):
            rng = np.random.default_rng(200 + trial)
            seed_p = rng.choice(pos_cand, size=max(1, int(len(pos_cand) * frac)), replace=False)
            seed_n = rng.choice(neg_cand, size=max(1, int(len(neg_cand) * frac)), replace=False)
            seed = np.concatenate([seed_p, seed_n])
            hold = np.setdiff1d(np.arange(n), seed)
            hold = hold[cand_mask[hold]]
            y_hold = y_pos[hold]

            def metrics(sc):
                return {
                    "auc_roc": float(roc_auc_score(y_hold, sc)),
                    "auc_pr": float(average_precision_score(y_hold, sc)),
                    "recall_at_fpr1": float(recall_at_fpr(y_hold, sc)[0]),
                }

            # LP
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

            # PPR
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

            # XGBoost
            ytr = y_pos[seed]
            xm = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                   subsample=0.9, colsample_bytree=0.8,
                                   scale_pos_weight=(ytr == 0).sum() / max((ytr == 1).sum(), 1),
                                   eval_metric="aucpr", tree_method="hist", random_state=0, n_jobs=-1)
            xm.fit(feat[seed], ytr)
            rows["xgb"].append(metrics(xm.predict_proba(feat[hold])[:, 1]))

            # MLP 特征
            torch.manual_seed(0)
            mlp = nn.Sequential(nn.Linear(feat.shape[1], 48), nn.ReLU(), nn.Dropout(0.3),
                                nn.Linear(48, 2)).to(device)
            opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
            cw = torch.tensor([1.0, (ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float, device=device)
            seed_t = torch.from_numpy(seed).long().to(device)
            yt = torch.from_numpy(ytr).long().to(device)
            for ep in range(args.epochs):
                mlp.train()
                opt.zero_grad()
                loss = F.cross_entropy(mlp(feat_t[seed_t]), yt, weight=cw)
                loss.backward()
                opt.step()
            mlp.eval()
            with torch.no_grad():
                p = torch.softmax(mlp(feat_t), dim=-1)[:, 1].cpu().numpy()
            rows["mlp"].append(metrics(p[hold]))

            # D-Credit：用官方验证期节点早停（与种子分离），训练用全部种子
            val_nodes = np.where(proto["val_mask"] & cand_mask)[0]
            val_nodes = val_nodes[~np.isin(val_nodes, seed)]
            if len(val_nodes) == 0 or (y_pos[val_nodes] == 1).sum() == 0 or (y_pos[val_nodes] == 0).sum() == 0:
                val_nodes = np.setdiff1d(np.arange(n), seed)
                val_nodes = val_nodes[cand_mask[val_nodes]]
            y_tr = y_pos[seed]
            y_va = y_pos[val_nodes]
            torch.manual_seed(0)
            model = Dcredit(feat.shape[1]).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            cw2 = torch.tensor([1.0, (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)], dtype=torch.float, device=device)
            seed_vec = torch.zeros(n, device=device)
            seed_vec[torch.from_numpy(seed_p).long().to(device)] = 1.0 / max(len(seed_p), 1)
            st_t = torch.from_numpy(seed).long().to(device)
            sv_t = torch.from_numpy(val_nodes).long().to(device)
            yt_t = torch.from_numpy(y_tr).long().to(device)
            yv_t = torch.from_numpy(y_va).long().to(device)
            best_va, best_state, bad = -1.0, None, 0
            for ep in range(args.epochs):
                model.train()
                opt.zero_grad()
                logits, cred = model(feat_t, src_d, dst_d, w_out, w_in, wb, seed_vec)
                loss = F.cross_entropy(logits[st_t], yt_t, weight=cw2)
                if args.aux_lambda > 0:
                    cred_n = torch.log1p(cred)
                    cred_n = cred_n / (cred_n.max() + 1e-8)
                    loss = loss + args.aux_lambda * F.mse_loss(
                        torch.sigmoid(logits[:, 1]), cred_n)
                loss.backward()
                opt.step()
                model.eval()
                with torch.no_grad():
                    pv = torch.softmax(model(feat_t, src_d, dst_d, w_out, w_in, wb, seed_vec)[0], dim=-1)
                auc_v = roc_auc_score(y_va, pv[sv_t, 1].cpu().numpy())
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
                p = torch.softmax(model(feat_t, src_d, dst_d, w_out, w_in, wb, seed_vec)[0], dim=-1)[:, 1].cpu().numpy()
            rows["dcredit"].append(metrics(p[hold]))
            gates.append(model.gate_values())

        report[frac] = {}
        for name in rows:
            agg = {}
            for metric in rows[name][0]:
                vals = [r[metric] for r in rows[name]]
                agg[metric] = [round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)]
            report[frac][name] = agg
        report[frac]["learned_gates"] = gates
        print(f"frac={frac}: " + json.dumps(report[frac], ensure_ascii=False))

    with open(os.path.join(OUT, f"dcredit_same_window_label{args.label}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("saved ->", os.path.join(OUT, f"dcredit_same_window_label{args.label}.json"))


if __name__ == "__main__":
    main()
