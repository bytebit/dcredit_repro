# -*- coding: utf-8 -*-
"""Multi-seed evaluation with raw per-trial values (ETH same-window regimes).

Unified protocol: within each trial a single seed draw is shared by every method
(LP, PPR, XGBoost, MLP, APPNP, and the D-Credit gate fusion), so all rows of a
table are directly comparable. Raw per-trial metrics are stored, which allows
means with 95% confidence intervals instead of means with standard deviations.

Methods
-------
LP    : undirected symmetric normalization with seed clamping (structure only).
PPR   : bidirectional personalized PageRank, alpha=0.15, seed on positives.
XGB   : XGBoost on node features, trained on the seed set.
MLP   : feature MLP (same architecture as the reported baseline).
APPNP : prediction diffusion over the same operator, trained end-to-end.
fusion: D-Credit gate fusion, score = w*z(LP) + (1-w)*z(XGB), with w selected on
        the validation period by AUC (primary) and by Recall@FPR=1% (recorded for
        the gate-selection analysis).
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
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--label", type=int, default=2,
                    help="positive class: 1=phishing, 2=tornado")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--seed-base", type=int, default=700)
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
        # The evaluation set is the test window, so the gate is selected on the
        # official validation period (disjoint from the evaluation nodes).
        val_nodes = np.where(proto["val_mask"]
                             & ((labels == 1) | (labels == 0)))[0]
    else:
        cand_mask = (labels == args.label) | (labels == 0)
        y_pos = (labels == args.label).astype(int)
        val_nodes = np.where(proto["val_mask"] & cand_mask)[0]
    pos_cand = np.where((y_pos == 1) & cand_mask)[0]
    neg_cand = np.where((y_pos == 0) & cand_mask)[0]
    print(f"label={args.label}: pos_cand={len(pos_cand)} neg_cand={len(neg_cand)}",
          flush=True)

    src = edges[:, 0].astype(int)
    dst = edges[:, 1].astype(int)
    src_t = torch.from_numpy(src).long()
    dst_t = torch.from_numpy(dst).long()
    deg_u = np.bincount(np.concatenate([src, dst]), minlength=n).astype(float)
    deg_u[deg_u == 0] = 1.0
    deg_t = torch.from_numpy(deg_u).float()
    w_u_s_np = (1.0 / (2.0 * deg_u[src])).astype(np.float32)
    w_u_d_np = (1.0 / (2.0 * deg_u[dst])).astype(np.float32)
    src_d, dst_d = src_t.to(device), dst_t.to(device)
    wu_s_d = torch.from_numpy(w_u_s_np).to(device)
    wu_d_d = torch.from_numpy(w_u_d_np).to(device)
    feat_t = torch.from_numpy(feat).float().to(device)

    grid = np.arange(0.0, 1.001, 0.05)
    report = {}
    for frac in [float(x) for x in args.fractions.split(",")]:
        rows = {k: [] for k in ("lp", "ppr", "xgb", "mlp", "appnp", "gate_fusion")}
        w_auc, w_rec, secs = [], [], []
        for trial in range(args.trials):
            t0 = time.time()
            rng = np.random.default_rng(args.seed_base + trial)
            seed_p = rng.choice(pos_cand, size=max(1, int(len(pos_cand) * frac)),
                                replace=False)
            seed_n = rng.choice(neg_cand, size=max(1, int(len(neg_cand) * frac)),
                                replace=False)
            seed = np.concatenate([seed_p, seed_n])
            hold = np.setdiff1d(np.arange(n), seed)
            hold = hold[cand_mask[hold]]
            y_hold = y_pos[hold]
            val_v = np.setdiff1d(val_nodes, seed)
            ytr = y_pos[seed]

            def metrics(sc):
                return {
                    "auc_roc": float(roc_auc_score(y_hold, sc)),
                    "auc_pr": float(average_precision_score(y_hold, sc)),
                    "recall_at_fpr1": float(recall_at_fpr(y_hold, sc)[0]),
                }

            # ---- LP (undirected symmetric, seed clamping) ----
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
            lp_all = Fm[:, 0].numpy()
            rows["lp"].append(metrics(lp_all[hold]))

            # ---- PPR (bidirectional, alpha=0.15) ----
            s = torch.zeros(n)
            s[seed_p] = 1.0 / max(len(seed_p), 1)
            c = s.clone()
            for _ in range(100):
                msg = (c[src_t] + c[dst_t]) / 2.0
                cn = torch.zeros(n)
                cn.scatter_add_(0, src_t, msg / deg_t[src_t])
                cn.scatter_add_(0, dst_t, msg / deg_t[dst_t])
                c = 0.85 * cn + 0.15 * s
            rows["ppr"].append(metrics(c[hold].numpy()))

            # ---- XGBoost on features ----
            xm = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.9,
                colsample_bytree=0.8,
                scale_pos_weight=(ytr == 0).sum() / max((ytr == 1).sum(), 1),
                eval_metric="aucpr", tree_method="hist", random_state=0, n_jobs=-1)
            xm.fit(feat[seed], ytr)
            xp_all = xm.predict_proba(feat)[:, 1]
            rows["xgb"].append(metrics(xp_all[hold]))

            # ---- MLP / APPNP (same training loop as the reported baselines) ----
            cw = torch.tensor([1.0, (ytr == 0).sum() / max((ytr == 1).sum(), 1)],
                              dtype=torch.float, device=device)
            st_t = torch.from_numpy(seed).long().to(device)
            yt_t = torch.from_numpy(ytr).long().to(device)
            sv_t = torch.from_numpy(val_v).long().to(device)
            yv_t = torch.from_numpy(y_pos[val_v]).long().to(device)

            def fit_and_eval(model, forward_fn):
                opt = torch.optim.Adam(model.parameters(), lr=1e-3)
                best_va, best_state, bad = -1.0, None, 0
                for _ in range(args.epochs):
                    model.train()
                    opt.zero_grad()
                    logits = forward_fn(model)
                    loss = F.cross_entropy(logits[st_t], yt_t, weight=cw)
                    loss.backward()
                    opt.step()
                    model.eval()
                    with torch.no_grad():
                        pv = torch.softmax(forward_fn(model), dim=-1)
                    if len(val_v) and (yv_t == 1).sum() and (yv_t == 0).sum():
                        auc_v = roc_auc_score(y_pos[val_v],
                                              pv[sv_t, 1].cpu().numpy())
                    else:
                        auc_v = 0.5
                    if auc_v > best_va:
                        best_va = auc_v
                        best_state = {k: v.detach().cpu().clone()
                                      for k, v in model.state_dict().items()}
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

            torch.manual_seed(0)
            mlp = nn.Sequential(nn.Linear(feat.shape[1], 48), nn.ReLU(),
                                nn.Dropout(0.3), nn.Linear(48, 2)).to(device)
            rows["mlp"].append(metrics(fit_and_eval(mlp, lambda m: m(feat_t))[hold]))

            torch.manual_seed(0)
            appnp = APPNP(feat.shape[1]).to(device)
            rows["appnp"].append(metrics(fit_and_eval(
                appnp, lambda m: m(feat_t, src_d, dst_d, wu_s_d, wu_d_d))[hold]))

            # ---- D-Credit gate fusion (w selected on the validation period) ----
            zv = lambda v: (v - v.mean()) / (v.std() + 1e-9)
            zlp, zxp = zv(lp_all), zv(xp_all)
            yv = y_pos[val_v]
            best_w, best_auc = 0.0, -1.0
            for w in grid:
                auc = roc_auc_score(yv, w * zlp[val_v] + (1 - w) * zxp[val_v])
                if auc > best_auc:
                    best_auc, best_w = auc, w
            best_wr, best_rec = 0.0, -1.0
            for w in grid:
                rec = recall_at_fpr(yv, w * zlp[val_v] + (1 - w) * zxp[val_v])[0]
                if rec > best_rec:
                    best_rec, best_wr = rec, w
            sc = best_w * zlp + (1 - best_w) * zxp
            rows["gate_fusion"].append(metrics(sc[hold]))
            w_auc.append(round(float(best_w), 2))
            w_rec.append(round(float(best_wr), 2))
            secs.append(round(time.time() - t0, 1))
            print(f"  frac={frac} trial={trial}: w_auc={best_w:.2f} "
                  f"w_recall={best_wr:.2f} ({secs[-1]}s)", flush=True)

        report[str(frac)] = {k: v for k, v in rows.items()}
        report[str(frac)]["gate_w_auc"] = w_auc
        report[str(frac)]["gate_w_recall"] = w_rec
        report[str(frac)]["seconds"] = secs

    out_path = os.path.join(OUT, f"multiseed_label{args.label}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": {"label": args.label, "trials": args.trials,
                            "seed_base": args.seed_base,
                            "fractions": [float(x) for x in args.fractions.split(",")],
                            "epochs": args.epochs},
                   "results": report}, f, ensure_ascii=False, indent=2)
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
