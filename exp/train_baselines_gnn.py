# -*- coding: utf-8 -*-
"""Week1 - GNN 基线重跑：官方时间划分（train t1-29 / val t30-34 / test t35-49）

模型：GCN / GraphSAGE / GAT / GIN / GCNII（复用 chain_fraud/models.py）
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "exp"))
from models import build_model  # noqa: E402

DATA = os.path.join(BASE, "data", "elliptic", "processed", "elliptic.pt")
PROTO = os.path.join(BASE, "results", "elliptic_protocol.pt")
OUT = os.path.join(BASE, "results")


def recall_at_fpr(y_metric, score, fpr_target=0.01):
    """正类=illicit(1)，负类=licit(0)；licit 误报率 ≤ fpr_target 时对 illicit 的召回。"""
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
            return float((y_metric[order[:k]] == 1).sum() / max(pos.sum(), 1))
    return float((y_metric[order] == 1).sum() / max(pos.sum(), 1))


def evaluate(model, x, edge_index, y, mask):
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        prob = torch.softmax(logits, dim=-1)
    # 指标方向：illicit=正类，score=P(illicit)=1-P(licit)
    y_metric = (y[mask] == 0).cpu().numpy().astype(int)
    score = (1.0 - prob[mask, 1]).cpu().numpy()
    return y_metric, score


def train(name, device, hidden=64, num_layers=2, lr=1e-3, wd=5e-4,
          epochs=600, patience=80, seed=0, dropout=0.5, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    d = torch.load(DATA, weights_only=False)
    x = d["x"].to(device)
    y = d["y"].to(device)
    edge_index = d["edge_index"].to(device)
    proto = torch.load(PROTO, weights_only=False)
    train_mask = proto["train_mask"].to(device)
    val_mask = proto["val_mask"].to(device)
    test_mask = proto["test_mask"].to(device)

    model = build_model(name, x.size(1), hidden, 2, num_layers=num_layers, dropout=dropout).to(device)
    n_pos = (y[train_mask] == 1).sum().float()
    n_neg = (y[train_mask] == 0).sum().float()
    class_w = torch.tensor([1.0, (n_neg / n_pos).item()], device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_state, bad_epochs = -1.0, None, 0
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[train_mask], y[train_mask], weight=class_w)
        loss.backward()
        opt.step()
        yv, pv = evaluate(model, x, edge_index, y, val_mask)
        # 验证指标：AUC-ROC（对类不平衡更稳健，避免 F1@0.5 早停过早）
        val = roc_auc_score(yv, pv)
        if val > best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
        if verbose and (epoch + 1) % 50 == 0:
            print(f"  {name} epoch {epoch+1:3d} loss={loss.item():.4f} val_macro_f1={val:.4f}")

    model.load_state_dict(best_state)
    yte, pte = evaluate(model, x, edge_index, y, test_mask)
    # 验证集上选最优阈值
    yv, pv = evaluate(model, x, edge_index, y, val_mask)
    cands = np.unique(np.concatenate([[0.0], pv, [1.0]]))
    best_t, best_f1 = 0.5, -1.0
    for c in cands:
        f1 = f1_score(yv, (pv >= c).astype(int), average="macro")
        if f1 > best_f1:
            best_t, best_f1 = c, f1
    pred = (pte >= best_t).astype(int)
    res = {
        "model": name,
        "test_macro_f1": float(f1_score(yte, pred, average="macro")),
        "auc_roc": float(roc_auc_score(yte, pte)),
        "auc_pr": float(average_precision_score(yte, pte)),
        "recall_at_fpr1": float(recall_at_fpr(yte, pte, 0.01)),
        "val_auc": float(best_val),
        "best_val_f1": float(best_f1),
        "seconds": float(time.time() - t0),
        "params": int(sum(p.numel() for p in model.parameters())),
        "epochs_run": int(epoch + 1),
    }
    print(f"[{name}] {json.dumps(res, ensure_ascii=False)}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all",
                    choices=["gcn", "sage", "gat", "gin", "gcnii", "all"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--patience", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    models = ["gcn", "sage", "gat", "gin", "gcnii"] if args.model == "all" else [args.model]
    results = []
    for name in models:
        print(f"\n===== {name.upper()} =====")
        results.append(train(name, device, hidden=args.hidden, num_layers=args.layers,
                             epochs=args.epochs, patience=args.patience, seed=args.seed))
    with open(os.path.join(OUT, "gnn_official_split.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved ->", os.path.join(OUT, "gnn_official_split.json"))


if __name__ == "__main__":
    main()
