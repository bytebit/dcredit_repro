# -*- coding: utf-8 -*-
"""GraphSAGE baseline under the same-seed, low-FPR protocol.

Matches exp/multiseed_eval.py exactly: same seed draw per trial, same candidate
and validation sets, the same strict isolation (evaluation excludes seeds and
validation nodes), and the same metrics. Training uses only the seed labels;
early stopping and the deployment threshold use the validation period only.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from metrics import recall_at_fpr
from models import GraphSAGE

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "bert4eth", "processed")
PROTO = os.path.join(BASE, "results", "eth_protocol.npz")
OUT = os.path.join(BASE, "results")


def val_threshold_recall(val_scores, val_labels, hold_scores, hold_labels):
    """Deployment threshold: 1% FPR on validation, then recall on hold."""
    neg = val_scores[val_labels == 0]
    pos_h = hold_scores[hold_labels == 1]
    if neg.size == 0 or pos_h.size == 0:
        return float("nan")
    thr = float(np.quantile(neg, 0.99))
    return float((pos_h > thr).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", default="0.1,0.5")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--label", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=700)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    nd = np.load(os.path.join(PROC, "nodes.npz"), allow_pickle=True)
    ed = np.load(os.path.join(PROC, "edges.npz"))
    proto = np.load(PROTO)
    labels = nd["label"].astype(int)
    feat = nd["feat"].astype(np.float32)
    edges = ed["edges"]
    n = len(labels)
    y = np.full(n, -1, dtype=int)
    y[labels == 1] = 1
    y[labels == 0] = 0
    test_mask = proto["test_mask"] & (y != -1)

    if args.label == 1:
        cand_mask = test_mask & (y != -1)
        y_pos = y
        val_nodes = np.where(proto["val_mask"]
                             & ((labels == 1) | (labels == 0)))[0]
    else:
        cand_mask = (labels == args.label) | (labels == 0)
        y_pos = (labels == args.label).astype(int)
        val_nodes = np.where(proto["val_mask"] & cand_mask)[0]
    pos_cand = np.where((y_pos == 1) & cand_mask)[0]
    neg_cand = np.where((y_pos == 0) & cand_mask)[0]
    print(f"label={args.label}: pos={len(pos_cand)} neg={len(neg_cand)}",
          flush=True)

    src = edges[:, 0].astype(int)
    dst = edges[:, 1].astype(int)
    edge_index = torch.from_numpy(
        np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1)
    ).long().to(device)
    mu, sd = feat.mean(0), feat.std(0)
    feat = (feat - mu) / (sd + 1e-9)
    x = torch.from_numpy(feat).float().to(device)
    y_t = torch.from_numpy(y).long().to(device)

    report = {}
    for frac in [float(x) for x in args.fractions.split(",")]:
        rows = []
        val_thr = []
        secs = []
        audit = {"seed_nodes_used_for_scoring": 0,
                 "validation_nodes_used_for_scoring": 0,
                 "evaluation_nodes_exclude_validation": True}
        for trial in range(args.trials):
            t0 = time.time()
            rng = np.random.default_rng(args.seed_base + trial)
            seed_p = rng.choice(pos_cand, size=max(1, int(len(pos_cand) * frac)),
                                replace=False)
            seed_n = rng.choice(neg_cand, size=max(1, int(len(neg_cand) * frac)),
                                replace=False)
            seed = np.concatenate([seed_p, seed_n])
            val_v = np.setdiff1d(val_nodes, seed)
            eval_mask = cand_mask.copy()
            eval_mask[seed] = False
            eval_mask[val_v] = False
            hold = np.where(eval_mask)[0]
            assert not np.intersect1d(seed, hold).size
            assert not np.intersect1d(val_v, hold).size
            y_hold = y_pos[hold]

            seed_t = torch.from_numpy(seed).long().to(device)
            val_t = torch.from_numpy(val_v).long().to(device)
            ytr = y_pos[seed]
            cw = torch.tensor([1.0, (ytr == 0).sum() / max((ytr == 1).sum(), 1)],
                              dtype=torch.float, device=device)

            torch.manual_seed(0)
            model = GraphSAGE(feat.shape[1], 64, 2, num_layers=2,
                              dropout=0.3).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
            best_auc, best_state, bad = -1.0, None, 0
            for _ in range(args.epochs):
                model.train()
                opt.zero_grad()
                logits = model(x, edge_index)
                loss = F.cross_entropy(logits[seed_t],
                                       torch.from_numpy(ytr).long().to(device),
                                       weight=cw)
                loss.backward()
                opt.step()
                model.eval()
                with torch.no_grad():
                    p_val = torch.softmax(model(x, edge_index), -1)[val_t, 1]
                y_val = y_pos[val_v]
                if y_val.min() != y_val.max():
                    auc = roc_auc_score(y_val, p_val.cpu().numpy())
                else:
                    auc = 0.5
                if auc > best_auc:
                    best_auc = auc
                    best_state = {k: v.detach().clone()
                                  for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= 30:
                        break
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                prob = torch.softmax(model(x, edge_index), -1)[:, 1].cpu().numpy()
            sc = prob[hold]
            rows.append({
                "auc_roc": float(roc_auc_score(y_hold, sc)),
                "auc_pr": float(average_precision_score(y_hold, sc)),
                "recall_at_fpr1": float(recall_at_fpr(y_hold, sc)[0]),
            })
            val_thr.append(val_threshold_recall(prob[val_v], y_pos[val_v],
                                                sc, y_hold))
            secs.append(round(time.time() - t0, 1))
            print(f"  frac={frac} trial={trial}: "
                  f"auc={rows[-1]['auc_roc']:.3f} "
                  f"rec={rows[-1]['recall_at_fpr1']:.3f} ({secs[-1]}s)",
                  flush=True)
        report[str(frac)] = {"graphsage": rows,
                             "recall_at_val_fpr1": val_thr,
                             "seconds": secs,
                             "evaluation_protocol": audit}

    out_path = args.out or os.path.join(OUT, f"sage_label{args.label}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": {"label": args.label, "trials": args.trials,
                            "seed_base": args.seed_base,
                            "fractions": [float(x) for x in
                                          args.fractions.split(",")],
                            "epochs": args.epochs,
                            "protocol": "transductive all-time graph"},
                   "results": report}, f, ensure_ascii=False, indent=2)
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
