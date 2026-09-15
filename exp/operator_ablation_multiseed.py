# -*- coding: utf-8 -*-
"""Operator-choice ablation on the Tornado graph, multi-seed.

Same protocol as draft_support.direction_ablation (10% of labels as seeds,
undirected / directed-out / directed-in label propagation), but repeated over
independent seed draws and with raw per-trial values stored.
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from metrics import recall_at_fpr
from draft_support import lp_undirected, lp_directed

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "bert4eth", "processed")
OUT = os.path.join(BASE, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--fraction", type=float, default=0.1)
    ap.add_argument("--seed-base", type=int, default=800)
    args = ap.parse_args()

    nd = np.load(os.path.join(PROC, "nodes.npz"), allow_pickle=True)
    ed = np.load(os.path.join(PROC, "edges.npz"))
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

    rows = {"undirected": [], "directed_out": [], "directed_in": []}
    for trial in range(args.trials):
        rng = np.random.default_rng(args.seed_base + trial)
        sp = rng.choice(pos, size=max(1, int(len(pos) * args.fraction)),
                        replace=False)
        sn = rng.choice(neg, size=max(1, int(len(neg) * args.fraction)),
                        replace=False)
        seed = np.concatenate([sp, sn])
        hold = np.setdiff1d(np.arange(n), seed)
        hold = hold[cand[hold]]
        yh = ypos[hold]
        c0 = np.zeros(n)
        c0[sp] = 1.0 / len(sp)
        for name, sc in [
            ("undirected", lp_undirected(c0, src, dst, deg_u, n)),
            ("directed_out", lp_directed(c0, src, dst, outdeg, n, "out")),
            ("directed_in", lp_directed(c0, src, dst, indeg, n, "in")),
        ]:
            rows[name].append({
                "auc_roc": float(roc_auc_score(yh, sc[hold])),
                "recall_at_fpr1": float(recall_at_fpr(yh, sc[hold])[0]),
            })
        print(f"  trial={trial} done", flush=True)

    out_path = os.path.join(OUT, "operator_ablation_multiseed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": {"trials": args.trials, "seed_base": args.seed_base,
                            "fraction": args.fraction},
                   "results": rows}, f, ensure_ascii=False, indent=2)
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
