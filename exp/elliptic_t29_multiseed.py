# -*- coding: utf-8 -*-
"""Elliptic t=29 snapshot: multi-seed evaluation with raw per-trial values.

Reproduces the exact protocol of draft_support.ellip_t29_fusion (same snapshot
construction, same LP/XGBoost settings, same seed-internal validation split for
the gate), but runs a configurable number of trials and stores the raw per-trial
metrics so that 95% confidence intervals can be computed.
"""
import argparse
import json
import os

import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from metrics import recall_at_fpr
from draft_support import lp_undirected, zscore

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "elliptic", "processed", "elliptic.pt")
OUT = os.path.join(BASE, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--fractions", default="0.1,0.5")
    ap.add_argument("--time-step", type=int, default=29)
    ap.add_argument("--seed-base", type=int, default=700)
    args = ap.parse_args()

    d = torch.load(DATA, weights_only=False)
    X, y, t, ei = (d["x"].numpy(), d["y"].numpy(),
                   d["time_step"].numpy(), d["edge_index"].numpy())
    k = args.time_step
    nodes = np.where(t == k)[0]
    node_set = set(nodes.tolist())
    u, v = ei
    keep = np.array([u[i] in node_set and v[i] in node_set for i in range(len(u))],
                    dtype=bool)
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
    print(f"t={k}: nodes={n} illicit={len(ill)} licit={len(lic)} edges={len(us)}",
          flush=True)

    report = {}
    for frac in [float(x) for x in args.fractions.split(",")]:
        rows = {"lp": [], "xgb": [], "gate_fusion": []}
        ws = []
        for trial in range(args.trials):
            rng = np.random.default_rng(args.seed_base + trial)
            sp = rng.choice(ill, size=max(1, int(len(ill) * frac)), replace=False)
            sn = rng.choice(lic, size=max(1, int(len(lic) * frac)), replace=False)
            seed = np.concatenate([sp, sn])
            hold = np.setdiff1d(np.arange(n), seed)
            y_hold = (y_loc[hold] == 0).astype(int)
            ytr = (y_loc[seed] == 0).astype(int)

            c0 = np.zeros(n)
            c0[sp] = 1.0 / len(sp)
            lp = lp_undirected(c0, us, vs, deg_u, n)
            xm = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.9,
                colsample_bytree=0.8,
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

            # Gate: seed-internal 20% validation split, w selected by AP.
            perm = rng.permutation(len(seed))
            val_s = seed[perm[: max(1, len(seed) // 5)]]
            yv = (y_loc[val_s] == 0).astype(int)
            lpv, xpv = zscore(lp)[val_s], zscore(xp)[val_s]
            best_w_auc, best_auc = 0.0, -1.0
            best_w_ap, best_ap = 0.0, -1.0
            for w in np.arange(0.0, 1.001, 0.05):
                sc_v = w * lpv + (1 - w) * xpv
                auc = roc_auc_score(yv, sc_v)
                if auc > best_auc:
                    best_auc, best_w_auc = auc, w
                ap = average_precision_score(yv, sc_v)
                if ap > best_ap:
                    best_ap, best_w_ap = ap, w
            sc = best_w_ap * zscore(lp) + (1 - best_w_ap) * zscore(xp)
            rows["gate_fusion"].append(met(sc[hold]))
            ws.append([round(float(best_w_auc), 2), round(float(best_w_ap), 2)])
            print(f"  frac={frac} trial={trial}: w_auc={best_w_auc:.2f} "
                  f"w_ap={best_w_ap:.2f}", flush=True)

        report[str(frac)] = {name: rows[name] for name in rows}
        report[str(frac)]["learned_w"] = ws

    out_path = os.path.join(OUT, "elliptic_t29_multiseed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": {"trials": args.trials, "seed_base": args.seed_base,
                            "time_step": k,
                            "fractions": [float(x) for x in args.fractions.split(",")]},
                   "results": report}, f, ensure_ascii=False, indent=2)
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
