# -*- coding: utf-8 -*-
"""Week2 - GNN 基线超参搜索 + 3-seed 均值（官方划分）

流程：每模型网格搜索（hidden×lr×dropout，300 epoch，val-AUC 早停）→ 最优配置 × 3 seed 正式跑（600 epoch）。
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch

from train_baselines_gnn import train

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results")


def mean_std(rows, key):
    vals = [r[key] for r in rows]
    return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all",
                    help="逗号分隔模型列表或 all")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    models = ["gcn", "sage", "gat", "gin", "gcnii"] if args.models == "all" else args.models.split(",")

    grid = [
        dict(hidden=h, lr=lr, dropout=dp)
        for h in [32, 64, 128]
        for lr in [5e-4, 1e-3]
        for dp in [0.2, 0.5]
    ]

    report = {}
    for name in models:
        print(f"\n===== {name.upper()} 超参搜索 =====")
        best_cfg, best_auc = None, -1.0
        for cfg in grid:
            r = train(name, device, hidden=cfg["hidden"], lr=cfg["lr"], dropout=cfg["dropout"],
                      epochs=300, patience=50, seed=0, verbose=False)
            print(f"  cfg={cfg} val_auc={r['val_auc']:.4f}")
            if r["val_auc"] > best_auc:
                best_auc = r["val_auc"]
                best_cfg = dict(cfg)
        print(f"  best_cfg={best_cfg} val_auc={best_auc:.4f}")

        seeds = []
        for seed in [0, 1, 2]:
            r = train(name, device, hidden=best_cfg["hidden"], lr=best_cfg["lr"],
                      dropout=best_cfg["dropout"], epochs=600, patience=80, seed=seed, verbose=False)
            seeds.append(r)
            print(f"  seed {seed}: auc={r['auc_roc']:.4f} ap={r['auc_pr']:.4f} "
                  f"f1={r['test_macro_f1']:.4f} rec@fpr1={r['recall_at_fpr1']:.4f}")

        report[name] = {
            "best_config": best_cfg,
            "val_auc_best_cfg": best_auc,
            "seeds": seeds,
            "mean_std": {
                "auc_roc": mean_std(seeds, "auc_roc"),
                "auc_pr": mean_std(seeds, "auc_pr"),
                "test_macro_f1": mean_std(seeds, "test_macro_f1"),
                "recall_at_fpr1": mean_std(seeds, "recall_at_fpr1"),
            },
        }

    with open(os.path.join(OUT, "gnn_official_split_tuned.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("saved ->", os.path.join(OUT, "gnn_official_split_tuned.json"))


if __name__ == "__main__":
    main()
