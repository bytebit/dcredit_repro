# -*- coding: utf-8 -*-
"""Week1 - 评测协议修复：官方时间划分 + 数据集统计

口径（文献常用，已核实）：
  - 原论文：train = 时间步 1-34，test = 时间步 35-49
  - 本脚本：train t∈[1,29]，val t∈[30,34]，test t∈[35,49]（30-34 从 train 中分出作验证）
  - 特征：166 维 = 94 local（含 time）+ 72 one-hop 聚合；local-only = 前 94 列

输出：
  results/elliptic_protocol.pt   （train/val/test mask + 特征掩码）
  results/elliptic_stats.json
  results/elliptic_stats.md
"""
import json
import os

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "elliptic", "processed", "elliptic.pt")
OUT = os.path.join(BASE, "results")
os.makedirs(OUT, exist_ok=True)


def main():
    d = torch.load(DATA, weights_only=False)
    x = d["x"].numpy()
    y = d["y"].numpy()
    t = d["time_step"].numpy()
    ei = d["edge_index"].numpy()

    labeled = y != 2
    train_mask = (t <= 29) & labeled
    val_mask = ((t >= 30) & (t <= 34)) & labeled
    test_mask = (t >= 35) & labeled

    def counts(m):
        return {
            "labeled": int(m.sum()),
            "illicit": int((y[m] == 0).sum()),
            "licit": int((y[m] == 1).sum()),
            "illicit_pct": round(100.0 * (y[m] == 0).mean(), 2),
        }

    split_stats = {
        "train_t1-29": counts(train_mask),
        "val_t30-34": counts(val_mask),
        "test_t35-49": counts(test_mask),
    }

    # 特征掩码（X 列序：col0=time，col1..93=local，col94..165=aggregated）
    assert x.shape[1] == 166, x.shape
    feat_masks = {
        "full": list(range(0, 166)),
        "local_only": list(range(0, 94)),
        "aggregated_only": list(range(94, 166)),
    }

    # 异质性 / 结构统计
    u, v = ei
    lab_u, lab_v = y[u], y[v]
    both = (lab_u != 2) & (lab_v != 2)
    any_ill = (lab_u == 0) | (lab_v == 0)
    mix_any_ill = (lab_u != lab_v)[any_ill]

    mixed_edges = set()
    for a, b in zip(u[both].tolist(), v[both].tolist()):
        if y[a] != y[b]:
            mixed_edges.add(a)
            mixed_edges.add(b)
    ill_nodes = set(np.where(y == 0)[0].tolist())
    ill_with_mixed = len(mixed_edges & ill_nodes)

    stats = {
        "nodes": int(x.shape[0]),
        "edges": int(ei.shape[1]),
        "features": int(x.shape[1]),
        "time_steps": [int(t.min()), int(t.max())],
        "labeled_edges": int(both.sum()),
        "labeled_edge_ratio_pct": round(100.0 * both.sum() / ei.shape[1], 2),
        "edges_touching_illicit": int(any_ill.sum()),
        "mixed_ratio_among_edges_touching_illicit_pct": round(100.0 * mix_any_ill.mean(), 2),
        "illicit_nodes": int(len(ill_nodes)),
        "illicit_nodes_with_opposite_labeled_neighbor": ill_with_mixed,
        "illicit_nodes_with_opposite_labeled_neighbor_pct": round(100.0 * ill_with_mixed / len(ill_nodes), 2),
        "split": split_stats,
        "feature_masks": feat_masks,
    }

    torch.save(
        {
            "train_mask": torch.from_numpy(train_mask),
            "val_mask": torch.from_numpy(val_mask),
            "test_mask": torch.from_numpy(test_mask),
            "feat_masks": feat_masks,
        },
        os.path.join(OUT, "elliptic_protocol.pt"),
    )
    with open(os.path.join(OUT, "elliptic_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    md = [
        "# Elliptic 协议与统计（Week1 核实）",
        "",
        f"- 节点 {stats['nodes']:,} / 有向边 {stats['edges']:,} / 特征 {stats['features']}（94 local 含 time + 72 聚合）",
        f"- 时间步 {stats['time_steps'][0]}-{stats['time_steps'][1]}",
        f"- 标注边占比 {stats['labeled_edge_ratio_pct']}%；与 illicit 相连边中混合标签占比 {stats['mixed_ratio_among_edges_touching_illicit_pct']}%",
        f"- illicit 节点 {stats['illicit_nodes']:,}，其中 {stats['illicit_nodes_with_opposite_labeled_neighbor_pct']}% 有直接异质标注邻居",
        "",
        "## 划分（官方口径）",
        "",
        "| 划分 | 时间步 | 标注 | illicit | licit | illicit% |",
        "|---|---|---|---|---|---|",
    ]
    for k, v in split_stats.items():
        md.append(f"| {k} | - | {v['labeled']:,} | {v['illicit']:,} | {v['licit']:,} | {v['illicit_pct']} |")
    md.append("")
    with open(os.path.join(OUT, "elliptic_stats.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
