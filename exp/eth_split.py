# -*- coding: utf-8 -*-
"""BERT4ETH 账户图：时间划分 + 跨期连通性验证（M2 时序故事的前提）

协议：标签账户按"首次出现区块"排序；train/val/test 用两个区块阈值切分。
验证：测试期 phishing 中，有多少能被训练期 phishing 种子经 k 跳传播到达。
"""
import json
import os
from collections import deque

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "bert4eth", "processed")
OUT = os.path.join(BASE, "results")
os.makedirs(OUT, exist_ok=True)


def main():
    nd = np.load(os.path.join(PROC, "nodes.npz"), allow_pickle=True)
    ed = np.load(os.path.join(PROC, "edges.npz"))
    labels = nd["label"].astype(int)
    edges = ed["edges"]  # src, dst, count, log_val, min_blk, max_blk
    n = len(labels)

    first_blk = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    last_blk = np.full(n, -1, dtype=np.int64)
    for src, dst, c, lv, b0, b1 in edges:
        s, d = int(src), int(dst)
        first_blk[s] = min(first_blk[s], int(b0))
        last_blk[s] = max(last_blk[s], int(b1))
        first_blk[d] = min(first_blk[d], int(b0))
        last_blk[d] = max(last_blk[d], int(b1))

    phish = labels == 1
    first_phish = first_blk[phish]
    b_test = int(np.median(first_phish))
    b_val = int(np.quantile(first_phish, 0.4))
    print("phish first-block median:", b_test, " q40:", b_val)

    train_mask = (first_blk <= b_val) & (labels != -1)
    val_mask = (first_blk > b_val) & (first_blk <= b_test) & (labels != -1)
    test_mask = (first_blk > b_test) & (labels != -1)
    print("labeled train/val/test:", int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum()))
    for name, m in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        print(f"  {name}: phish={int((labels[m]==1).sum())} normal={int((labels[m]==0).sum())} "
              f"tornado={int((labels[m]==2).sum())} ens={int((labels[m]==3).sum())}")

    # 跨期边
    cross = (edges[:, 4] <= b_test) & (edges[:, 5] > b_test)
    print("edges crossing test cutoff:", int(cross.sum()), "/", len(edges))

    # 邻接（无向 + 有向出/入）
    adj = [[] for _ in range(n)]
    adj_out = [[] for _ in range(n)]
    adj_in = [[] for _ in range(n)]
    for src, dst, c, lv, b0, b1 in edges:
        s, d = int(src), int(dst)
        adj[s].append(d)
        adj[d].append(s)
        adj_out[s].append(d)
        adj_in[d].append(s)

    seeds = np.where((labels == 1) & train_mask)[0]
    targets = np.where((labels == 1) & test_mask)[0]
    print("train phish seeds:", len(seeds), "test phish targets:", len(targets))

    def reachable(seed_list, adjlist, k):
        seen = set()
        q = deque((s, 0) for s in seed_list)
        for s in seed_list:
            seen.add(s)
        while q:
            x, d = q.popleft()
            if d >= k:
                continue
            for y in adjlist[x]:
                if y not in seen:
                    seen.add(y)
                    q.append((y, d + 1))
        return seen

    for name, adjlist, k in [("undirected", adj, 3), ("directed_out", adj_out, 3),
                             ("directed_in", adj_in, 3), ("undirected", adj, 5)]:
        seen = reachable(seeds, adjlist, k)
        hit = sum(1 for t in targets if t in seen)
        print(f"{name} k={k}: test phish reachable = {hit}/{len(targets)} ({100*hit/max(len(targets),1):.1f}%)")

    np.savez(
        os.path.join(OUT, "eth_protocol.npz"),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        first_blk=first_blk,
        last_blk=last_blk,
    )
    with open(os.path.join(OUT, "eth_split_stats.json"), "w", encoding="utf-8") as f:
        json.dump({
            "nodes": n,
            "edges": len(edges),
            "b_val": b_val,
            "b_test": b_test,
            "labeled": {"train": int(train_mask.sum()), "val": int(val_mask.sum()),
                        "test": int(test_mask.sum())},
            "edges_crossing_cutoff": int(cross.sum()),
        }, f, ensure_ascii=False, indent=2)
    print("saved ->", os.path.join(OUT, "eth_protocol.npz"))


if __name__ == "__main__":
    main()
