# -*- coding: utf-8 -*-
"""BERT4ETH 预处理：账户 × 账户交易图（M2 的金额/时序主战场）

任务：钓鱼账户检测（phishing vs normal 二分类，半监督 + 传播评测）
图：节点 = 标签账户 + 一阶交易邻居；边 = 账户间交易聚合（笔数/总金额 wei/区块范围）

两遍扫描：
  Pass 1：确定标签账户的一阶邻居（邻居与标签账户交易 ≥ min_txs 才保留）
  Pass 2：抽取两端都在节点集内的交易，聚合为边
"""
import argparse
import csv
import math
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

csv.field_size_limit(2**31 - 1)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "bert4eth", "raw")
REPO = os.path.join(BASE, "repos", "BERT4ETH")
OUT = os.path.join(BASE, "data", "bert4eth", "processed")
os.makedirs(OUT, exist_ok=True)

FILES = [
    "phisher_transaction_out.csv",
    "phisher_transaction_in.csv",
    "tornado_trans_in_removed.csv",
    "tornado_trans_out_removed.csv",
    "dean_trans_out_new.csv",
    "dean_trans_in_new.csv",
    "normal_eoa_transaction_in_slice_1000K.csv",
    "normal_eoa_transaction_out_slice_1000K.csv",
]


def load_labels():
    phish = set()
    with open(os.path.join(REPO, "Data", "phisher_account.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            if line.startswith("0x"):
                phish.add(line)

    tornado = set()
    with open(os.path.join(REPO, "Data", "tornado_raw_data", "tornado_account.csv"), encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) and row[0].startswith("0x"):
                tornado.add(row[0].strip().lower())

    ens = set()
    with open(os.path.join(REPO, "Data", "dean_all_ens_pairs.csv"), encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) > 1 and row[1].startswith("0x"):
                ens.add(row[1].strip().lower())
    return phish, tornado, ens


def iter_rows(path):
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if len(row) >= 8:
                try:
                    yield row[5].lower(), row[6].lower(), int(row[3]), int(row[7])
                except ValueError:
                    pass


def sample_normal(phish, n_normal, rng_seed=42):
    """normal 标签：normal_in 中作为收款方出现 ≥2 次的 EOA，抽样 n_normal 个（排除已在其他标签集的）。"""
    cnt = defaultdict(int)
    path = os.path.join(RAW, "normal_eoa_transaction_in_slice_1000K.csv")
    for a, b, _, _ in iter_rows(path):
        if b.startswith("0x"):
            cnt[b] += 1
    exclude = phish
    cands = sorted(a for a, c in cnt.items() if c >= 2 and a not in exclude)
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(len(cands), size=min(n_normal, len(cands)), replace=False)
    return {cands[i] for i in idx}


def pass1_collect(labeled, min_txs):
    """标签账户 ↔ 邻居 的交易计数；邻居保留条件：与任一标签账户交易 ≥ min_txs。"""
    neighbor_cnt = defaultdict(int)
    for fn in FILES:
        p = os.path.join(RAW, fn)
        if not os.path.exists(p):
            continue
        for a, b, _, _ in iter_rows(p):
            if a in labeled and b in labeled:
                continue
            if a in labeled:
                neighbor_cnt[b] += 1
            elif b in labeled:
                neighbor_cnt[a] += 1
    keep = {n for n, c in neighbor_cnt.items() if c >= min_txs}
    return labeled | keep, keep


def pass2_edges(nodes):
    """抽取两端都在 nodes 内的交易，聚合为边 (src,dst) → (count, sum_value, min_block, max_block)。"""
    edges = {}
    for fn in FILES:
        p = os.path.join(RAW, fn)
        if not os.path.exists(p):
            continue
        for a, b, blk, val in iter_rows(p):
            if a in nodes and b in nodes:
                key = (a, b)
                e = edges.get(key)
                if e is None:
                    edges[key] = [1, val, blk, blk]
                else:
                    e[0] += 1
                    e[1] += val
                    e[2] = min(e[2], blk)
                    e[3] = max(e[3], blk)
    return edges


def node_features(nodes, edges, labeled):
    """节点特征：交易量/度（对数）。label: 0=normal,1=phishing,2=tornado,3=ens,-1=unknown"""
    n_in = defaultdict(int)
    n_out = defaultdict(int)
    v_in = defaultdict(int)
    v_out = defaultdict(int)
    for (a, b), (c, v, _, _) in edges.items():
        n_out[a] += c
        n_in[b] += c
        v_out[a] += v
        v_in[b] += v
    feat = {}
    for node in nodes:
        feat[node] = (
            math.log1p(float(n_in[node])),
            math.log1p(float(n_out[node])),
            math.log1p(float(v_in[node])),
            math.log1p(float(v_out[node])),
        )
    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_normal", type=int, default=7000)
    ap.add_argument("--min_txs", type=int, default=2)
    args = ap.parse_args()

    phish, tornado, ens = load_labels()
    normal = sample_normal(phish, args.n_normal)
    labeled = phish | tornado | ens | normal
    print(f"labeled: phish={len(phish)} tornado={len(tornado)} ens={len(ens)} normal={len(normal)}")

    cache_file = os.path.join(OUT, f"cache_min{args.min_txs}_n{args.n_normal}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            nodes, edges = pickle.load(f)
        print("loaded cache:", cache_file)
    else:
        nodes, keep = pass1_collect(labeled, args.min_txs)
        print(f"nodes total={len(nodes)} (neighbors kept={len(keep)})")
        edges = pass2_edges(nodes)
        print(f"edges={len(edges):,}")
        with open(cache_file, "wb") as f:
            pickle.dump((nodes, edges), f)

    feat = node_features(nodes, edges, labeled)
    label_of = {}
    for a in nodes:
        if a in phish:
            label_of[a] = 1
        elif a in tornado:
            label_of[a] = 2
        elif a in ens:
            label_of[a] = 3
        elif a in normal:
            label_of[a] = 0
        else:
            label_of[a] = -1

    addr_list = sorted(nodes)
    addr2id = {a: i for i, a in enumerate(addr_list)}
    node_file = os.path.join(OUT, "nodes.npz")
    edge_file = os.path.join(OUT, "edges.npz")
    np.savez_compressed(
        node_file,
        address=np.array(addr_list),
        label=np.array([label_of[a] for a in addr_list]),
        feat=np.array([feat[a] for a in addr_list], dtype=np.float32),
    )
    es = []
    for (a, b), (c, v, blk0, blk1) in edges.items():
        # 金额以 log1p(wei) 存储，避免 int64 溢出
        es.append((addr2id[a], addr2id[b], c, math.log1p(float(v)), blk0, blk1))
    arr = np.array(es, dtype=np.float64)
    np.savez_compressed(edge_file, edges=arr)
    print("saved ->", node_file, edge_file)
    print("node labels:", {k: int((np.array([label_of[a] for a in addr_list]) == k).sum())
                            for k in [-1, 0, 1, 2, 3]})
    print("block range:", int(arr[:, 4].min()), "-", int(arr[:, 5].max()))


if __name__ == "__main__":
    main()
