# -*- coding: utf-8 -*-
"""Systematic audit of the path-backtracking claim (Section 8 of the rework).

Instead of three illustrative paths, evaluate the top-K highest-credit held-out
accounts in the mixer regime and report how often a path to a confirmed seed can
be retrieved, how long it is, and whether it passes through normal/unknown nodes.
"""
import argparse
import collections
import io
import json
import os

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, 'data', 'bert4eth', 'processed')
PROTO = os.path.join(BASE, 'results', 'eth_protocol.npz')
OUT_JSON = os.path.join(BASE, 'analysis', 'interpretability_audit.json')
OUT_MD = os.path.join(BASE, 'analysis', 'interpretability_audit.md')


def lp_credit(n, src, dst, deg, seed_p, seed_n, iters=100):
    F = np.zeros((n, 2))
    F[seed_p, 0] = 1.0
    F[seed_n, 1] = 1.0
    onehot = F.copy()
    seed = np.concatenate([seed_p, seed_n])
    for _ in range(iters):
        msg = (F[src] + F[dst]) / 2.0
        new = np.zeros_like(F)
        np.add.at(new, src, msg / deg[src, None])
        np.add.at(new, dst, msg / deg[dst, None])
        new[seed] = onehot[seed]
        if np.abs(new - F).max() < 1e-10:
            F = new
            break
        F = new
    return F[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', type=int, default=2)
    ap.add_argument('--frac', type=float, default=0.1)
    ap.add_argument('--trial', type=int, default=0)
    ap.add_argument('--seed-base', type=int, default=700)
    ap.add_argument('--top-k', type=int, default=100)
    ap.add_argument('--max-depth', type=int, default=6)
    args = ap.parse_args()

    nd = np.load(os.path.join(PROC, 'nodes.npz'), allow_pickle=True)
    ed = np.load(os.path.join(PROC, 'edges.npz'))
    proto = np.load(PROTO)
    labels = nd['label'].astype(int)
    edges = ed['edges']
    n = len(labels)
    y_pos = (labels == args.label).astype(int)
    cand = (labels == args.label) | (labels == 0)
    pos = np.where((labels == args.label) & cand)[0]
    neg = np.where((labels == 0) & cand)[0]
    src = edges[:, 0].astype(int)
    dst = edges[:, 1].astype(int)
    deg = np.bincount(np.concatenate([src, dst]), minlength=n).astype(float)
    deg[deg == 0] = 1.0

    rng = np.random.default_rng(args.seed_base + args.trial)
    sp = rng.choice(pos, size=max(1, int(len(pos) * args.frac)), replace=False)
    sn = rng.choice(neg, size=max(1, int(len(neg) * args.frac)), replace=False)
    seed = np.concatenate([sp, sn])
    val_nodes = np.where(proto['val_mask'] & cand)[0]
    val_v = np.setdiff1d(val_nodes, seed)
    eval_mask = cand.copy()
    eval_mask[seed] = False
    eval_mask[val_v] = False
    hold = np.where(eval_mask)[0]
    assert not np.intersect1d(seed, hold).size
    assert not np.intersect1d(val_v, hold).size

    credit = lp_credit(n, src, dst, deg, sp, sn)
    order = hold[np.argsort(-credit[hold])]
    top = order[:args.top_k]

    adj = collections.defaultdict(list)
    for a, b in zip(src, dst):
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    seed_pos = set(int(x) for x in sp)

    paths = []
    for node in top:
        node = int(node)
        prev = {node: None}
        q = collections.deque([node])
        found = None
        while q and found is None:
            u = q.popleft()
            for v in adj.get(u, ()):
                if v in prev:
                    continue
                prev[v] = u
                if v in seed_pos:
                    found = v
                    break
                q.append(v)
        if found is None:
            paths.append({'node': node, 'path_found': False, 'length': None,
                          'via_nonfraud': None})
        else:
            path, cur = [], found
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            via_nonfraud = any(labels[x] == 0 for x in path)
            paths.append({'node': node, 'path_found': True,
                          'length': len(path) - 1,
                          'via_nonfraud': bool(via_nonfraud)})

    found_paths = [p for p in paths if p['path_found']]
    lengths = sorted(p['length'] for p in found_paths)
    summary = {
        'regime': 'Tornado (mixer), transductive all-time graph',
        'seed_fraction': args.frac,
        'trial': args.trial,
        'seed_base': args.seed_base,
        'top_k': args.top_k,
        'evaluated_accounts': len(paths),
        'path_retrieval_rate': round(len(found_paths) / len(paths), 3),
        'median_path_length': (lengths[len(lengths) // 2]
                               if lengths else None),
        'no_path_fraction': round(1 - len(found_paths) / len(paths), 3),
        'path_via_normal_or_licit_fraction': (
            round(sum(1 for p in found_paths if p['via_nonfraud'])
                  / len(found_paths), 3) if found_paths else None),
    }
    with io.open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'paths': paths}, f,
                  ensure_ascii=False, indent=2)
    lines = ['# Interpretability audit (path backtracking)', '',
             'Regime: %s; %g%% seeds; trial %d of seed base %d.' % (
                 summary['regime'], args.frac * 100, args.trial,
                 args.seed_base), '',
             '| Quantity | Value |', '|---|---|',
             '| High-credit held-out accounts evaluated | %d |'
             % summary['evaluated_accounts'],
             '| Path retrieval rate | %.3f |' % summary['path_retrieval_rate'],
             '| Median path length (hops) | %s |'
             % summary['median_path_length'],
             '| Accounts with no seed path | %.3f |'
             % summary['no_path_fraction'],
             '| Paths passing through normal/licit nodes | %s |'
             % summary['path_via_normal_or_licit_fraction'], '',
             'These quantities describe a systematic audit of the top-%d '
             'accounts. They do not establish human utility; no analyst user '
             'study was conducted.' % args.top_k]
    with io.open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('saved ->', OUT_JSON)
    print('saved ->', OUT_MD)


if __name__ == '__main__':
    main()
