# -*- coding: utf-8 -*-
"""Compare run 1 and run 2 of the corrected pipeline.

Both runs use the same seed base and per-trial seeds. Every per-trial metric is
compared element-wise; the check passes when the maximum absolute difference is
below TOL.
"""
import io
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN1 = os.path.join(BASE, 'results')
RUN2 = os.path.join(BASE, '..', 'results_run2')
OUT = os.path.join(BASE, 'analysis', 'reproduction_check.md')
TOL = 5e-3          # GPU float32 kernels are not bit-reproducible

FILES = [('multiseed_label2.json', None),
         ('multiseed_label1.json', None),
         ('sage_label2.json', 'graphsage'),
         ('sage_label1.json', 'graphsage')]


def walk(a, b, path, diffs, mismatches):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k == 'seconds':      # wall-clock timings are expected to differ
                continue
            if k in b:
                walk(a[k], b[k], path + '/' + str(k), diffs, mismatches)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                diffs.append(abs(float(x) - float(y)))
            else:
                walk(x, y, path + '[%d]' % i, diffs, mismatches)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        diffs.append(abs(float(a) - float(b)))
    elif a != b:
        mismatches.append(path)


def main():
    lines = ['# Reproduction check (run 1 vs run 2)', '',
             'Same seed base, same per-trial seeds. Tolerance for per-trial '
             'values: %g (GPU float32). Reported means must agree at three '
             'decimals.' % TOL, '',
             '| File | Max abs per-trial difference | Mismatched rounded means | Verdict |',
             '|---|---|---|---|']
    ok_all = True
    for name, _ in FILES:
        p1 = os.path.join(RUN1, name)
        p2 = os.path.join(os.path.normpath(RUN2), name)
        if not (os.path.exists(p1) and os.path.exists(p2)):
            lines.append('| %s | n/a | missing file | SKIP |' % name)
            continue
        a = json.load(io.open(p1, encoding='utf-8'))['results']
        b = json.load(io.open(p2, encoding='utf-8'))['results']
        diffs, mismatches = [], []
        rounded_mismatch = 0
        for frac in a:
            if frac in b:
                walk(a[frac], b[frac], frac, diffs, mismatches)
                for m, rows in a[frac].items():
                    if not isinstance(rows, list) or not rows:
                        continue
                    if not isinstance(rows[0], dict):
                        continue
                    for met in ('auc_roc', 'auc_pr', 'recall_at_fpr1'):
                        v1 = sum(r[met] for r in rows) / len(rows)
                        v2 = sum(r[met] for r in b[frac][m]) / len(b[frac][m])
                        if round(v1, 3) != round(v2, 3):
                            rounded_mismatch += 1
        mx = max(diffs) if diffs else 0.0
        ok = mx <= TOL and not mismatches and rounded_mismatch == 0
        ok_all &= ok
        lines.append('| %s | %.2e | %d | %s |' % (name, mx, rounded_mismatch,
                                                  'PASS' if ok else 'FAIL'))
    lines += ['', 'Overall: **%s**' % ('PASS' if ok_all else 'FAIL')]
    with io.open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('saved ->', OUT)


if __name__ == '__main__':
    main()
