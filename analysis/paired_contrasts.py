# -*- coding: utf-8 -*-
"""Paired gate contrasts over the per-trial arrays in the result JSON files.

For every regime and seed fraction:
    gate - LP      and     gate - XGBoost

Reported per contrast: mean paired difference, Student-t 95% CI, the ten paired
differences, a two-sided paired t-test p value, and Holm-adjusted p values across
the planned family. Marginal confidence intervals are never used to infer an
advantage.

Primary family: Recall@FPR=1% (the deployment metric), 2 regimes x 2 fractions x
2 contrasts = 8 tests. AUC-ROC and AUC-PR are reported with their own Holm
families for completeness and are labelled as secondary.
"""
import io
import json
import os

import numpy as np
from scipy import stats as st

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, 'results')
OUT_MD = os.path.join(BASE, 'analysis', 'paired_contrasts.md')
OUT_JSON = os.path.join(BASE, 'analysis', 'paired_contrasts.json')

REGIMES = [('multiseed_label2.json', 'Tornado'),
           ('multiseed_label1.json', 'Phishing')]
FRACTIONS = ['0.1', '0.5']
CONTRASTS = [('gate_fusion', 'lp', 'gate - LP'),
             ('gate_fusion', 'xgb', 'gate - XGBoost')]
METRICS = [('auc_roc', 'AUC-ROC', 'secondary'),
           ('auc_pr', 'AUC-PR', 'secondary'),
           ('recall_at_fpr1', 'Recall@FPR=1%', 'primary')]


def holm(pvals):
    """Holm step-down adjusted p values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adj[idx] = running
    return adj


def main():
    records = []
    for fname, regime in REGIMES:
        path = os.path.join(RES, fname)
        if not os.path.exists(path):
            print('missing:', path)
            continue
        data = json.load(io.open(path, encoding='utf-8'))['results']
        for frac in FRACTIONS:
            block = data[frac]
            for metric, metric_label, tier in METRICS:
                for num, base, label in CONTRASTS:
                    a = np.array([r[metric] for r in block[num]], dtype=float)
                    b = np.array([r[metric] for r in block[base]], dtype=float)
                    if len(a) != len(b) or len(a) < 2:
                        continue
                    d = a - b
                    n = len(d)
                    mean = float(d.mean())
                    sd = float(d.std(ddof=1))
                    half = float(st.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
                    tstat, pval = st.ttest_rel(a, b)
                    records.append({
                        'comparison': '%s, %s, %s seeds' % (label, regime,
                                                           str(float(frac) * 100)
                                                           .rstrip('0').rstrip('.')),
                        'regime': regime,
                        'seed_fraction': float(frac),
                        'metric': metric,
                        'metric_label': metric_label,
                        'tier': tier,
                        'n_trials': n,
                        'mean_difference': round(mean, 4),
                        'ci95_half_width': round(half, 4),
                        'ci95': [round(mean - half, 4), round(mean + half, 4)],
                        'paired_differences': [round(float(x), 4) for x in d],
                        'test': 'two-sided paired t-test (ttest_rel)',
                        't_statistic': round(float(tstat), 3),
                        'p_value': round(float(pval), 6),
                    })

    for tier in ('primary', 'secondary'):
        idx = [i for i, r in enumerate(records) if r['tier'] == tier]
        if idx:
            for i, adj in zip(idx, holm([records[i]['p_value'] for i in idx])):
                records[i]['holm_adjusted_p'] = round(float(adj), 6)

    payload = {
        'design': {
            'paired_within_trial': True,
            'n_trials': 10,
            'primary_family': 'Recall@FPR=1%: 2 regimes x 2 fractions x 2 contrasts',
            'correction': 'Holm step-down within each metric family',
            'note': 'Advantages are inferred from paired differences and their '
                    'confidence intervals, never from overlap of marginal CIs.',
        },
        'contrasts': records,
    }
    with io.open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = ['# Paired gate contrasts', '',
             'Paired within-trial differences (gate minus baseline) over 10 '
             'independent seed draws. Intervals are Student-t 95% CIs of the '
             'paired differences; p values come from two-sided paired t-tests, '
             'Holm-adjusted within each metric family.', '',
             'Primary family: Recall@FPR=1%. AUC-ROC and AUC-PR are secondary.', '']
    for tier, title in (('primary', 'Primary: Recall@FPR=1%'),
                        ('secondary', 'Secondary: AUC metrics')):
        rows = [r for r in records if r['tier'] == tier]
        if not rows:
            continue
        lines += ['## ' + title, '',
                  '| Comparison | Metric | Mean paired diff | 95% CI | '
                  'Holm p | n |',
                  '|---|---|---|---|---|---|']
        for r in rows:
            lines.append('| %s | %s | %+.3f | [%+.3f, %+.3f] | %.4f | %d |' % (
                r['comparison'], r['metric_label'], r['mean_difference'],
                r['ci95'][0], r['ci95'][1],
                r.get('holm_adjusted_p', float('nan')), r['n_trials']))
        lines.append('')
    lines += ['## Ten paired differences', '']
    for r in records:
        lines.append('- **%s** (%s): %s' % (
            r['comparison'], r['metric_label'],
            ', '.join('%+.3f' % x for x in r['paired_differences'])))
    with io.open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('saved ->', OUT_JSON)
    print('saved ->', OUT_MD)
    for r in records:
        if r['tier'] == 'primary':
            print('%-38s %+.3f [%+.3f,%+.3f] Holm p=%.4f' % (
                r['comparison'], r['mean_difference'], r['ci95'][0],
                r['ci95'][1], r.get('holm_adjusted_p', float('nan'))))


if __name__ == '__main__':
    main()
