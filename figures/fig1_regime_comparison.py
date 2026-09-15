# -*- coding: utf-8 -*-
"""Fig. 1 - Four-regime comparison: LP vs XGBoost, 10% seeds.

Means and 95% confidence intervals (Student-t, n = 10) are computed from the
multi-seed result files in ../results.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SKILL = r'C:\Users\Administrator\.codex\skills\nature-figure\scripts'
sys.path.insert(0, SKILL)
from audit_panel_alignment import require_matplotlib_panel_alignment

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

OUT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(OUT), 'results')

LP = '#0F4D92'
XGB = '#9AA0A6'

regimes = ['Laundering\n(t=29)', 'Mixer\n(Tornado)', 'Phishing\n(ETH)',
           'Cross-time\n(official split)']

T95 = {9: 2.262}  # n = 10 trials


def load_multi(name):
    with open(os.path.join(RES, name), encoding='utf-8') as f:
        return json.load(f)['results']['0.1']


def mean_ci(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    return v.mean(), T95[len(v) - 1] * v.std(ddof=1) / np.sqrt(len(v))


elliptic = load_multi('elliptic_t29_multiseed.json')
mixer = load_multi('multiseed_label2.json')
phish = load_multi('multiseed_label1.json')

lp_auc = np.array([mean_ci(elliptic['lp'], 'auc_roc')[0],
                   mean_ci(mixer['lp'], 'auc_roc')[0],
                   mean_ci(phish['lp'], 'auc_roc')[0], 0.500])
lp_auc_sd = np.array([mean_ci(elliptic['lp'], 'auc_roc')[1],
                      mean_ci(mixer['lp'], 'auc_roc')[1],
                      mean_ci(phish['lp'], 'auc_roc')[1], np.nan])
xgb_auc = np.array([mean_ci(elliptic['xgb'], 'auc_roc')[0],
                    mean_ci(mixer['xgb'], 'auc_roc')[0],
                    mean_ci(phish['xgb'], 'auc_roc')[0], np.nan])
xgb_auc_sd = np.array([mean_ci(elliptic['xgb'], 'auc_roc')[1],
                       mean_ci(mixer['xgb'], 'auc_roc')[1],
                       mean_ci(phish['xgb'], 'auc_roc')[1], np.nan])

lp_rec = np.array([mean_ci(elliptic['lp'], 'recall_at_fpr1')[0],
                   mean_ci(mixer['lp'], 'recall_at_fpr1')[0],
                   mean_ci(phish['lp'], 'recall_at_fpr1')[0], 0.018])
lp_rec_sd = np.array([mean_ci(elliptic['lp'], 'recall_at_fpr1')[1],
                      mean_ci(mixer['lp'], 'recall_at_fpr1')[1],
                      mean_ci(phish['lp'], 'recall_at_fpr1')[1], np.nan])
xgb_rec = np.array([mean_ci(elliptic['xgb'], 'recall_at_fpr1')[0],
                    mean_ci(mixer['xgb'], 'recall_at_fpr1')[0],
                    mean_ci(phish['xgb'], 'recall_at_fpr1')[0], np.nan])
xgb_rec_sd = np.array([mean_ci(elliptic['xgb'], 'recall_at_fpr1')[1],
                       mean_ci(mixer['xgb'], 'recall_at_fpr1')[1],
                       mean_ci(phish['xgb'], 'recall_at_fpr1')[1], np.nan])


def grouped(ax, lp, lp_sd, xgb, xgb_sd, ylabel, ymax):
    x = np.arange(len(regimes))
    w = 0.34
    b1 = ax.bar(x - w / 2, lp, width=w, color=LP, edgecolor='black',
                linewidth=0.6, label='LP (propagation)', yerr=lp_sd,
                error_kw={'elinewidth': 0.8, 'capthick': 0.8, 'capsize': 2.2})
    b2 = ax.bar(x + w / 2, xgb, width=w, color=XGB, edgecolor='black',
                linewidth=0.6, label='XGBoost (features)', yerr=xgb_sd,
                error_kw={'elinewidth': 0.8, 'capthick': 0.8, 'capsize': 2.2})
    for i, v in enumerate(lp):
        if not np.isnan(v):
            top = v + (0.0 if np.isnan(lp_sd[i]) else lp_sd[i]) + 0.014
            ax.text(x[i] - w / 2, top, f'{v:.3f}'.rstrip('0'),
                    ha='center', va='bottom', fontsize=5.4)
    for i, v in enumerate(xgb):
        if np.isnan(v):
            ax.text(x[i] + w / 2, 0.012, 'n/a†', ha='center', va='bottom',
                    fontsize=5.4, color='#555555')
        else:
            top = v + (0.0 if np.isnan(xgb_sd[i]) else xgb_sd[i]) + 0.014
            ax.text(x[i] + w / 2, top, f'{v:.3f}'.rstrip('0'),
                    ha='center', va='bottom', fontsize=5.4)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, fontsize=6.2)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ymax)
    ax.tick_params(axis='y', labelsize=6.2)


fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 2.25))
fig.subplots_adjust(wspace=0.34, top=0.80, bottom=0.24, left=0.08, right=0.98)

grouped(axa, lp_auc, lp_auc_sd, xgb_auc, xgb_auc_sd, 'AUC-ROC', 1.0)
grouped(axb, lp_rec, lp_rec_sd, xgb_rec, xgb_rec_sd, 'Recall@FPR=1%', 0.85)

handles, labels = axa.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=6.6,
           handlelength=1.2, columnspacing=1.4, bbox_to_anchor=(0.5, 1.0))

from matplotlib.transforms import ScaledTranslation
for ax, label in ((axa, 'a'), (axb, 'b')):
    off = ScaledTranslation(-4 / 72, 3 / 72, fig.dpi_scale_trans)
    ax.text(0, 1, label, transform=ax.transAxes + off, fontsize=8,
            fontweight='bold', ha='left', va='bottom')

fig.text(0.5, 0.015,
         '† No same-seed feature baseline exists under the official split '
         '(full-label XGBoost = 0.9231, §6.2). Bars: 10-trial means, 95% CI; '
         'cross-time LP is a single run.',
         ha='center', fontsize=5.4, color='#444444')

require_matplotlib_panel_alignment(
    fig,
    axes=[axa, axb],
    panel_ids=['a', 'b'],
    json_out=os.path.join(OUT, 'fig1_regime_comparison.alignment.json'),
    overlay_svg=os.path.join(OUT, 'fig1_regime_comparison.alignment.svg'),
    tolerance_pt=1.5,
    gutter_tolerance_pt=1.5,
    require_panel_labels=True,
    strict=True,
)

fig.savefig(os.path.join(OUT, 'fig1_regime_comparison.svg'),
            bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'fig1_regime_comparison.pdf'),
            bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'fig1_regime_comparison.png'),
            dpi=600, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'fig1_regime_comparison.tiff'),
            dpi=600, bbox_inches='tight')
plt.close(fig)
print('fig1 saved')
