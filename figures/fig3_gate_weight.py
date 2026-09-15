# -*- coding: utf-8 -*-
"""Fig. 3 - Learned gate weight w across regimes (10 trials, one dot per trial)."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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

TEAL = '#42949E'

groups = ['Mixer\n10% seeds', 'Mixer\n50% seeds',
          'Phishing\n10% seeds', 'Phishing\n50% seeds']


def gate_values(name, frac):
    with open(os.path.join(RES, name), encoding='utf-8') as f:
        return json.load(f)['results'][frac]['gate_w_auc']


trials = [
    gate_values('multiseed_label2.json', '0.1'),
    gate_values('multiseed_label2.json', '0.5'),
    gate_values('multiseed_label1.json', '0.1'),
    gate_values('multiseed_label1.json', '0.5'),
]
means = [np.mean(t) for t in trials]

# Fixed display offsets only (jitter for readability); they are not data.
jitter = np.linspace(-0.16, 0.16, len(trials[0]))
fig, ax = plt.subplots(figsize=(3.5, 2.35))
for i, t in enumerate(trials):
    # Per-trial w values are plotted directly (n = 10 trials), so the full
    # distribution is visible and no summary error bar is needed.
    ax.scatter(np.full(len(t), i) + jitter, t, s=22, color=TEAL,
               edgecolor='black', linewidth=0.5, zorder=3,
               label='learned w (10 trials)' if i == 0 else None)
    ax.scatter([i], [means[i]], marker='D', s=28, color='#0F4D92',
               edgecolor='black', linewidth=0.6, zorder=4,
               label='mean' if i == 0 else None)
    ax.text(i, max(t) + 0.15, f'{means[i]:.2f}', ha='center', va='bottom',
            fontsize=5.8)

ax.axhline(0.5, color='#767676', linestyle='--', linewidth=0.8)
ax.text(0.0, 0.58, 'w = 0.5 (balanced)', ha='left', va='bottom',
        fontsize=5.4, color='#555555')
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(groups, fontsize=6.2)
ax.set_ylabel('Learned gate weight w', fontsize=6.8)
ax.set_ylim(-0.05, 1.35)
ax.tick_params(axis='y', labelsize=6.2)
ax.legend(fontsize=5.8, loc='upper left')
ax.set_title('The gate learns the regime', fontsize=7.2, pad=3)

fig.savefig(os.path.join(OUT, 'fig3_gate_weight.svg'), bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'fig3_gate_weight.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'fig3_gate_weight.png'), dpi=600,
            bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'fig3_gate_weight.tiff'), dpi=600,
            bbox_inches='tight')
plt.close(fig)
print('fig3 saved')
