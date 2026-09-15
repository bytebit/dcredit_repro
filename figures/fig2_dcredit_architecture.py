# -*- coding: utf-8 -*-
"""Fig. 1 - D-Credit architecture (schematic, single panel)."""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.linewidth": 0.8,
})

OUT = os.path.dirname(os.path.abspath(__file__))

BLUE = '#0F4D92'
GREY = '#767676'
TEAL = '#42949E'


def box(ax, x, y, w, h, text, sub=None, fc='#FFFFFF', ec=BLUE, fs=7,
        sub_fs=5.5, gid=None):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle='round,pad=0.008', linewidth=1.1,
                       edgecolor=ec, facecolor=fc, zorder=3, gid=gid)
    ax.add_patch(p)
    if sub:
        ax.text(x, y + 0.12, text, ha='center', va='center', fontsize=fs,
                fontweight='bold', color='#111111', zorder=4,
                gid=f'{gid}-title' if gid else None)
        ax.text(x, y - 0.12, sub, ha='center', va='center', fontsize=sub_fs,
                color='#444444', zorder=4,
                gid=f'{gid}-sub' if gid else None)
    else:
        ax.text(x, y, text, ha='center', va='center', fontsize=fs,
                fontweight='bold', color='#111111', zorder=4,
                gid=f'{gid}-text' if gid else None)


def arrow(ax, x0, y0, x1, y1, color=GREY, lw=1.3, style='-|>',
          linestyle='solid', gid=None):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                        mutation_scale=11, linewidth=lw, color=color, zorder=2,
                        linestyle=linestyle, gid=gid)
    ax.add_patch(a)


fig, ax = plt.subplots(figsize=(7.2, 3.15))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis('off')

# Credit channel (top row)
box(ax, 1.0, 4.25, 1.6, 0.72, 'Labeled\nfraud seeds', ec=BLUE,
    gid='box-seeds')
box(ax, 3.6, 4.25, 2.6, 1.05, 'Credit channel (LP / PPR)',
    sub='teleport α · operator γ · amount β',
    ec=BLUE, gid='box-credit')
box(ax, 6.4, 4.25, 1.5, 0.72, 'Credit score\nφ(c)', ec=BLUE,
    gid='box-credit-score')
box(ax, 8.85, 2.75, 1.7, 0.98, 'score =\nw·φ(c) + (1−w)·b(x)', ec='#111111',
    gid='box-score')

# Behavior channel (bottom row)
box(ax, 1.0, 1.55, 1.6, 0.72, 'Node features\n(degree, amount)', ec=GREY,
    gid='box-features')
box(ax, 3.6, 1.55, 2.4, 1.0, 'Behavior channel b(x)',
    sub='MLP or XGBoost', ec=GREY, gid='box-behavior')

# Gate + fusion
box(ax, 6.0, 2.75, 2.15, 1.0, 'Learnable gate w in [0,1]',
    sub='learned on validation', ec=TEAL, gid='box-gate')
box(ax, 3.7, 2.75, 1.1, 0.55, 'Validation\nset', fc='#FFFFFF',
    ec=TEAL, fs=5.8, gid='box-validation')

# Arrows
arrow(ax, 1.80, 4.25, 2.30, 4.25, gid='arrow-seeds-credit')
arrow(ax, 4.90, 4.25, 5.65, 4.25, gid='arrow-credit-score')
arrow(ax, 7.05, 3.89, 8.45, 3.24, gid='arrow-phi-score')
arrow(ax, 1.80, 1.55, 2.40, 1.55, gid='arrow-features-behavior')
arrow(ax, 4.80, 1.55, 8.45, 2.26, gid='arrow-bx-score')
arrow(ax, 4.25, 2.75, 4.925, 2.75, color=TEAL, linestyle='dashed',
      gid='arrow-validation-gate')
arrow(ax, 7.075, 2.75, 8.00, 2.75, linestyle='dashed',
      gid='arrow-gate-score')

# Arrow labels (data flows)
ax.text(7.78, 3.70, 'φ(c)', ha='center', va='bottom', fontsize=5.8,
        color='#111111', gid='label-phi')
ax.text(6.60, 1.55, 'b(x)', ha='center', va='bottom', fontsize=5.8,
        color='#111111', gid='label-bx')
ax.text(7.52, 2.92, 'w', ha='center', va='bottom', fontsize=5.8,
        color='#111111', gid='label-w')

# Regime annotation under the gate
ax.text(6.2, 0.32, 'gate adapts to the regime: w ≈ 0.85–0.90 on the mixer, '
        'w ≈ 0.3–0.65 on phishing',
        ha='center', va='center', fontsize=5.8, color='#333333',
        gid='label-annotation')

fig.savefig(os.path.join(OUT, 'fig2_dcredit_architecture.svg'),
            bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'fig2_dcredit_architecture.pdf'),
            bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'fig2_dcredit_architecture.png'),
            dpi=600, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'fig2_dcredit_architecture.tiff'),
            dpi=600, bbox_inches='tight')
plt.close(fig)
print('fig2 saved')
