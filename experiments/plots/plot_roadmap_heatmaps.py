"""
4-panel figure: row-normalised confusion matrix for each model vs Roadmap 15-state.
Rows = HMM states, columns = Roadmap annotations, colour = % of bins in that state
assigned to each Roadmap class.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from experiments.label_states import get_state_labels

ROADMAP_DIR = os.path.join(ROOT, 'results', 'roadmap_comparison')
OUT_DIR = os.path.join(ROOT, 'results', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ['bernoulli', 'poisson', 'nb', 'chromhmm']
TITLES = ['Bernoulli', 'Poisson', 'Negative Binomial', 'ChromHMM (reference)']

# Roadmap 15-state colour palette (standard ChromHMM colours)
ROADMAP_COLORS = [
    '#FF0000', '#FF4500', '#32CD32', '#008000', '#006400',
    '#C2E105', '#FFFF00', '#66CDAA', '#8A91D0', '#E6B8B7',
    '#7B9D7B', '#FFC34D', '#808080', '#C0C0C0', '#FFFFFF',
]


def load(model_name):
    path = os.path.join(ROADMAP_DIR, f'roadmap_confusion_{model_name}.npz')
    d = np.load(path, allow_pickle=True)
    pct = d['pct'].astype(float)
    counts = d['counts'].astype(float)
    rlabels = [str(s) for s in d['roadmap_labels']]
    active = counts.sum(axis=1) > 0
    bio_labels = get_state_labels(model_name) # {0-based idx: label}
    active_labels = [bio_labels[i] for i in range(len(active)) if active[i]]
    return pct[active], active_labels, rlabels


def run():
    for model, title in zip(MODELS, TITLES):
        pct, active_labels, rlabels = load(model)

        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

        im = ax.imshow(pct, aspect='auto', vmin=0, vmax=100,
                       cmap='YlOrRd', interpolation='nearest')

        short = [s.split('_', 1)[1] for s in rlabels]
        ax.set_xticks(range(len(rlabels)))
        ax.set_xticklabels(short, rotation=55, ha='right', fontsize=8)
        ax.set_yticks(range(len(active_labels)))
        ax.set_yticklabels(active_labels, fontsize=9)
        ax.set_title(f'{title} — HMM states vs Roadmap E116', fontsize=11, fontweight='bold')

        for r in range(pct.shape[0]):
            for c in range(pct.shape[1]):
                v = pct[r, c]
                if v >= 10:
                    ax.text(c, r, f'{v:.0f}', ha='center', va='center',
                            fontsize=7, color='black' if v < 70 else 'white')

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
        cbar.set_label('% of state bins', fontsize=9)

        out = os.path.join(OUT_DIR, f'roadmap_heatmap_{model}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved to {out}')


if __name__ == '__main__':
    run()
