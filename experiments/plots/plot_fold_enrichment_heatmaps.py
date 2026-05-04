"""
4-panel figure: fold enrichment over genomic background for each model.
Rows = HMM states (active only), columns = annotations.
Colour is log2(fold enrichment), capped at log2(40) to prevent outliers
from washing out the scale.
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

BIO_DIR = os.path.join(ROOT, 'results', 'biological_analysis')
OUT_DIR = os.path.join(ROOT, 'results', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ['bernoulli', 'poisson', 'nb', 'chromhmm']
TITLES = ['Bernoulli', 'Poisson', 'Negative Binomial', 'ChromHMM (reference)']
VMAX = np.log2(40) # cap: 40x enrichment


def load(model_name):
    path = os.path.join(BIO_DIR, f'biological_analysis_{model_name}.npz')
    d = np.load(path, allow_pickle=True)
    fe = d['fold_enrichment'].astype(float)
    total_bp = d['total_bp'].astype(float)
    anno_lbl = [str(s) for s in d['annotation_labels']]
    active = total_bp > 0
    bio_labels = get_state_labels(model_name)
    slabels = [bio_labels[i] for i in range(len(active)) if active[i]]
    return fe[active], slabels, anno_lbl


def run():
    for model, title in zip(MODELS, TITLES):
        fe, slabels, alabels = load(model)

        with np.errstate(divide='ignore', invalid='ignore'):
            log_fe = np.where(fe > 0, np.log2(np.maximum(fe, 1e-9)), 0.0)

        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

        im = ax.imshow(log_fe, aspect='auto', vmin=0, vmax=VMAX,
                       cmap='Blues', interpolation='nearest')

        ax.set_xticks(range(len(alabels)))
        ax.set_xticklabels(alabels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(slabels)))
        ax.set_yticklabels(slabels, fontsize=9)
        ax.set_title(f'{title} — Fold enrichment over background', fontsize=11, fontweight='bold')
        ax.axvline(2.5, color='grey', linewidth=0.8, linestyle='--')

        for r in range(fe.shape[0]):
            for c in range(fe.shape[1]):
                v = fe[r, c]
                if v >= 2.0:
                    ax.text(c, r, f'{v:.1f}x', ha='center', va='center',
                            fontsize=7,
                            color='white' if log_fe[r, c] > VMAX * 0.65 else 'black')

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
        ticks_fe = [1, 2, 5, 10, 20, 40]
        cbar.set_ticks([np.log2(t) for t in ticks_fe])
        cbar.set_ticklabels([f'{t}x' for t in ticks_fe])
        cbar.set_label('Fold enrichment over background', fontsize=9)

        out = os.path.join(OUT_DIR, f'fold_enrichment_heatmap_{model}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved to {out}')


if __name__ == '__main__':
    run()
