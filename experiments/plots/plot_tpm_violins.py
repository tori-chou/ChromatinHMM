"""
Plot per-state TPM distributions as violin plots, one figure per model.
States are ordered by median TPM (ascending) and labeled with biological names.

Saves: results/figures/tpm_violins_{model}.png
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from experiments.label_states import get_state_labels

MODELS = ['bernoulli', 'poisson', 'nb', 'chromhmm']
TITLES = ['Bernoulli', 'Poisson', 'Negative Binomial', 'ChromHMM (reference)']

EXPR_DIR = os.path.join(ROOT, 'results', 'expression_correlation')
FIG_DIR = os.path.join(ROOT, 'results', 'figures')

LOG1P_1 = np.log1p(1.0) # reference line at TPM = 1


def run():
    os.makedirs(FIG_DIR, exist_ok=True)

    for model, title in zip(MODELS, TITLES):

        npz_path = os.path.join(EXPR_DIR, f'expression_correlation_{model}.npz')
        data = np.load(npz_path, allow_pickle=True)

        state_labels = get_state_labels(model)

        states_data = []
        for key in data.files:
            if not key.startswith('tpm_state'):
                continue
            i = int(key[len('tpm_state'):])
            arr = data[key]
            if len(arr) >= 5:
                states_data.append((i, arr))

        if not states_data:
            print(f'[{model}] No states with n_genes >= 5 — skipping.')
            continue

        states_data.sort(key=lambda x: float(np.median(x[1])))

        indices = [s[0] for s in states_data]
        tpm_arrs = [s[1] for s in states_data]
        xlabels = [state_labels.get(i, f'State {i}') for i in indices]

        log_arrs = [np.log1p(arr.astype(float)) for arr in tpm_arrs]

        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

        parts = ax.violinplot(
            log_arrs,
            positions=range(len(log_arrs)),
            showmedians=True,
            showextrema=False,
        )

        for pc in parts['bodies']:
            pc.set_facecolor('steelblue')
            pc.set_edgecolor('steelblue')
            pc.set_alpha(0.7)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.5)

        ax.axhline(LOG1P_1, color='firebrick', linestyle='--', linewidth=1.0,
                   label='TPM = 1 threshold')

        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('TPM + 1 (log scale)', fontsize=10)
        ax.set_xlabel('')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')

        out_path = os.path.join(FIG_DIR, f'tpm_violins_{model}.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {out_path}')


if __name__ == '__main__':
    run()
