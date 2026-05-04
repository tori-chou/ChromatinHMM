"""
One heatmap per model (Bernoulli, Poisson, NB) showing emission parameters.
Rows = states (biological labels), columns = histone marks.

Count models (Poisson, NB) are column-normalised so every mark sits on [0,1].
Bernoulli probabilities are already in [0,1] and are left as-is.
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

MODELS_DIR = os.path.join(ROOT, 'models')
OUT_DIR = os.path.join(ROOT, 'results', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

MARK_NAMES = ['H3K27me3', 'H3K36me3', 'H3K4me1', 'H3K4me3', 'H3K9me3']

MODELS = ['bernoulli', 'poisson', 'nb']
TITLES = ['Bernoulli', 'Poisson', 'Negative Binomial']

MODEL_KEY = {'bernoulli': 'theta', 'poisson': 'lam', 'nb': 'mu'}
IS_COUNT = {'bernoulli': False,   'poisson': True,   'nb': True}


def load_emissions(model_name):
    """Return raw emission matrix (K, 5) for the given model."""
    path = os.path.join(MODELS_DIR, f'{model_name}_model.npz')
    d = np.load(path, allow_pickle=True)
    key = MODEL_KEY[model_name]
    return d[key].astype(float) # (10, 5)


def plot_emission_heatmap(model_name, title):
    params = load_emissions(model_name) # (K, 5)
    bio_labels = get_state_labels(model_name) # {int: str}
    n_states = params.shape[0]

    # Build ordered state labels (0-based index → label string)
    state_labels = [bio_labels.get(i, f'State {i + 1}') for i in range(n_states)]

    # Column-normalise count models so all marks are on [0,1]
    count_model = IS_COUNT[model_name]
    if count_model:
        col_max = params.max(axis=0, keepdims=True) # (1, 5)
        col_max = np.where(col_max == 0, 1, col_max) # avoid /0
        display_data = params / col_max
    else:
        display_data = params

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    im = ax.imshow(display_data, aspect='auto', vmin=0, vmax=1,
                   cmap='Reds', interpolation='nearest')

    # x-axis: mark names, no rotation
    ax.set_xticks(range(len(MARK_NAMES)))
    ax.set_xticklabels(MARK_NAMES, rotation=0, fontsize=9)

    # y-axis: biological state labels
    ax.set_yticks(range(n_states))
    ax.set_yticklabels(state_labels, fontsize=9)

    # title
    suffix = ' (column-normalized)' if count_model else ''
    param_lbl = 'Emission means' if count_model else 'Emission probabilities'
    ax.set_title(f'{title}{suffix} — {param_lbl}',
                 fontsize=11, fontweight='bold')

    # cell annotations
    if count_model:
        fmt = lambda v: f'{v:.1f}' # raw value
    else:
        fmt = lambda v: f'{v:.3f}'

    for r in range(n_states):
        for c in range(len(MARK_NAMES)):
            raw_val = params[r, c]
            disp_val = display_data[r, c]
            txt_color = 'white' if disp_val > 0.65 else 'black'
            ax.text(c, r, fmt(raw_val),
                    ha='center', va='center',
                    fontsize=7, color=txt_color)

    # colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
    if count_model:
        cbar.set_label('Normalized mean count', fontsize=9)
    else:
        cbar.set_label('Probability', fontsize=9)

    out = os.path.join(OUT_DIR, f'emission_heatmap_{model_name}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def run():
    for model, title in zip(MODELS, TITLES):
        plot_emission_heatmap(model, title)


if __name__ == '__main__':
    run()
