"""
Summary bar-chart figure comparing four emission models across four biological
quality metrics. Produces a 2x2 panel figure saved to results/figures/.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Path setup – ROOT is three directories up from this script
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from experiments.summarize_evaluation import (
    roadmap_ari,
    mean_state_purity,
    tpm_dynamic_range,
    promoter_states,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS = ['bernoulli', 'poisson', 'nb', 'chromhmm']
DISPLAY_NAMES = ['Bernoulli', 'Poisson', 'NB', 'ChromHMM']
COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

CHROMHMM_IDX = MODELS.index('chromhmm') # index 3


def _collect_values(fn):
    """Call *fn* for every model and return a list of floats."""
    return [fn(m) for m in MODELS]


def _bar_panel(ax, values, ylabel, title, fmt_fn, ref_line_y, ref_label=None):
    """Draw a single grouped-bar panel.

    Parameters
    ----------
    ax : matplotlib Axes
    values : list of numeric values, one per model
    ylabel : y-axis label string
    title : panel title
    fmt_fn : callable(value) -> str for the bar-top annotation
    ref_line_y : y-value for the horizontal dashed reference line (or None)
    ref_label : optional text label for the reference line
    """
    x = np.arange(len(MODELS))
    bars = ax.bar(x, values, color=COLORS, width=0.55, zorder=3)

    # Value labels on top of each bar
    for bar, val in zip(bars, values):
        label = fmt_fn(val)
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01 * max(abs(v) for v in values if not np.isnan(v)),
            label,
            ha='center', va='bottom',
            fontsize=9, fontweight='bold',
        )

    # Horizontal dashed reference line
    if ref_line_y is not None:
        ax.axhline(ref_line_y, color='#555555', linewidth=1.2,
                   linestyle='--', zorder=2,
                   label=ref_label if ref_label else '_nolegend_')

    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY_NAMES, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.7, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Small breathing room above the tallest bar
    valid = [v for v in values if not np.isnan(v)]
    top = max(valid) if valid else 1.0
    ax.set_ylim(bottom=0, top=top * 1.18)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    ari_vals = _collect_values(roadmap_ari)
    purity_vals = _collect_values(mean_state_purity)
    tpm_vals = _collect_values(tpm_dynamic_range)
    promoter_vals = _collect_values(promoter_states)

    chromhmm_ari = ari_vals[CHROMHMM_IDX]
    chromhmm_purity = purity_vals[CHROMHMM_IDX]
    chromhmm_tpm = tpm_vals[CHROMHMM_IDX]

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    fig.subplots_adjust(top=0.88, hspace=0.4, wspace=0.35)
    fig.suptitle(
        'Biological quality comparison across emission models',
        fontsize=13, fontweight='bold',
    )

    _bar_panel(axes[0, 0], ari_vals, 'ARI', 'Roadmap ARI',
               lambda v: f'{v:.3f}', chromhmm_ari)
    _bar_panel(axes[0, 1], purity_vals, 'Mean purity (%)', 'Mean state purity (%)',
               lambda v: f'{v:.1f}', chromhmm_purity)
    _bar_panel(axes[1, 0], tpm_vals, 'Max / mean median TPM', 'TPM dynamic range',
               lambda v: f'{v:.1f}x', chromhmm_tpm)
    _bar_panel(axes[1, 1], promoter_vals, 'States with PLS FE > 5×', 'Promoter states',
               lambda v: f'{int(v):d}', ref_line_y=1)

    out_dir = os.path.join(ROOT, 'results', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'summary_barchart.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    run()
