"""
Compare Bernoulli model emission probabilities against ChromHMM's
to validate that both models learned similar chromatin state biology.

States are matched via the Hungarian algorithm (maximise correlation).
"""

import os
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
from utils import MARKS, MODELS as MODELS_DIR

CHROMHMM_MARK_ORDER = ['H3K27me3', 'H3K4me1', 'H3K4me3', 'H3K36me3', 'H3K9me3']
REORDER = [CHROMHMM_MARK_ORDER.index(m) for m in MARKS]


def run():
    data = np.load(os.path.join(ROOT, MODELS_DIR, 'bernoulli_model.npz'))
    theta = data['theta']

    chromhmm = np.loadtxt(
        os.path.join(ROOT, 'data', 'emissions_10_42.txt'),
        skiprows=1, usecols=range(1, 6),
    )[:, REORDER]

    K = theta.shape[0]
    corr_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            r, _ = pearsonr(theta[i], chromhmm[j])
            corr_matrix[i, j] = r

    our_idx, chromhmm_idx = linear_sum_assignment(-corr_matrix)
    correlations = [corr_matrix[i, j] for i, j in zip(our_idx, chromhmm_idx)]

    print(f"{'='*65}")
    print(f"Emission comparison: Bernoulli model vs ChromHMM (seed 42)")
    print(f"Marks (canonical order): {MARKS}")
    print(f"{'='*65}\n")

    print(f"{'Our state':>10} {'ChromHMM state':>14} {'Pearson r':>10}")
    print('-' * 40)
    for i, j, r in zip(our_idx, chromhmm_idx, correlations):
        print(f" State {i+1:2d} →   E{j+1:<2d} {r:>10.4f}")

    print(f"\nMean r:   {np.mean(correlations):.4f}")
    print(f"Min r:    {np.min(correlations):.4f}")
    print(f"States with r > 0.9: {sum(r > 0.9 for r in correlations)}/10")
    print(f"States with r > 0.7: {sum(r > 0.7 for r in correlations)}/10")

    print(f"\n{'='*65}")
    print("Per-state emission profiles (after matching):")
    print(f"{'State':<8} {'Source':<12} " + " ".join(f"{m[:8]:>8}" for m in MARKS))
    print('-' * 65)
    for i, j, r in zip(our_idx, chromhmm_idx, correlations):
        print(f"S{i+1}→E{j+1:<2} {'ours':<12} " + " ".join(f"{theta[i, m]:>8.4f}" for m in range(5)))
        print(f"{'':8} {'ChromHMM':<12} " + " ".join(f"{chromhmm[j, m]:>8.4f}" for m in range(5)))
        print(f"{'':8} {'r=':<4}{r:.4f}")
        print()


if __name__ == '__main__':
    run()
