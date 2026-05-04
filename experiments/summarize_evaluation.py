"""
Summarize biological quality of each HMM model with four quantitative metrics:

  1. Roadmap ARI — ARI between model states and Roadmap 15-state annotation (computed from the saved contingency matrix)
  2. Mean state purity — average of the max Roadmap % per state row; measures how cleanly each state maps to a single known chromatin class
  3. TPM dynamic range — max median TPM / mean median TPM across states with ≥5 genes; captures how well the model separates expressed from silent genes
  4. Promoter states — number of states with cCRE PLS overlap > 5%
                    
ChromHMM is included as a reference ceiling.
"""
import os
import sys
import numpy as np
from math import comb

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

ROADMAP_DIR = os.path.join(ROOT, 'results', 'roadmap_comparison')
EXPR_DIR = os.path.join(ROOT, 'results', 'expression_correlation')
BIO_DIR = os.path.join(ROOT, 'results', 'biological_analysis')

MODELS = ['bernoulli', 'poisson', 'nb', 'chromhmm']


def ari_from_contingency(C):
    """Compute Adjusted Rand Index from a contingency matrix."""
    n = int(C.sum())
    if n < 2:
        return float('nan')

    def sum_comb(arr):
        return sum(int(x) * (int(x) - 1) // 2 for x in arr.flatten() if x >= 2)

    sum_C = sum_comb(C)
    sum_row = sum_comb(C.sum(axis=1))
    sum_col = sum_comb(C.sum(axis=0))
    total = n * (n - 1) // 2

    expected = sum_row * sum_col / total
    maximum = (sum_row + sum_col) / 2

    if maximum == expected:
        return 1.0
    return (sum_C - expected) / (maximum - expected)


def roadmap_ari(model_name):
    path = os.path.join(ROADMAP_DIR, f'roadmap_confusion_{model_name}.npz')
    C = np.load(path)['counts'] # (K, 15)
    return ari_from_contingency(C)


def mean_state_purity(model_name):
    path = os.path.join(ROADMAP_DIR, f'roadmap_confusion_{model_name}.npz')
    pct = np.load(path)['pct'] # (K, 15) row-normalised %
    # Only include states that actually have bins
    C = np.load(path)['counts']
    active = C.sum(axis=1) > 0
    return float(pct[active].max(axis=1).mean())


def tpm_dynamic_range(model_name):
    path = os.path.join(EXPR_DIR, f'expression_correlation_{model_name}.npz')
    data = np.load(path, allow_pickle=True)
    medians = data['medians']
    n_genes = data['n_genes']
    valid = medians[n_genes >= 5]
    if len(valid) == 0 or valid.mean() == 0:
        return float('nan')
    return float(valid.max() / valid.mean())


def promoter_states(model_name):
    path = os.path.join(BIO_DIR, f'biological_analysis_{model_name}.npz')
    data = np.load(path, allow_pickle=True)
    labels = list(data['annotation_labels'])
    if 'cCRE PLS' not in labels:
        return float('nan')
    pls_col = labels.index('cCRE PLS')
    fe = data['fold_enrichment'][:, pls_col]
    return int((fe > 5.0).sum())


def main():
    metrics = {m: {} for m in MODELS}
    for m in MODELS:
        metrics[m]['Roadmap ARI'] = roadmap_ari(m)
        metrics[m]['Mean purity (%)'] = mean_state_purity(m)
        metrics[m]['TPM dyn. range'] = tpm_dynamic_range(m)
        metrics[m]['Promoter states'] = promoter_states(m)

    col_w = 18
    header = f" {'Metric':<22}" + ''.join(f' {m:>{col_w}}' for m in MODELS)
    sep = '-' * len(header)

    rows = [
        ('Roadmap ARI',      '{:.4f}',  lambda m: metrics[m]['Roadmap ARI']),
        ('Mean purity (%)',  '{:.1f}',  lambda m: metrics[m]['Mean purity (%)']),
        ('TPM dyn. range',  '{:.1f}x', lambda m: metrics[m]['TPM dyn. range']),
        ('Promoter states', '{:d}',    lambda m: int(metrics[m]['Promoter states'])),
    ]

    print('\nBiological quality summary\n')
    print(header)
    print(sep)
    for label, fmt, fn in rows:
        vals = []
        for m in MODELS:
            v = fn(m)
            try:
                vals.append(fmt.format(v))
            except (ValueError, TypeError):
                vals.append('n/a')
        print(f" {label:<22}" + ''.join(f' {v:>{col_w}}' for v in vals))
    print()

    # Save
    out = os.path.join(ROOT, 'results', 'evaluation_summary.txt')
    lines = ['Biological quality summary\n', header, sep]
    for label, fmt, fn in rows:
        vals = []
        for m in MODELS:
            v = fn(m)
            try:
                vals.append(fmt.format(v))
            except (ValueError, TypeError):
                vals.append('n/a')
        lines.append(f" {label:<22}" + ''.join(f' {v:>{col_w}}' for v in vals))
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Saved to {out}')


if __name__ == '__main__':
    main()
