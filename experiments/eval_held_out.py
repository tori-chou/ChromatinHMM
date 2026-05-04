"""
Evaluate trained models on a held-out chromosome via log-likelihood.

Only Poisson and NB are compared — Bernoulli uses binarized data and
a different likelihood scale, so cross-model LL comparison is invalid.

Usage:
    python3 eval_held_out.py --chrom chr18
"""

import argparse
import time
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
from utils import load_counts, load_model, MODELS as MODELS_DIR

MODELS = ['poisson', 'nb']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chrom', default='chr18',
                        help='Held-out chromosome (must not be in training set)')
    args = parser.parse_args()

    print(f"Loading count data for {args.chrom}...")
    t0 = time.time()
    X = load_counts(args.chrom)
    print(f" {X.shape[0]} bins x {X.shape[1]} marks ({time.time()-t0:.1f}s)\n")

    results = {}
    for name in MODELS:
        npz = os.path.join(MODELS_DIR, f'{name}_model.npz')
        if not os.path.exists(npz):
            print(f" [{name}] no saved model found at {npz} — skipping")
            continue
        model, _ = load_model(name)

        log_emission = model.emission.log_prob(X)
        n_underflow = int(np.all(log_emission < -700, axis=1).sum())
        print(f" [{name}] underflow bins: {n_underflow} / {X.shape[0]} ({100*n_underflow/X.shape[0]:.1f}%)")

        t0 = time.time()
        ll = model.log_likelihood(X)
        ll_per_bin = ll / X.shape[0]
        print(f" [{name}] LL = {ll:>16.2f} ({ll_per_bin:.4f} per bin) ({time.time()-t0:.1f}s)\n")
        results[name] = ll

    if len(results) == 2:
        diff = results['nb'] - results['poisson']
        winner = 'NB' if diff > 0 else 'Poisson'
        print(f"\n NB − Poisson = {diff:+.2f} ({winner} fits {args.chrom} better)")

    out = os.path.join(ROOT, 'results', f'held_out_likelihood_{args.chrom}.txt')
    lines = [f'Held-out log-likelihood on {args.chrom}', f'Bins: {X.shape[0]} Marks: {X.shape[1]}', '']
    for name, ll in results.items():
        lines.append(f' {name:<12} LL = {ll:>18.2f} ({ll/X.shape[0]:.4f} per bin)')
    if len(results) == 2:
        lines.append(f'\n NB − Poisson = {diff:+.2f} ({winner} fits {args.chrom} better)')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nSaved to {out}')


if __name__ == '__main__':
    main()
