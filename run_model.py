"""
Train and evaluate a ChromatinHMM model with a given emission distribution.

Usage — single seed:
    python3 run_model.py --model bernoulli --seed 42

Usage — pick best over multiple seeds:
    python3 run_model.py --model bernoulli --seeds 0 1 2 3
    Saves the best (highest final training LL) as {model}_model.npz.
"""

import argparse
import time
import os
import numpy as np
from sklearn.metrics import adjusted_rand_score
from utils import (
    load_binary, load_counts, load_chromhmm_states, build_model,
    CHROMS_TRAIN, CHROM_EVAL, MARKS, MODELS
)


def _train_one(model_name, train, ref, seed):
    """Train a single model and return results dict."""
    print(f"\nSeed {seed} — building {model_name} model...")
    model = build_model(model_name, len(MARKS), seed=seed)

    t0 = time.time()
    ll_history, converged = model.fit(train)
    print(f" Training complete in {time.time()-t0:.1f}s ({len(ll_history)} iterations)")

    states_all = [model.predict(X) for X in train]
    eval_states = states_all[CHROMS_TRAIN.index(CHROM_EVAL)]
    n = min(len(eval_states), len(ref))
    ari = adjusted_rand_score(ref[:n], eval_states[:n])
    status = 'converged' if converged else 'DID NOT CONVERGE'
    print(f" Final LL: {ll_history[-1]:.2f} ARI: {ari:.4f} [{status}]")

    return dict(model=model, ll_history=ll_history, converged=converged,
                states_all=states_all, ari=ari, seed=seed)


def _save(model_name, result, path):
    save_kwargs = dict(
        log_A = result['model'].log_A,
        log_pi = result['model'].log_pi,
        ll_history = np.array(result['ll_history']),
        ari = np.array(result['ari']),
        seed = np.array(result['seed']),
    )
    for chrom, states in zip(CHROMS_TRAIN, result['states_all']):
        save_kwargs[f'states_{chrom}'] = states
    if model_name == 'bernoulli':
        save_kwargs['theta'] = result['model'].emission.theta
    elif model_name == 'poisson':
        save_kwargs['lam'] = result['model'].emission.lam
    elif model_name == 'nb':
        save_kwargs['mu'] = result['model'].emission.mu
        save_kwargs['r'] = result['model'].emission.r
    np.savez(path, **save_kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['bernoulli', 'poisson', 'nb'], required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--seed',  type=int, help='Train a single seed')
    group.add_argument('--seeds', type=int, nargs='+', help='Train multiple seeds, save the best')
    args = parser.parse_args()

    load_data = load_binary if args.model == 'bernoulli' else load_counts

    print(f"Loading {'binary' if args.model == 'bernoulli' else 'count'} data...")
    t0 = time.time()
    train = [load_data(c) for c in CHROMS_TRAIN]
    for c, X in zip(CHROMS_TRAIN, train):
        print(f" {c}: {X.shape[0]} bins x {X.shape[1]} marks")
    print(f" Loaded in {time.time()-t0:.1f}s")

    ref = load_chromhmm_states(CHROM_EVAL)

    if args.seed is not None:
        result = _train_one(args.model, train, ref, args.seed)
        if not result['converged']:
            print(f"\n WARNING: model did not converge — consider increasing n_iter.")
        out = os.path.join(MODELS, f'{args.model}_seed{args.seed}_model.npz')
        _save(args.model, result, out)
        print(f"\nModel saved to {out}")

    else:
        results = [_train_one(args.model, train, ref, s) for s in args.seeds]

        converged = [r for r in results if r['converged']]
        pool = converged if converged else results
        best = max(pool, key=lambda r: r['ll_history'][-1])

        print(f"\n{'='*50}")
        print(f" Seed comparison for {args.model}:")
        for r in results:
            marker = ' ← best' if r['seed'] == best['seed'] else ''
            status = 'converged' if r['converged'] else 'did not converge'
            print(f" seed {r['seed']:3d}:  LL={r['ll_history'][-1]:>14.2f} "
                  f"ARI={r['ari']:.4f} [{status}]{marker}")
        if not converged:
            print(f"\n WARNING: no seeds converged — increase --n_iter and rerun.")
        print(f"{'='*50}\n")

        seed_out = os.path.join(MODELS, f'{args.model}_seed{best["seed"]}_model.npz')
        canon = os.path.join(MODELS, f'{args.model}_model.npz')
        _save(args.model, best, seed_out)
        _save(args.model, best, canon)
        print(f"Best model (seed {best['seed']}) saved to {seed_out} and {canon}")


if __name__ == '__main__':
    main()
