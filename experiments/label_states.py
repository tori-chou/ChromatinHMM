"""
Assign biological labels to HMM states using the dominant Roadmap E116 15-state class.

Labels are derived solely from the Roadmap confusion matrix.

Saves results/state_labels.json and prints assignments.
Importable: get_state_labels(model_name) -> dict {state_idx (0-based): label}
"""

import os
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
ROADMAP_DIR = os.path.join(ROOT, 'results', 'roadmap_comparison')
OUT_PATH = os.path.join(ROOT, 'results', 'state_labels.json')

MODELS = ['bernoulli', 'poisson', 'nb', 'chromhmm']

ROADMAP_TO_LABEL = {
    '1_TssA':      'Active TSS',
    '2_TssAFlnk':  'Flanking TSS',
    '3_TxFlnk':    'Tx Flanking',
    '4_Tx':        'Transcribed',
    '5_TxWk':      'Weak Transcription',
    '6_EnhG':      'Genic Enhancer',
    '7_Enh':       'Enhancer',
    '8_ZNF/Rpts':  'ZNF/Repeats',
    '9_Het':       'Heterochromatin',
    '10_TssBiv':   'Bivalent TSS',
    '11_BivFlnk':  'Bivalent Flanking',
    '12_EnhBiv':   'Bivalent Enhancer',
    '13_ReprPC':   'Polycomb',
    '14_ReprPCWk': 'Weak Polycomb',
    '15_Quies':    'Quiescent',
}


def _deduplicate(labels_dict):
    """Append I, II, III... to duplicate labels within one model."""
    from collections import Counter
    counts = Counter(labels_dict.values())
    seen = {}
    result = {}
    for idx, lbl in sorted(labels_dict.items()):
        if counts[lbl] > 1:
            seen[lbl] = seen.get(lbl, 0) + 1
            numerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']
            result[idx] = f'{lbl} {numerals[seen[lbl] - 1]}'
        else:
            result[idx] = lbl
    return result


def compute_labels(model_name):
    """Return {state_idx (0-based): label} for active states of model_name."""
    rm = np.load(os.path.join(ROADMAP_DIR, f'roadmap_confusion_{model_name}.npz'),
                 allow_pickle=True)

    counts = rm['counts'].astype(float) # (K, 15) raw bin counts
    roadmap_lbl = [str(s) for s in rm['roadmap_labels']]

    raw = {}
    for s in range(counts.shape[0]):
        if counts[s].sum() == 0:
            continue
        dom = roadmap_lbl[int(counts[s].argmax())]
        raw[s] = ROADMAP_TO_LABEL.get(dom, dom)

    return _deduplicate(raw)


def get_state_labels(model_name):
    """Load precomputed labels from JSON. Falls back to computing if file missing."""
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            all_labels = json.load(f)
        if model_name in all_labels:
            return {int(k): v for k, v in all_labels[model_name].items()}
    return compute_labels(model_name)


def main():
    all_labels = {}
    for model in MODELS:
        labels = compute_labels(model)
        all_labels[model] = {str(k): v for k, v in labels.items()}

        print(f'\n{model}')
        for idx, lbl in sorted(labels.items()):
            print(f' State {idx + 1:2d} →  {lbl}')

    with open(OUT_PATH, 'w') as f:
        json.dump(all_labels, f, indent=2)
    print(f'\nSaved to {OUT_PATH}')


if __name__ == '__main__':
    main()
