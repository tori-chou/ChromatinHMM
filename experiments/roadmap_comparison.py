"""
Compare HMM states to Roadmap Epigenomics 15-state model for GM12878 (E116).
Runs Viterbi on all training chromosomes and reports per-chromosome and
aggregate confusion matrices.

Roadmap 15-state mnemonic labels:
  1_TssA Active TSS
  2_TssAFlnk Flanking Active TSS
  3_TxFlnk Transcr. at gene 5' and 3'
  4_Tx Strong transcription
  5_TxWk Weak transcription
  6_EnhG Genic enhancers
  7_Enh Enhancers
  8_ZNF/Rpts ZNF genes & repeats
  9_Het Heterochromatin
  10_TssBiv Bivalent/Poised TSS
  11_BivFlnk Flanking Bivalent TSS/Enh
  12_EnhBiv Bivalent Enhancer
  13_ReprPC Repressed PolyComb
  14_ReprPCWk Weak Repressed PolyComb
  15_Quies Quiescent/Low
"""

import os
import sys
import gzip
import urllib.request
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
from utils import BIN_SIZE, CHROMS_TRAIN, K, load_model, load_binary, load_counts, load_chromhmm_states

def _get_states(model, data, chrom, load_data):
    key = f'states_{chrom}'
    if key in data.files:
        return data[key]
    print(f' (no cached states for {chrom}, running Viterbi)')
    return model.predict(load_data(chrom))

ROADMAP_URL = (
    'https://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/'
    'ChmmModels/coreMarks/jointModel/final/E116_15_coreMarks_hg38lift_mnemonics.bed.gz'
)
ANNO_DIR = os.path.join(ROOT, 'annotations')
BED_GZ = os.path.join(ANNO_DIR, 'E116_15_coreMarks_hg38lift_mnemonics.bed.gz')

ROADMAP_LABELS = [
    '1_TssA', '2_TssAFlnk', '3_TxFlnk', '4_Tx', '5_TxWk',
    '6_EnhG', '7_Enh', '8_ZNF/Rpts', '9_Het', '10_TssBiv',
    '11_BivFlnk', '12_EnhBiv', '13_ReprPC', '14_ReprPCWk', '15_Quies',
]
LABEL_TO_IDX = {l: i for i, l in enumerate(ROADMAP_LABELS)}
N_ROADMAP = len(ROADMAP_LABELS)


def _download():
    os.makedirs(ANNO_DIR, exist_ok=True)
    if os.path.exists(BED_GZ):
        return
    print(f' Downloading {os.path.basename(BED_GZ)}...')
    urllib.request.urlretrieve(ROADMAP_URL, BED_GZ)
    print(' Done.')


def _load_roadmap_states(chrom):
    bins = {}
    with gzip.open(BED_GZ, 'rt') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if parts[0] != chrom:
                continue
            label = parts[3]
            if label not in LABEL_TO_IDX:
                continue
            idx = LABEL_TO_IDX[label]
            for b in range(int(parts[1]) // BIN_SIZE, int(parts[2]) // BIN_SIZE):
                bins[b] = idx
    if not bins:
        raise ValueError(f'No Roadmap segments found for {chrom}')
    arr = np.full(max(bins) + 1, fill_value=-1, dtype=np.int32)
    for b, s in bins.items():
        arr[b] = s
    return arr


def run(model_name):
    _download()

    if model_name == 'chromhmm':
        get_states = lambda chrom: load_chromhmm_states(chrom)
    else:
        load_data = load_binary if model_name == 'bernoulli' else load_counts
        model, data = load_model(model_name)
        get_states = lambda chrom: _get_states(model, data, chrom, load_data)

    conf = np.zeros((K, N_ROADMAP), dtype=np.int64)

    for chrom in CHROMS_TRAIN:
        print(f' [{model_name}] {chrom}...')
        ours = get_states(chrom)
        ref = _load_roadmap_states(chrom)
        n = min(len(ours), len(ref))
        mask = ref[:n] >= 0 # exclude bins not covered by the Roadmap liftover
        conf += np.bincount(
            np.ravel_multi_index((ours[:n][mask], ref[:n][mask]), (K, N_ROADMAP)),
            minlength=K * N_ROADMAP,
        ).reshape(K, N_ROADMAP)

    row_totals = conf.sum(axis=1, keepdims=True)
    conf_pct = np.where(row_totals > 0, 100.0 * conf / row_totals, 0.0)

    header = ('Aggregate confusion matrix: '
              f'{model_name} vs Roadmap E116 ({", ".join(CHROMS_TRAIN)})\n'
              'Values = % of our state bins assigned to each Roadmap state\n\n'
              ' ' + ''.join(f' {l[:8]:>8}' for l in ROADMAP_LABELS))
    rows = []
    for i in range(K):
        if conf[i].sum() == 0:
            continue
        rows.append(' State{:2d} {}'.format(
            i + 1, ''.join(f' {conf_pct[i, j]:>7.1f}%' for j in range(N_ROADMAP))
        ))
    table = header + '\n' + '\n'.join(rows) + '\n'

    print('\n' + table)

    base = os.path.join(ROOT, 'results', 'roadmap_comparison', f'roadmap_confusion_{model_name}')
    np.savez(base + '.npz',
             counts=conf,
             pct=conf_pct,
             hmm_state_labels=np.array([f'State {i+1}' for i in range(K)]),
             roadmap_labels=np.array(ROADMAP_LABELS))
    with open(base + '.txt', 'w') as f:
        f.write(table)
    print(f'Saved to {base}.npz and {base}.txt')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['bernoulli', 'poisson', 'nb', 'chromhmm'], required=True)
    args = parser.parse_args()
    run(args.model)