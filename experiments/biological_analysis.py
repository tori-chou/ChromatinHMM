"""
Overlap HMM state assignments with genomic annotations using bedtools.

For each state, reports the % of bins overlapping:
  - TSS ±2kb (proxy for promoter activity)
  - Gene bodies
  - CpG islands
  - cCRE categories

Aggregates across all training chromosomes.
Run prepare_annotations.py first to generate annotation BED files.
"""

import subprocess
import shutil
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
from utils import CHROMS_TRAIN, BIN_SIZE, K, load_model, load_binary, load_counts, load_chromhmm_states

ANNO_DIR = os.path.join(ROOT, 'annotations')
TSS_BED = os.path.join(ANNO_DIR, 'tss_2kb.bed')
GENE_BED = os.path.join(ANNO_DIR, 'gene_bodies.bed')
CPG_BED = os.path.join(ANNO_DIR, 'cpg_islands.bed')

ANNOTATIONS = [
    ('TSS ±2kb',        TSS_BED),
    ('Gene body',       GENE_BED),
    ('CpG islands',     CPG_BED),
    ('cCRE PLS',        os.path.join(ANNO_DIR, 'ccre_PLS.bed')),
    ('cCRE pELS',       os.path.join(ANNO_DIR, 'ccre_pELS.bed')),
    ('cCRE dELS',       os.path.join(ANNO_DIR, 'ccre_dELS.bed')),
    ('cCRE CTCF',       os.path.join(ANNO_DIR, 'ccre_CTCF.bed')),
    ('cCRE DNaseK4me3', os.path.join(ANNO_DIR, 'ccre_DNaseH3K4me3.bed')),
]

ANNOTATIONS = [(name, path) for name, path in ANNOTATIONS if os.path.exists(path)]


def _get_states(model, data, chrom, load_data):
    key = f'states_{chrom}'
    if key in data.files:
        return data[key]
    print(f' (no cached states for {chrom}, running Viterbi)')
    return model.predict(load_data(chrom))


def _count_overlapping_bp(state_bed, anno_bed):
    result = subprocess.run(
        ['bedtools', 'intersect', '-a', state_bed, '-b', anno_bed, '-wo'],
        capture_output=True, text=True,
    )
    return sum(int(line.split('\t')[-1])
               for line in result.stdout.strip().split('\n') if line)


def _annotation_bp(anno_bed, chroms):
    """Total merged bp in anno_bed restricted to given chromosomes."""
    result = subprocess.run(
        ['bedtools', 'merge', '-i', anno_bed],
        capture_output=True, text=True,
    )
    chrom_set = set(chroms)
    total = 0
    for line in result.stdout.strip().split('\n'):
        if not line:
            continue
        parts = line.split('\t')
        if parts[0] in chrom_set:
            total += int(parts[2]) - int(parts[1])
    return total


def run(model_name):
    out_dir = os.path.join(ROOT, 'results', 'biological_analysis')
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = os.path.join(out_dir, f'tmp_{model_name}')
    os.makedirs(tmp_dir, exist_ok=True)

    if model_name == 'chromhmm':
        get_states = lambda chrom: load_chromhmm_states(chrom)
    else:
        load_data = load_binary if model_name == 'bernoulli' else load_counts
        model, data = load_model(model_name)
        get_states = lambda chrom: _get_states(model, data, chrom, load_data)

    # Accumulate per-state BED entries and total bp across all chromosomes
    state_lines = {s: [] for s in range(K)}
    state_total_bp = {s: 0 for s in range(K)}

    for chrom in CHROMS_TRAIN:
        print(f' [{model_name}] {chrom}...')
        states = get_states(chrom)
        i, n = 0, len(states)
        while i < n:
            s = states[i]
            j = i + 1
            while j < n and states[j] == s:
                j += 1
            bp = (j - i) * BIN_SIZE
            state_lines[s].append(f'{chrom}\t{i*BIN_SIZE}\t{j*BIN_SIZE}\tState{s+1}\n')
            state_total_bp[s] += bp
            i = j

    # Write per-state BED files (multi-chromosome)
    state_beds = {}
    present_states = [s for s in range(K) if state_total_bp[s] > 0]
    for s in present_states:
        path = os.path.join(tmp_dir, f'state{s+1}.bed')
        with open(path, 'w') as f:
            f.writelines(state_lines[s])
        state_beds[s] = path

    # Background: annotation bp and genome bp (on training chroms only)
    genome_bp = sum(state_total_bp.values())
    print(f' Computing annotation sizes (genome bp = {genome_bp:,})...')
    anno_bp = [_annotation_bp(path, CHROMS_TRAIN) for _, path in ANNOTATIONS]

    # Compute overlaps and fold enrichment
    overlap_pct = np.zeros((K, len(ANNOTATIONS)))
    fold_enrich = np.zeros((K, len(ANNOTATIONS)))
    for s in present_states:
        for j, (_, anno_bed) in enumerate(ANNOTATIONS):
            bp = _count_overlapping_bp(state_beds[s], anno_bed)
            overlap_pct[s, j] = 100.0 * bp / state_total_bp[s]
            background = anno_bp[j] / genome_bp
            if background > 0:
                fold_enrich[s, j] = (bp / state_total_bp[s]) / background

    # Format text output
    chroms_str = ', '.join(CHROMS_TRAIN)
    col_w = 14

    def make_table(title, mat, fmt_str):
        hdr = f' {"State":<10} {"Total bp":>10} ' + ' '.join(f'{n:>{col_w}}' for n, _ in ANNOTATIONS)
        rows = [title, '', hdr, '-' * len(hdr)]
        for s in present_states:
            vals = ' '.join(fmt_str.format(mat[s, j]) for j in range(len(ANNOTATIONS)))
            rows.append(f' State{s+1:<5} {state_total_bp[s]:>10,} {vals}')
        return '\n'.join(rows)

    output = (
        make_table(f'Biological analysis: {model_name} on {chroms_str}\n% overlap',
                   overlap_pct, f'{{:>{col_w}.1f}}%') +
        '\n\n' +
        make_table(f'Fold enrichment over background '
                   f'(genome bp = {genome_bp:,})',
                   fold_enrich, f'{{:>{col_w}.2f}}x')
        + '\n'
    )
    print('\n' + output)

    base = os.path.join(out_dir, f'biological_analysis_{model_name}')
    with open(base + '.txt', 'w') as f:
        f.write(output)

    np.savez(base + '.npz',
             overlap_pct=overlap_pct,
             fold_enrichment=fold_enrich,
             anno_bp=np.array(anno_bp),
             genome_bp=np.array([genome_bp]),
             total_bp=np.array([state_total_bp[s] for s in range(K)]),
             state_labels=np.array([f'State {s+1}' for s in range(K)]),
             annotation_labels=np.array([name for name, _ in ANNOTATIONS]))
    print(f'Saved to {base}.txt and {base}.npz')

    shutil.rmtree(tmp_dir)


ALL_MODELS = ['bernoulli', 'poisson', 'nb', 'chromhmm']

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=ALL_MODELS + ['all'], default='all',
                        help='Model to run (default: all)')
    args = parser.parse_args()
    targets = ALL_MODELS if args.model == 'all' else [args.model]
    for name in targets:
        print(f'\n{"="*60}\nRunning {name}\n{"="*60}')
        run(name)
