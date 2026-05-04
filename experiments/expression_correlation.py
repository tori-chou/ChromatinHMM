"""
Correlate HMM state assignments with gene expression (TPM) for GM12878.

For each protein-coding gene on each training chromosome:
  - Find which state its TSS falls in (Viterbi decoding)
  - Collect TPM from RNA-seq

Aggregates across all training chromosomes, then reports median TPM per
state and runs Kruskal-Wallis and Mann-Whitney tests.
"""
import os
import sys
import gzip
import numpy as np
from scipy.stats import kruskal, mannwhitneyu

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
from utils import BIN_SIZE, CHROMS_TRAIN, K, load_model, load_binary, load_counts, load_chromhmm_states

def _get_states(model, data, chrom, load_data):
    key = f'states_{chrom}'
    if key in data.files:
        return data[key]
    print(f' (no cached states for {chrom}, running Viterbi)')
    return model.predict(load_data(chrom))

ANNO_GTF = os.path.join(ROOT, 'annotations', 'gencode.v45.basic.gtf.gz')
RNA_TSV = os.path.join(ROOT, 'data', 'gm12878_rnaseq_gene_quant.tsv')


def _load_tpm():
    tpm = {}
    with open(RNA_TSV) as f:
        header = f.readline().split('\t')
        tpm_col = header.index('TPM')
        for line in f:
            parts = line.rstrip('\n').split('\t')
            gene_id = parts[0].split('.')[0]
            tpm[gene_id] = float(parts[tpm_col])
    return tpm


def _load_genes(chrom):
    genes = []
    with gzip.open(ANNO_GTF, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if fields[0] != chrom or fields[2] != 'gene':
                continue
            if 'gene_type "protein_coding"' not in fields[8]:
                continue
            start = int(fields[3]) - 1
            strand = fields[6]
            end = int(fields[4])
            gene_id = ''
            for attr in fields[8].split(';'):
                attr = attr.strip()
                if attr.startswith('gene_id'):
                    gene_id = attr.split('"')[1].split('.')[0]
                    break
            tss = start if strand == '+' else end - 1
            genes.append((gene_id, tss))
    return genes


def run(model_name):
    print('Loading RNA-seq TPM...')
    tpm = _load_tpm()
    print(f' {len(tpm)} genes loaded')

    if model_name == 'chromhmm':
        get_states = lambda chrom: load_chromhmm_states(chrom)
    else:
        load_data = load_binary if model_name == 'bernoulli' else load_counts
        model, data = load_model(model_name)
        get_states = lambda chrom: _get_states(model, data, chrom, load_data)

    state_tpms = {}
    total_genes = total_skipped = 0

    for chrom in CHROMS_TRAIN:
        print(f' [{model_name}] {chrom}...')
        states = get_states(chrom)
        genes = _load_genes(chrom)

        skipped = 0
        for gene_id, tss in genes:
            if gene_id not in tpm:
                skipped += 1
                continue
            bin_idx = tss // BIN_SIZE
            if bin_idx >= len(states):
                skipped += 1
                continue
            state_tpms.setdefault(int(states[bin_idx]), []).append(tpm[gene_id])
        total_genes += len(genes)
        total_skipped += skipped
        print(f' {len(genes)} genes, {skipped} skipped')

    print(f'\n Total: {total_genes} genes, {total_skipped} skipped')

    present = sorted(state_tpms.keys())
    lines = [
        f'Gene expression (TPM) by HMM state — TSS assignment on {", ".join(CHROMS_TRAIN)} [model: {model_name}]',
        f'RNA-seq: GM12878 total RNA-seq GRCh38 (ENCFF910XWA)\n',
        f' {"State":<10} {"N genes":>7} {"Median TPM":>10} {"Mean TPM":>10} {"% expressed":>12}',
        '-' * 58,
    ]
    for s in present:
        vals = np.array(state_tpms[s])
        pct_expr = 100 * np.mean(vals > 0)
        lines.append(
            f' State{s+1:<5} {len(vals):>7} '
            f'{np.median(vals):>10.2f} {np.mean(vals):>10.2f} {pct_expr:>11.1f}%'
        )

    groups = [np.array(state_tpms[s]) for s in present if len(state_tpms[s]) > 1]
    if len(groups) >= 2:
        stat, p = kruskal(*groups)
        lines.append(f'\nKruskal-Wallis test across states: H={stat:.2f}, p={p:.2e}')

    medians = {s: np.median(v) for s, v in state_tpms.items() if len(v) > 1}
    if len(medians) >= 2:
        hi = max(medians, key=medians.get)
        lo = min(medians, key=medians.get)
        u, p = mannwhitneyu(state_tpms[hi], state_tpms[lo], alternative='greater')
        lines.append(
            f'Mann-Whitney (State{hi+1} median={medians[hi]:.2f} > '
            f'State{lo+1} median={medians[lo]:.2f}): U={u:.0f}, p={p:.2e}'
        )

    output = '\n'.join(lines) + '\n'
    print('\n' + output)

    base = os.path.join(ROOT, 'results', 'expression_correlation', f'expression_correlation_{model_name}')

    with open(base + '.txt', 'w') as f:
        f.write(output)

    save_kwargs = {f'tpm_state{s}': np.array(state_tpms[s]) for s in present}
    save_kwargs['state_labels'] = np.array([f'State {s+1}' for s in present])
    save_kwargs['medians'] = np.array([np.median(state_tpms[s]) for s in present])
    save_kwargs['means'] = np.array([np.mean(state_tpms[s]) for s in present])
    save_kwargs['n_genes'] = np.array([len(state_tpms[s]) for s in present])
    np.savez(base + '.npz', **save_kwargs)

    print(f'Saved to {base}.txt and {base}.npz')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['bernoulli', 'poisson', 'nb', 'chromhmm'], required=True)
    args = parser.parse_args()
    run(args.model)