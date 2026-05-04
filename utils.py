"""
Shared utilities for the chromatin state HMM project.
Provides data loaders, constants, and model factory/loader functions
used by run_model.py, run_analysis.py, and the experiment scripts.
"""

import os
import numpy as np
import pysam

from chromatin_hmm import ChromatinHMM, BernoulliEmission, PoissonEmission, NBEmission

# Constants
# Only needed if regenerating count matrices from BAM files.
# Set the CHROMATIN_PROJECT_DIR environment variable to the directory containing bam_files/ on your machine, e.g.:
#   export CHROMATIN_PROJECT_DIR="/path/to/chromatin_project"
DRIVE = os.environ.get('CHROMATIN_PROJECT_DIR', '')

CHROMS_TRAIN = ['chr17', 'chr19', 'chr21', 'chr22']
CHROM_EVAL = 'chr21'
CHROM_HELD = 'chr18'
K = 10
BIN_SIZE = 200
MARKS = ['H3K27me3', 'H3K36me3', 'H3K4me1', 'H3K4me3', 'H3K9me3']
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
MODELS = 'models'

def load_binary(chrom: str) -> np.ndarray:
    """Load ChromHMM binarized data. Returns float (T, M) array."""
    path = os.path.join('data', f'GM12878_{chrom}_binary.txt')
    return np.loadtxt(path, skiprows=2, dtype=float)


def _extract_counts_bam(chrom: str) -> np.ndarray:
    """Extract per-bin read counts from BAM files."""
    bam_dir = os.path.join(DRIVE, 'bam_files')
    cols = []
    for mark in MARKS:
        bam_path = os.path.join(bam_dir, f'{mark}.bam')
        bam = pysam.AlignmentFile(bam_path, 'rb')
        chrom_len = dict(zip(bam.references, bam.lengths))[chrom]
        n_bins = chrom_len // BIN_SIZE
        counts = np.zeros(n_bins, dtype=np.int32)
        for read in bam.fetch(chrom):
            if read.is_unmapped or read.is_duplicate:
                continue
            idx = read.reference_start // BIN_SIZE
            if idx < n_bins:
                counts[idx] += 1
        try:
            bam.close()
        except Exception:
            pass
        print(f" {mark}: {n_bins} bins, mean={counts.mean():.2f}, max={counts.max()}")
        cols.append(counts)
    return np.stack(cols, axis=1).astype(float)


def load_counts(chrom: str) -> np.ndarray:
    """
    Load count matrix for chrom from .npy cache.
    If cache is missing, extract from BAM files and save it.
    Returns float (T, M) array with marks in MARKS order.
    """
    npy_path = os.path.join('data', f'count_matrix_{chrom}.npy')
    if os.path.exists(npy_path):
        return np.load(npy_path).astype(float)

    print(f"Cache miss — extracting counts for {chrom} from BAM files...")
    arr = _extract_counts_bam(chrom)
    np.save(npy_path, arr)
    print(f" Saved to {npy_path}")
    return arr.astype(float)


def load_chromhmm_states(chrom: str) -> np.ndarray:
    """
    Parse ChromHMM segments.bed and return a per-bin 0-indexed state array.
    State label format in file: 'E1' ... 'E10'.
    """
    bed_path = os.path.join('data', 'GM12878_segments.bed')
    segments = []
    with open(bed_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4 or parts[0] != chrom:
                continue
            segments.append((int(parts[1]) // BIN_SIZE, int(parts[2]) // BIN_SIZE,
                             int(parts[3].lstrip('E')) - 1))
    if not segments:
        raise ValueError(f"No segments found for {chrom} in {bed_path}")
    arr = np.zeros(max(b_end for _, b_end, _ in segments), dtype=np.int32)
    for b_start, b_end, state in segments:
        arr[b_start:b_end] = state
    return arr


def load_model(name: str):
    """Load a saved model from an NPZ file. Returns (model, data_dict)."""
    npz_path = os.path.join(MODELS, f'{name}_model.npz')
    data = np.load(npz_path)
    rng = np.random.default_rng(42)
    M = len(MARKS)
    if name == 'bernoulli':
        emission = BernoulliEmission(K=K, M=M, rng=rng)
        emission.theta = data['theta']
    elif name == 'poisson':
        emission = PoissonEmission(K=K, M=M, rng=rng)
        emission.lam = data['lam']
    elif name == 'nb':
        emission = NBEmission(K=K, M=M, rng=rng)
        emission.mu = data['mu']
        emission.r = data['r']
    else:
        raise ValueError(f'Unknown model: {name}')
    model = ChromatinHMM(K=K, emission=emission, seed=42)
    model.log_A = data['log_A']
    model.log_pi = data['log_pi']
    return model, data


def build_model(name: str, M: int, seed: int = 42) -> ChromatinHMM:
    """Construct a ChromatinHMM with the named emission distribution."""
    rng = np.random.default_rng(seed)
    if name == 'bernoulli':
        emission = BernoulliEmission(K=K, M=M, rng=rng)
    elif name == 'poisson':
        emission = PoissonEmission(K=K, M=M, rng=rng)
    elif name == 'nb':
        emission = NBEmission(K=K, M=M, rng=rng)
    else:
        raise ValueError(f"Unknown emission: {name}")
    return ChromatinHMM(K=K, emission=emission, n_iter=2000, seed=seed)