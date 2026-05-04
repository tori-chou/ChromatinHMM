# ChromatinHMM

A controlled comparison of Bernoulli, Poisson, and Negative Binomial emission distributions in a Hidden Markov Model for chromatin state discovery. All three models share the same architecture: K=10 states, Baum-Welch EM training, Viterbi decoding.

Applied to GM12878 (human lymphoblastoid cell line) ChIP-seq data for five histone marks:  H3K27me3, H3K36me3, H3K4me1, H3K4me3, H3K9me3.

---

## Repository structure

```
ChromatinHMM/
├── chromatin_hmm.py          # HMM: emission classes, forward-backward, Viterbi
├── utils.py                  # Shared constants, data loaders, model 
├── run_model.py              # CLI: train a model
├── run_analysis.py           # CLI: run all analyses and generate all figures
├── requirements.txt
│
├── data/
│   ├── GM12878_chr{17,19,21,22}_binary.txt   # Binarized ChIP-seq (ChromHMM format)
│   ├── GM12878_chr18_binary.txt              # Held-out chromosome (not used in training)
│   ├── count_matrix_chr{17,18,19,21,22}.npy  # Raw read counts (bins × marks)
│   ├── GM12878_segments.bed                  # ChromHMM reference segmentation (K=10)
│   └── gm12878_rnaseq_gene_quant.tsv         # RNA-seq gene quantification (TPM)
│
├── annotations/
│   ├── tss_2kb.bed              # TSS ±2 kb windows
│   ├── gene_bodies.bed          # Protein-coding gene bodies
│   ├── cpg_islands.bed          # CpG islands (UCSC)
│   └── ccre_*.bed               # ENCODE cCRE subsets (PLS, pELS, dELS, CTCF, DNaseH3K4me3)
│
├── models/
│   ├── bernoulli_model.npz      # Trained Bernoulli model parameters + Viterbi states
│   ├── poisson_model.npz        # Trained Poisson model
│   └── nb_model.npz             # Trained Negative Binomial model
│
├── experiments/
│   ├── prepare_annotations.py       # One-time setup: build annotation BED files
│   ├── confusion_matrix.py          # State assignment vs ChromHMM reference
│   ├── biological_analysis.py       # Annotation overlaps + fold enrichment (bedtools)
│   ├── roadmap_comparison.py        # Comparison vs Roadmap E116 15-state model
│   ├── expression_correlation.py    # TPM by state using RNA-seq
│   ├── eval_held_out.py             # Held-out log-likelihood on chr18
│   ├── compare_emissions.py         # Bernoulli emission correlation vs ChromHMM
│   ├── label_states.py              # Assign biological labels from Roadmap dominant class
│   ├── summarize_evaluation.py      # Summary metrics table (ARI, purity, TPM range)
│   └── plots/
│       ├── plot_emission_heatmaps.py
│       ├── plot_fold_enrichment_heatmaps.py
│       ├── plot_roadmap_heatmaps.py
│       ├── plot_tpm_violins.py
│       └── plot_summary_barchart.py
│
├── results/                     # Generated outputs (text summaries + figures)
│   ├── evaluation_summary.txt
│   ├── held_out_likelihood_chr18.txt
│   ├── state_labels.json
│   ├── figures/                 # PNG figures (all models)
│   ├── biological_analysis/
│   ├── expression_correlation/
│   └── roadmap_comparison/
│
└── notebooks/
    └── ChromHMM.ipynb           # Exploratory notebook
```

---

## Large input data (not in this repository)

The following files are too large to include. Download instructions:

### ChIP-seq BAM files (GM12878, GRCh38)

Required only if re-extracting count matrices from scratch. The pre-computed `.npy` count matrices in `data/` can be used directly.

Available from the [ENCODE portal](https://www.encodeproject.org). Search for cell line **GM12878**, genome **GRCh38**, assay **ChIP-seq**, with the following targets: H3K27me3, H3K36me3, H3K4me1, H3K4me3, H3K9me3. Download the filtered alignments (BAM + BAI) for each mark and place them in a directory:

```
/your/path/to/chromatin_project/bam_files/H3K27me3.bam   (and .bai)
/your/path/to/chromatin_project/bam_files/H3K36me3.bam
/your/path/to/chromatin_project/bam_files/H3K4me1.bam
/your/path/to/chromatin_project/bam_files/H3K4me3.bam
/your/path/to/chromatin_project/bam_files/H3K9me3.bam
```

Then set the environment variable before running:

```bash
export CHROMATIN_PROJECT_DIR="/your/path/to/chromatin_project"
```

### GENCODE annotation

Required only for `prepare_annotations.py` (one-time setup).

- File: `gencode.v45.basic.gtf.gz`
- Source: [GENCODE](https://www.gencodegenes.org/human/release_45.html) → Release 45 → Basic gene annotation (GRCh38)
- Place at: `annotations/gencode.v45.basic.gtf.gz`

### RNA-seq gene quantification

Already included in `data/gm12878_rnaseq_gene_quant.tsv`. If missing, download from ENCODE accession **ENCFF910XWA** (GM12878 total RNA-seq, GRCh38).

---

## System requirements

- **Python**: 3.10 or later (developed on 3.13.2)
- **OS**: macOS or Linux (pysam and pybedtools do not support Windows natively)
- **External tool**: `bedtools` must be on your PATH (used by `biological_analysis.py`)
- **Memory**: 8 GB RAM recommended
- **Runtime** (Apple M-series or comparable laptop CPU):
  - Bernoulli/Poisson training: ~10–20 min per seed
  - NB training: ~20 min per seed 

---

## Installation

```bash
# Clone the repo
git clone <repo-url>
cd ChromatinHMM

# Install Python dependencies
pip install -r requirements.txt

# Install bedtools (required for biological_analysis.py)
brew install bedtools        # macOS
# sudo apt install bedtools  # Debian/Ubuntu
```

---

## Quick start — reproduce results from saved models

Trained models are included in `models/`. To regenerate all analysis outputs and figures:

```bash
python3 run_analysis.py --model all
```

Outputs are written to `results/`. To evaluate held-out generalization:

```bash
python3 experiments/eval_held_out.py --chrom chr18
```

---

## Full pipeline: training from scratch

### Step 1: Prepare annotation files (one-time)

Requires `annotations/gencode.v45.basic.gtf.gz` and internet access (downloads CpG islands and cCREs automatically):

```bash
python3 experiments/prepare_annotations.py
```

### Step 2: Train models

```bash
python3 run_model.py --model bernoulli --seed 42
python3 run_model.py --model poisson   --seed 42
python3 run_model.py --model nb        --seed 42
```

To train multiple seeds and automatically keep the best:

```bash
python3 run_model.py --model nb --seeds 0 1 2 3
```

Trained models are saved to `models/{model}_model.npz`.

### Step 3: Run all analyses and generate figures

```bash
python3 run_analysis.py --model all
```

Or for a single model:

```bash
python3 run_analysis.py --model nb
```

This runs, in order: confusion matrix → biological annotation → Roadmap comparison → expression correlation → state labeling → all figures → summary table.

### Step 4: Evaluate held-out LL

```bash
python3 experiments/eval_held_out.py --chrom chr18
```

---

## Key parameters

| Parameter | Value | Defined in |
|-----------|-------|------------|
| Number of states (K) | 10 | `utils.py` |
| Bin size | 200 bp | `utils.py` |
| Training chromosomes | chr17, chr19, chr21, chr22 | `utils.py` |
| ARI evaluation chromosome | chr21 (vs ChromHMM) | `utils.py` |
| Held-out chromosome | chr18 | `utils.py` |
| Random seed | 42 | `run_model.py` |
| EM convergence tolerance | ΔLL < 0.1 | `chromatin_hmm.py` |
| Max EM iterations | 2000 | `utils.py` → `build_model` |
| NB dispersion bounds | (0.1, 100) | `chromatin_hmm.py` |

---

## Sample output

`results/evaluation_summary.txt` — biological quality metrics across all models:

```
  Metric                    bernoulli           poisson                nb          chromhmm
  Roadmap ARI                  0.XXXX            0.XXXX            0.XXXX            0.XXXX
  Mean purity (%)                XX.X              XX.X              XX.X              XX.X
  TPM dyn. range                XX.XX             XX.XX             XX.XX             XX.XX
  Promoter states                   X                 X                 X                 X
```

`results/held_out_likelihood_chr18.txt` — per-bin log-likelihood on the held-out chromosome:

```
  poisson       LL = ...   (-689.42 per bin)
  nb            LL = ...     (-6.41 per bin)
```

Figures are saved to `results/figures/`.