"""
Download GENCODE v45 basic annotation for GRCh38 and extract:
  - annotations/tss_2kb.bed : ±2kb around TSS of protein-coding genes
  - annotations/gene_bodies.bed : full gene body (TSS to TES)
  - annotations/cpg_islands.bed : CpG islands from UCSC
  - annotations/ccre_PLS.bed : ENCODE cCRE promoter-like signatures (GM12878)
  - annotations/ccre_pELS.bed : ENCODE cCRE proximal enhancer-like signatures
  - annotations/ccre_dELS.bed : ENCODE cCRE distal enhancer-like signatures
  - annotations/ccre_CTCF.bed : ENCODE cCRE CTCF-only sites
  - annotations/ccre_DNaseH3K4me3.bed: ENCODE cCRE DNase+H3K4me3 sites

Run once before biological_analysis.py.
"""

import os
import urllib.request
import gzip

ANNO_DIR = 'annotations'
GENCODE_URL = (
    'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/'
    'gencode.v45.basic.annotation.gtf.gz'
)
CPG_URL = 'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cpgIslandExt.txt.gz'
CCRE_URL = (
    'https://www.encodeproject.org/files/ENCFF590IMH/@@download/ENCFF590IMH.bed.gz'
)

GTF_GZ = os.path.join(ANNO_DIR, 'gencode.v45.basic.gtf.gz')
CPG_RAW = os.path.join(ANNO_DIR, 'cpgIslandExt.txt.gz')
CCRE_GZ = os.path.join(ANNO_DIR, 'ENCFF590IMH.bed.gz')
TSS_BED = os.path.join(ANNO_DIR, 'tss_2kb.bed')
GENE_BED = os.path.join(ANNO_DIR, 'gene_bodies.bed')
CPG_BED = os.path.join(ANNO_DIR, 'cpg_islands.bed')

# cCRE type label (column 10, 0-indexed) → output filename
CCRE_TYPES = {
    'PLS':            os.path.join(ANNO_DIR, 'ccre_PLS.bed'),
    'pELS':           os.path.join(ANNO_DIR, 'ccre_pELS.bed'),
    'dELS':           os.path.join(ANNO_DIR, 'ccre_dELS.bed'),
    'CTCF-only':      os.path.join(ANNO_DIR, 'ccre_CTCF.bed'),
    'DNase-H3K4me3':  os.path.join(ANNO_DIR, 'ccre_DNaseH3K4me3.bed'),
}

FLANK = 2000 # bp each side of TSS


def download(url, dest):
    if os.path.exists(dest):
        print(f' Already exists: {dest}')
        return
    print(f' Downloading {os.path.basename(dest)}...')
    urllib.request.urlretrieve(url, dest)
    print(f' Done.')


def parse_gtf():
    tss_rows, gene_rows = [], []
    print(' Parsing GTF...')
    with gzip.open(GTF_GZ, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if fields[2] != 'gene':
                continue
            attrs = fields[8]
            if 'gene_type "protein_coding"' not in attrs:
                continue
            chrom = fields[0]
            start = int(fields[3]) - 1 # GTF is 1-based
            end = int(fields[4])
            strand = fields[6]
            name = ''
            for attr in attrs.split(';'):
                attr = attr.strip()
                if attr.startswith('gene_name'):
                    name = attr.split('"')[1]
                    break

            # gene body
            gene_rows.append((chrom, start, end, name, '.', strand))

            # TSS ±2kb
            tss = start if strand == '+' else end - 1
            tss_start = max(0, tss - FLANK)
            tss_end = tss + FLANK
            tss_rows.append((chrom, tss_start, tss_end, name, '.', strand))

    return tss_rows, gene_rows


def write_bed(rows, path):
    rows_sorted = sorted(rows, key=lambda r: (r[0], r[1]))
    with open(path, 'w') as f:
        for r in rows_sorted:
            f.write('\t'.join(str(x) for x in r) + '\n')
    print(f' Written: {path} ({len(rows_sorted)} entries)')


def parse_cpg():
    rows = []
    with gzip.open(CPG_RAW, 'rt') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            chrom = parts[1]
            start = int(parts[2])
            end = int(parts[3])
            name = parts[4]
            rows.append((chrom, start, end, name))
    return rows


def parse_ccre():
    """Split cCRE BED file by type (column 9) into per-type row lists.

    Column 9 can be compound (e.g. 'PLS,CTCF-bound'), so we use substring
    matching: a row goes into a bucket if the bucket key appears in column 9.
    """
    type_rows = {t: [] for t in CCRE_TYPES}
    with gzip.open(CCRE_GZ, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 10:
                continue
            label = parts[9]
            for ccre_type in CCRE_TYPES:
                if ccre_type in label:
                    type_rows[ccre_type].append((parts[0], int(parts[1]), int(parts[2]), parts[3]))
    return type_rows


os.makedirs(ANNO_DIR, exist_ok=True)

print('Downloading annotations...')
download(GENCODE_URL, GTF_GZ)
download(CPG_URL, CPG_RAW)
download(CCRE_URL, CCRE_GZ)

print('\nExtracting TSS and gene body regions...')
tss_rows, gene_rows = parse_gtf()
write_bed(tss_rows, TSS_BED)
write_bed(gene_rows, GENE_BED)

print('\nExtracting CpG islands...')
cpg_rows = parse_cpg()
write_bed(cpg_rows, CPG_BED)

print('\nExtracting ENCODE cCREs by type...')
ccre_rows = parse_ccre()
for ccre_type, path in CCRE_TYPES.items():
    write_bed(ccre_rows[ccre_type], path)

print('\nDone. Annotation files ready in', ANNO_DIR)