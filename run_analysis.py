"""
Run all validation analyses and generate all figures for one or more HMM emission models.

Usage:
    python3 run_analysis.py --model bernoulli
    python3 run_analysis.py --model poisson
    python3 run_analysis.py --model all
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from experiments.biological_analysis import run as run_biological
from experiments.roadmap_comparison import run as run_roadmap
from experiments.expression_correlation import run as run_expression
from experiments.label_states import main as run_label_states
from experiments.summarize_evaluation import main as run_summary
from experiments.plots.plot_emission_heatmaps import run as plot_emissions
from experiments.plots.plot_fold_enrichment_heatmaps import run as plot_fold_enrichment
from experiments.plots.plot_roadmap_heatmaps import run as plot_roadmap_heatmaps
from experiments.plots.plot_tpm_violins import run as plot_tpm_violins
from experiments.plots.plot_summary_barchart import run as plot_summary_barchart

MODELS = ['bernoulli', 'poisson', 'nb']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS + ['all'], default='bernoulli',
                        help="Model to analyse, or 'all' to run all three")
    args = parser.parse_args()

    models = MODELS if args.model == 'all' else [args.model]

    for model in models:
        print(f'\n{"="*60}')
        print(f' Running analyses for: {model}')
        print(f'{"="*60}\n')

        print('\n--- Biological analysis ---')
        run_biological(model)

        print('\n--- Roadmap comparison ---')
        run_roadmap(model)

        print('\n--- Gene expression correlation ---')
        run_expression(model)

        print(f'\nAnalyses complete for {model}.')

    print('\n--- Assigning biological state labels ---')
    run_label_states()

    print('\n--- Generating figures ---')
    plot_emissions()
    plot_fold_enrichment()
    plot_roadmap_heatmaps()
    plot_tpm_violins()
    plot_summary_barchart()

    print('\n--- Biological quality summary (all models) ---')
    run_summary()

    print('\nDone.')

if __name__ == '__main__':
    main()
