#!/usr/bin/env python3
"""
precompute_targetscan.py

Pre-computes TargetScan miRNA assignment and variance ratio for all
high-scoring genes in each sample, using the same sequential RPM-tier
exclusion logic as miRNA_gene_ranking.py.

Logic
-----
For each sample, genes are tested against RPM tiers in order (highest
confidence first).  A gene is assigned to the first tier where its
TargetScan weighted context++ score is non-zero; genes with no targets at
any tier are labelled 'none'.  Processing a tier only considers genes that
have not been assigned by a previous tier (sequential exclusion).

Output
------
One CSV per sample saved to:
  data_files/miRNA_files/precomputed/{cell_line}_{rep}_targetscan.csv

Columns
-------
  gene_name      — gene symbol
  rpm_tier       — tier that assigned the gene: '1000_None', '0_100', or 'none'
  weight         — cumulative TargetScan weighted context++ score (0 if none)
  miRNAs         — slash-separated targeting miRNA list (empty string if none)
  variance_ratio — spliced variance / unspliced variance across all cells
"""

import os
import pandas as pd

os.chdir('/media/yohanlefol/Expansion/CC_vel/')

from snake_scripts.snake_functions import snake_analysis_miRNA_functions as my_miR_func
from snake_scripts.snake_functions import snake_utils as my_utils

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SAMPLES = {
    'A549-CRISPRi-Dicer-EXP':'A_B',
    'A549-CRISPRi-AGO2-EXP':'A_B',
    'A549-CRISPRi-Dicer-KO':'A_B',
    'A549-CRISPRi-AGO2-KO':'A_B',
    'HaCat-CRISPRi-Dicer-EXP':'A_B',
    'HaCat-CRISPRi-AGO2-EXP':'A_B',
    'HaCat-CRISPRi-Dicer-KO':'A_B',
    'HaCat-CRISPRi-AGO2-KO':'A_B',
    'A549-Control':'A_B',
    'HaCat-Control':'A_B',
    'HaCat':'A_B'
}

# Order matters: high-confidence tier checked first
MIRNA_THRESH_LIST = ['1000_None', '0_100']

TS_PATH = ('data_files/miRNA_files/TS_files/'
           'Predicted_Targets_Context_Scores.default_predictions.txt')

OUT_DIR = 'data_files/miRNA_files/precomputed'

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)

for cell_line, rep in SAMPLES.items():
    print(f'\n=== {cell_line} / {rep} ===')

    ranked_path = f'data_files/data_results/rank/{cell_line}/{rep}_ranked_genes.csv'
    ranked_df   = pd.read_csv(ranked_path)
    all_genes   = list(ranked_df.loc[ranked_df['high_score'] > 0, 'gene_name'])
    print(f'  High-scoring genes: {len(all_genes)}')

    # Compute variance ratio (spliced_var / unspliced_var) across all cells
    mean_dict, _, _, _, boundary_dict = my_utils.get_CI_data(
        cell_line, ['spliced', 'unspliced'], rep, gene_selection=all_genes
    )
    _, variance_dict = my_miR_func.prep_variances_for_TS(
        mean_dict, boundary_dict, keys_to_use='All'
    )
    variance_ratio = variance_dict['All']   # {gene_name: float}

    gene_tier      = {}
    gene_weight    = {}
    gene_miRNAs    = {}

    unassigned = set(all_genes)

    for thresh in MIRNA_THRESH_LIST:
        if not unassigned:
            break

        miRNA_path = (f'data_files/miRNA_files/categorized/'
                      f'{cell_line}_miRNA_{thresh}.csv')
        miRNA_list = list(pd.read_csv(miRNA_path)['found'])

        working_df = pd.DataFrame({'gene_names': sorted(unassigned)})
        working_df, _ = my_miR_func.target_scan_analysis(
            TS_PATH, working_df, cell_line, thresh, miRNA_list=miRNA_list
        )

        assigned = working_df['weight'] != 0
        print(f'  {thresh}: {assigned.sum()} genes assigned')

        for _, row in working_df[assigned].iterrows():
            g = row['gene_names']
            gene_tier[g]   = thresh
            gene_weight[g] = row['weight']
            gene_miRNAs[g] = row['miRNAs']
            unassigned.discard(g)

    for g in unassigned:
        gene_tier[g]   = 'none'
        gene_weight[g] = 0.0
        gene_miRNAs[g] = ''

    result_df = pd.DataFrame({
        'gene_name':      all_genes,
        'rpm_tier':       [gene_tier[g]            for g in all_genes],
        'weight':         [gene_weight[g]           for g in all_genes],
        'miRNAs':         [gene_miRNAs[g]           for g in all_genes],
        'variance_ratio': [variance_ratio.get(g, float('nan')) for g in all_genes],
    })

    out_path = os.path.join(OUT_DIR, f'{cell_line}_{rep}_targetscan.csv')
    result_df.to_csv(out_path, index=False)
    print(f'  Saved → {out_path}')

    counts = result_df['rpm_tier'].value_counts().to_string()
    print(f'  Tier breakdown:\n    ' + counts.replace('\n', '\n    '))

print('\nDone.')
