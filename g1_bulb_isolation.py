#!/usr/bin/env python3
"""
g1_bulb_isolation.py

Isolates the dense "bulb" sub-population sitting in the high-PC1/high-PC2
corner of the G1 cluster (visible in
all_figures/{CELL_LINE}/{REPLICATE}/pca_bar_line_plots/reassigned/phase.pdf)
from the rest of the G1 cells, using a simple tunable rectangle gate in PCA
space.

Workflow
--------
1. Load the CC-genes phase-reassigned loom and recompute the PCA (the
   original PCA coordinates are never written to the loom, but the loom's
   .X still holds the exact RPM+log1p, CC-gene-only matrix that produced it,
   so recomputing reproduces the same PC1/PC2).
2. Plot all cells colored by phase — this should visually match the
   reference phase.pdf. Check this first.
3. Subset to G1 cells and split them into "bulb" / "main" using
   PC1_MIN / PC2_MIN.
4. Plot the G1-only split with the gate boundary drawn, to verify/tune.
5. Save per-cell group labels to CSV for the (future) pseudobulk DE step.

Tuning
------
Run once, inspect g1_bulb_split.pdf, adjust PC1_MIN/PC2_MIN below, re-run.
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.chdir('/home/yohanlefol/A_Projects/CC_vel/')

# snake_reassignment_functions.py itself does `from snake_functions import
# snake_utils` (a bare, non-package-qualified import) - that only resolves if
# snake_scripts/ is directly on sys.path, not just the repo root.
sys.path.insert(0, os.path.join(os.getcwd(), 'snake_scripts'))
from snake_scripts.snake_functions import snake_reassignment_functions as my_reassign_func

os.chdir('/media/yohanlefol/Expansion/CC_vel/')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CELL_LINE = 'A549-Control'
REPLICATE = 'A'    # flip to 'B' to check the other replicate
CC_LOOM_PATH = f'data_files/phase_reassigned/CC_{CELL_LINE}_{REPLICATE}.loom'
INITIAL_LOOM_PATH = f'data_files/initial_data/loom_files/{CELL_LINE}_{REPLICATE}.loom'
CC_GENES_PATH = 'data_files/initial_data/Original_cell_cycle_genes_with_new_candidates.csv'
SAVE_DIR = 'g1_bulb_isolation'

# Rectangle gate in PCA space - a G1 cell is part of the "bulb" if
# PC1 > PC1_MIN AND PC2 > PC2_MIN. Tune these after inspecting the first run.
PC1_MIN = 12.0
PC2_MIN = 2.5

PHASE_COLORS = {
    'G1':  np.array([52, 127, 184]) / 256,
    'S':   np.array([37, 139, 72]) / 256,
    'G2M': np.array([223, 127, 49]) / 256,
}
GROUP_COLORS = {
    'G1_bulb': '#d62728',
    'G1_main': '#1f77b4',
}

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1: replicate snake_reassignment.py's PCA, starting from the initial
# (raw) loom rather than the already-reassigned one.
#
# Recomputing PCA directly off the post-reassignment CC_*.loom didn't work -
# even rebuilding .X from its 'spliced' layer gave a degenerate PCA (one
# huge outlier, near-zero PC2), the classic sign of PCA on the wrong/raw
# matrix surviving some part of the write/read round trip. So instead we
# redo the actual computation on the initial loom:
#
#   - cell set: rather than re-running the *interactive* QC thresholds
#     (never saved anywhere), we reuse the exact barcode set that already
#     passed QC - it's just the obs_names of the existing CC_*.loom.
#     Subsetting the initial loom to those barcodes reproduces the original
#     post-QC cell set exactly, with no thresholds to re-guess.
#   - gene set: reproduced deterministically via the same
#     my_func.selection_method()/check_cols_and_rows() calls the original
#     script uses (snake_reassignment.py:109,112) - these only depend on
#     adata.var_names + the CC gene CSV, not on QC or scoring, so re-running
#     them here is exact, not approximate.
#   - normalize_total -> log1p -> PCA: identical calls/params to
#     snake_reassignment.py:101,102,114.
# ---------------------------------------------------------------------------
adata_reassigned = sc.read_loom(CC_LOOM_PATH, X_name="")
qc_barcodes = adata_reassigned.obs_names
phase_lookup = adata_reassigned.obs['phase']

adata = sc.read_loom(INITIAL_LOOM_PATH, X_name="")
adata.var_names_make_unique()

found_mask = adata.obs_names.isin(qc_barcodes)
print(f'{found_mask.sum()} / {len(qc_barcodes)} QC-passing barcodes found in '
      f'the initial loom (should be all of them).')
adata = adata[found_mask, :].copy()

sc.pp.normalize_total(adata, target_sum=1e6)
sc.pp.log1p(adata)

# Keep an all-genes copy before restricting to CC markers (same role as
# `adata_pre_selection` in snake_reassignment.py:106) - this is what Step 6
# uses for genome-wide DE, so we don't need to reload any other loom file.
adata_pre_selection = adata.copy()

adata = my_reassign_func.selection_method(adata, highly_variable=False, CC_path=CC_GENES_PATH)
my_reassign_func.check_cols_and_rows(adata)

sc.tl.pca(adata, svd_solver='arpack')

adata.obs['phase'] = phase_lookup.reindex(adata.obs_names)

pc1 = adata.obsm['X_pca'][:, 0]
pc2 = adata.obsm['X_pca'][:, 1]
phase = adata.obs['phase'].to_numpy()

# ---------------------------------------------------------------------------
# Step 2: sanity-check plot - should match the reference phase.pdf
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for ph, color in PHASE_COLORS.items():
    mask = phase == ph
    ax.scatter(pc1[mask], pc2[mask], s=10, color=color, label=ph)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title(f'{CELL_LINE} {REPLICATE} - phase (sanity check vs reference plot)')
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
fig.savefig(os.path.join(SAVE_DIR, 'phase_pca_full.pdf'), bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------------------
# Step 3: subset to G1, apply the rectangle gate
# ---------------------------------------------------------------------------
g1_mask = phase == 'G1'
g1_pc1 = pc1[g1_mask]
g1_pc2 = pc2[g1_mask]

bulb_mask = (g1_pc1 > PC1_MIN) & (g1_pc2 > PC2_MIN)

group = np.full(len(adata), 'other', dtype=object)
g1_group = np.where(bulb_mask, 'G1_bulb', 'G1_main')
group[g1_mask] = g1_group

n_bulb = int(bulb_mask.sum())
n_main = int((~bulb_mask).sum())
print(f'{CELL_LINE} {REPLICATE}: G1 cells = {len(g1_pc1)} '
      f'(G1_bulb = {n_bulb}, G1_main = {n_main})')

# ---------------------------------------------------------------------------
# Step 4: verification plot for the gate itself
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for grp, color in GROUP_COLORS.items():
    mask = g1_group == grp
    ax.scatter(g1_pc1[mask], g1_pc2[mask], s=10, color=color, label=grp)
ax.axvline(PC1_MIN, color='black', lw=1, linestyle='--')
ax.axhline(PC2_MIN, color='black', lw=1, linestyle='--')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title(f'{CELL_LINE} {REPLICATE} - G1 bulb gate '
             f'(PC1_MIN={PC1_MIN}, PC2_MIN={PC2_MIN})')
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
fig.savefig(os.path.join(SAVE_DIR, 'g1_bulb_split.pdf'), bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------------------
# Step 5: save per-cell group labels for the downstream pseudobulk DE step
# ---------------------------------------------------------------------------
labels_df = pd.DataFrame({
    'barcode': adata.obs_names,
    'phase': phase,
    'group': group,
})
labels_path = os.path.join(SAVE_DIR, f'g1_group_labels_{REPLICATE}.csv')
labels_df.to_csv(labels_path, index=False)
print(f'Saved group labels: {labels_path}')

# ---------------------------------------------------------------------------
# Step 6: pseudobulk-style differential expression, G1_bulb vs G1_main
#
# Reuses `adata_pre_selection` from Step 1 (same QC'd cell set, RPM + log1p
# normalized, but with every gene, not just the CC markers used for PCA) -
# no need to reload any other loom file. Restricted to the two G1 sub-groups
# and compared gene-by-gene with scanpy's rank_genes_groups (Wilcoxon
# rank-sum, BH-adjusted p-values, log2 fold change) - the standard tool for
# this kind of two-group single-cell comparison.
# ---------------------------------------------------------------------------
de_mask = np.isin(group, ['G1_bulb', 'G1_main'])
de_barcodes = adata.obs_names[de_mask]
de_groups = pd.Series(group[de_mask], index=de_barcodes)

adata_de = adata_pre_selection[de_barcodes, :].copy()
adata_de.obs['group'] = de_groups.reindex(adata_de.obs_names).astype('category')
my_reassign_func.check_cols_and_rows(adata_de)

sc.tl.rank_genes_groups(adata_de, groupby='group', groups=['G1_bulb'],
                         reference='G1_main', method='wilcoxon')

de_results = sc.get.rank_genes_groups_df(adata_de, group='G1_bulb')
de_results = de_results.rename(columns={'names': 'gene'}).sort_values('pvals_adj')

de_path = os.path.join(SAVE_DIR, f'g1_bulb_vs_main_DE_{REPLICATE}.csv')
de_results.to_csv(de_path, index=False)

n_sig = int((de_results['pvals_adj'] < 0.05).sum())
print(f'Saved DE results: {de_path}')
print(f'{n_sig} / {len(de_results)} genes significant at padj < 0.05')
print(de_results.head(15).to_string(index=False))

# ---------------------------------------------------------------------------
# Step 7: GSEA (preranked) on the DE results
#
# Ranks all tested genes by their Wilcoxon score from Step 6 (already
# signed: positive -> higher in G1_bulb, negative -> higher in G1_main) and
# runs gseapy's preranked GSEA against the Enrichr gene set libraries below.
# Requires `gseapy` (pip install gseapy) and internet access - gseapy fetches
# each library from the Enrichr API at run time.
# ---------------------------------------------------------------------------
import gseapy as gp

GENE_SET_LIBRARIES = ['MSigDB_Hallmark_2020', 'Reactome_2022']

ranking = de_results.set_index('gene')['scores'].sort_values(ascending=False)
ranking = ranking[~ranking.index.duplicated()]

for gene_set in GENE_SET_LIBRARIES:
    gsea_res = gp.prerank(
        rnk=ranking,
        gene_sets=gene_set,
        outdir=None,
        seed=42,
        permutation_num=1000,
    )
    gsea_df = gsea_res.res2d
    gsea_path = os.path.join(SAVE_DIR, f'g1_bulb_vs_main_GSEA_{gene_set}_{REPLICATE}.csv')
    gsea_df.to_csv(gsea_path, index=False)
    print(f'Saved GSEA ({gene_set}) results: {gsea_path}')
    print(gsea_df.head(10).to_string(index=False))

    # Dotplot of the top 20 positive-NES + top 20 negative-NES terms (up to
    # 40 total) - dot size = gene overlap %, color = FDR.
    gsea_df['NES'] = gsea_df['NES'].astype(float)
    top_pos = gsea_df[gsea_df['NES'] > 0].sort_values('NES', ascending=False).head(20)
    top_neg = gsea_df[gsea_df['NES'] < 0].sort_values('NES', ascending=True).head(20)
    top_terms_df = pd.concat([top_pos, top_neg])

    dotplot_path = os.path.join(
        SAVE_DIR, f'g1_bulb_vs_main_GSEA_dotplot_{gene_set}_{REPLICATE}.pdf')
    gp.dotplot(
        top_terms_df,
        column='FDR q-val',
        title=gene_set,
        top_term=len(top_terms_df),
        cmap='RdBu_r',
        size=8,
        figsize=(6, 15),
        ofname=dotplot_path,
    )
    print(f'Saved GSEA dotplot ({gene_set}): {dotplot_path}')
