#!/usr/bin/env python3
"""
cross_sample_comparison.py

One plot per gene. Layout: rows = samples, columns = layer plot (wide) +
[spacer] + velocity plot (wide) + counts plot (wide) + annotation strips (narrow).

Columns
-------
  col 0 — Layer plot  (spliced / unspliced expression across pseudotime)
  col 1 — Spacer      (empty; width set by GAP_WIDTH)
  col 2 — Velocity plot (spliced / unspliced velocity with CIs)
  col 3 — Spacer      (empty; width set by COUNT_GAP_WIDTH)
  col 4 — Counts plot (+1/0/-1 velocity calls across pseudotime)
  col 5 — % time block (spliced/unspliced % of pseudotime at +1/0/-1)
  col 6 — Quantile bin  (unspliced_low_CI, full per-sample ranked list)
  col 7 — RPM threshold tier  (requires precomputed TargetScan CSV)
  col 8 — TargetScan miRNA score  (requires precomputed TargetScan CSV)
"""

import os
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


from snake_scripts.snake_functions import snake_analysis_functions as my_func
from snake_scripts.snake_functions import snake_utils as my_utils

os.chdir('/media/yohanlefol/Expansion/CC_vel/')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SAMPLES = {
    'HaCat-Control': 'A_B',
    'HaCat-CRISPRi-Dicer-EXP': 'A_B',
    'HaCat-CRISPRi-Dicer-KO': 'A_B',
    'HaCat-CRISPRi-AGO2-EXP': 'A_B',
    'HaCat-CRISPRi-AGO2-KO': 'A_B',
    'A549-Control':  'A_B',
    'A549-CRISPRi-Dicer-EXP': 'A_B',
    'A549-CRISPRi-Dicer-KO': 'A_B',
    'A549-CRISPRi-AGO2-EXP': 'A_B',
    'A549-CRISPRi-AGO2-KO': 'A_B',
}
LAYERS        = ['spliced', 'unspliced']
CC_GENES_PATH = 'data_files/initial_data/Original_cell_cycle_genes_with_new_candidates.csv'
N_CC          = 10   # known CC marker genes to plot
N_NON_CC      = 10   # non-CC genes to plot
SAVE_DIR      = 'cross_sample_facet_plots'

LAYER_WIDTH = 5.0   # inches allocated to the layer (expression) subplot
GAP_WIDTH   = 0.5   # inches of empty space between layer and velocity plots
VEL_WIDTH   = 5.0   # inches allocated to the velocity subplot
COUNT_GAP_WIDTH = 0.35  # inches of empty space between velocity and counts plots
COUNT_WIDTH = 5.0   # inches allocated to the counts subplot (matches LAYER_WIDTH/VEL_WIDTH)
BLOCK_WIDTH = 1.8   # inches allocated to the % time block (3-way split)
STRIP_WIDTH = 0.8   # inches per annotation strip


# One entry per sample row (same order as SAMPLES).
# Set to 1 to draw a dotted divider line below that sample, 0 for no line.
# e.g. [0,0,0,0,1,0,0,0,0,0] draws a line between the 5th and 6th sample.
ROW_DIVIDERS  = [1, 0, 1, 0, 1, 1, 0, 1, 0, 0]

QUANTILE_COLORS = {
    '0-25':   '#d6eaf8',
    '25-50':  '#7fb3d3',
    '50-75':  '#2471a3',
    '75-100': '#1a5276',
}
QUANTILE_TEXT_COLORS = {
    '0-25':   'black',
    '25-50':  'black',
    '50-75':  'white',
    '75-100': 'white',
}

RPM_TIER_COLORS = {
    '1000_None': '#1d6a27',
    '0_100':     '#82c987',
    'none':      '#e8e8e8',
}
RPM_TIER_LABELS = {
    '1000_None': '≥1000',
    '0_100':     '0–100',
    'none':      'none',
}
RPM_TIER_TEXT_COLORS = {
    '1000_None': 'white',
    '0_100':     'black',
    'none':      'black',
}

WEIGHT_STRONG   = -0.3   # TargetScan threshold for strong targets
SCORE_COLORS    = {
    'strong':   '#7b2d8b',   # weight < WEIGHT_STRONG
    'weak':     '#c49dc8',   # WEIGHT_STRONG <= weight < 0
    'none':     '#e8e8e8',   # weight == 0
}
SCORE_TEXT_COLORS = {
    'strong': 'white',
    'weak':   'black',
    'none':   'black',
}

# ---------------------------------------------------------------------------
# Step 1: load ranked data and pre-computed TargetScan results
#   ranked_full     — all genes per sample, used for quantile computation
#   ranked_filtered — high_score > 0 only, used for common gene selection
#   ts_data         — pre-computed TargetScan results (rpm_tier, weight, miRNAs)
#                     produced by precompute_targetscan.py; set to {} to skip
# ---------------------------------------------------------------------------
ranked_full     = {}
ranked_filtered = {}
ts_data         = {}

for cell_line, rep in SAMPLES.items():
    path = f'data_files/data_results/rank/{cell_line}/{rep}_ranked_genes.csv'
    df   = pd.read_csv(path).set_index('gene_name')
    ranked_full[cell_line]     = df
    ranked_filtered[cell_line] = df[df['high_score'] > 0]

    ts_path = f'data_files/miRNA_files/precomputed/{cell_line}_{rep}_targetscan.csv'
    if os.path.exists(ts_path):
        ts_data[cell_line] = pd.read_csv(ts_path).set_index('gene_name')
    else:
        print(f'  [TargetScan] {ts_path} not found — RPM and score strips will be skipped')

TS_READY = len(ts_data) == len(SAMPLES)

common_genes = sorted(set.intersection(*[set(df.index) for df in ranked_filtered.values()]))
print(f"Common high-scoring genes across all samples: {len(common_genes)}")

avg_score = pd.Series({
    g: np.mean([ranked_filtered[cl].loc[g, 'high_score'] for cl in SAMPLES])
    for g in common_genes
})

# ---------------------------------------------------------------------------
# Step 2: classify and select genes
# ---------------------------------------------------------------------------
cc_df       = pd.read_csv(CC_GENES_PATH)
cc_gene_set = set(cc_df['gene'])

cc_candidates     = avg_score[avg_score.index.isin(cc_gene_set)].sort_values(ascending=False)
non_cc_candidates = avg_score[~avg_score.index.isin(cc_gene_set)].sort_values(ascending=False)

selected_cc     = list(cc_candidates.index[:N_CC])
selected_non_cc = list(non_cc_candidates.index[:N_NON_CC])
genes_to_plot   = selected_cc + selected_non_cc

print(f"CC genes selected ({len(selected_cc)}):     {selected_cc}")
print(f"Non-CC genes selected ({len(selected_non_cc)}): {selected_non_cc}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_orientation(boundary_dict):
    """Derive plot orientation from cell-cycle boundaries.

    Uses the same condition as the velocity/counts plots (plot_vels_and_CIs,
    plot_counts) so the layer plot's x-axis inversion stays consistent with
    theirs and the phase-boundary color bars line up across all subplots.
    """
    return 'G2M' if boundary_dict['G2M'] != 0 else 'G1'


def get_quantile_bin(gene, full_ranked_df, value_col='unspliced_low_CI'):
    q   = full_ranked_df[value_col].quantile([0.25, 0.5, 0.75])
    val = full_ranked_df.loc[gene, value_col]
    if val <= q[0.25]:
        return '0-25'
    elif val <= q[0.50]:
        return '25-50'
    elif val <= q[0.75]:
        return '50-75'
    else:
        return '75-100'


def draw_tile(ax, label, bg_color, text_color='black', header=None):
    """Colored annotation tile with a centered label and an optional column header."""
    ax.set_facecolor(bg_color)
    ax.text(0.5, 0.5, label, ha='center', va='center',
            transform=ax.transAxes, fontsize=7, color=text_color)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
    if header is not None:
        ax.set_title(header, fontsize=8, pad=3)


def draw_count_block(ax, spli_pct, unspli_pct, header=None):
    """Annotation block showing % of pseudotime spent at +1/0/-1 velocity calls.

    Divided into three horizontal bands (+1 / 0 / -1, top to bottom to match
    the counts plot's y-axis), each listing the spliced (purple) and
    unspliced (orange) percentage for that band.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)

    for row, state in enumerate((1, 0, -1)):  # top to bottom
        y_center = 2.5 - row
        label = f'{state:+d}' if state != 0 else '0'
        ax.text(0.06, y_center, label, ha='left', va='center', fontsize=7, fontweight='bold')
        ax.text(0.38, y_center, f"{spli_pct[state]:.0f}%", ha='left', va='center',
                fontsize=7, color="#542788")
        ax.text(0.70, y_center, f"{unspli_pct[state]:.0f}%", ha='left', va='center',
                fontsize=7, color="#b35806")

    ax.axhline(1, color='black', linewidth=0.4)
    ax.axhline(2, color='black', linewidth=0.4)
    if header is not None:
        ax.set_title(header, fontsize=7.5, pad=3)


def pct_time(counts_arr):
    """Percentage of pseudotime points spent at each velocity call (+1/0/-1)."""
    counts_arr = np.asarray(counts_arr)
    total = len(counts_arr)
    return {v: 100 * np.sum(counts_arr == v) / total for v in (1, 0, -1)}


def weight_to_score_group(weight):
    if weight < WEIGHT_STRONG:
        return 'strong'
    elif weight < 0:
        return 'weak'
    else:
        return 'none'


def draw_row_dividers(fig, gs, dividers):
    """Draw dotted horizontal lines across the figure between sample rows.

    Parameters
    ----------
    fig : matplotlib Figure
    gs  : GridSpec used for the figure
    dividers : sequence of int/bool, length == number of rows in gs.
        dividers[i] == 1 → draw a line between row i and row i+1.
    """
    n_rows = gs.get_geometry()[0]
    for i, draw in enumerate(dividers):
        if draw and i < n_rows - 1:
            bottom = gs[i,     0].get_position(fig).y0
            top    = gs[i + 1, 0].get_position(fig).y1
            y_line = (bottom + top) / 2
            fig.add_artist(Line2D(
                [0, 1], [y_line, y_line],
                transform=fig.transFigure,
                color='black', linestyle='dotted', linewidth=1.5,
                clip_on=False,
            ))


# ---------------------------------------------------------------------------
# Step 3: load velocity data and expression data for each sample
# ---------------------------------------------------------------------------
sample_data = {}
for cell_line, rep in SAMPLES.items():
    mean_dict, CI_dict, bool_dict, count_dict, boundary_dict = my_utils.get_CI_data(
        cell_line, LAYERS, rep, gene_selection=genes_to_plot
    )
    vlm_dict      = my_utils.get_vlm_values(cell_line, LAYERS, rep, get_mean=False)
    vlm_mean_dict = my_utils.get_vlm_values(cell_line, LAYERS, rep, get_mean=True)
    for layer in LAYERS:
        keep = [g for g in genes_to_plot if g in vlm_dict[layer].columns]
        vlm_dict[layer]      = vlm_dict[layer][keep]
        vlm_mean_dict[layer] = vlm_mean_dict[layer][keep]
    sample_data[cell_line] = dict(
        mean_dict=mean_dict,
        CI_dict=CI_dict,
        bool_dict=bool_dict,
        count_dict=count_dict,
        boundary_dict=boundary_dict,
        vlm_dict=vlm_dict,
        vlm_mean_dict=vlm_mean_dict,
        orientation=get_orientation(boundary_dict),
    )


# ---------------------------------------------------------------------------
# Step 4: one figure per gene
# ---------------------------------------------------------------------------
# Strip column definitions.
# TargetScan strips are included only when pre-computed data exists for all samples.
strip_headers = ['Quantile']
if TS_READY:
    strip_headers += ['RPM tier', 'Score']
n_strips = len(strip_headers)

width_ratios = [LAYER_WIDTH, GAP_WIDTH, VEL_WIDTH, COUNT_GAP_WIDTH, COUNT_WIDTH, BLOCK_WIDTH] + [STRIP_WIDTH] * n_strips
fig_width    = (LAYER_WIDTH + GAP_WIDTH + VEL_WIDTH + COUNT_GAP_WIDTH + COUNT_WIDTH + BLOCK_WIDTH
                 + STRIP_WIDTH * n_strips)
sample_names = list(SAMPLES.keys())
n_samples    = len(sample_names)

os.makedirs(SAVE_DIR, exist_ok=True)

for gene in genes_to_plot:
    is_cc = gene in cc_gene_set
    title = f'[CC marker]  {gene}' if is_cc else gene

    fig = plt.figure(figsize=(fig_width, 2.8 * n_samples))
    gs  = GridSpec(n_samples, 6 + n_strips, figure=fig,
                   width_ratios=width_ratios,
                   top=0.97, hspace=0.35, wspace=0.08)

    for row_idx, cell_line in enumerate(sample_names):
        d = sample_data[cell_line]

        # layer (expression) subplot — col 0
        ax_layer = fig.add_subplot(gs[row_idx, 0])
        plt.sca(ax_layer)
        if gene in d['vlm_dict']['spliced'].columns:
            my_func.plot_layer_plot(
                ax_layer, d['vlm_dict'], d['vlm_mean_dict'],
                d['boundary_dict'], gene, d['orientation'],
            )
        ax_layer.set_ylabel(cell_line, fontsize=9, labelpad=4)

        # velocity subplot — col 2
        ax_vel = fig.add_subplot(gs[row_idx, 2])
        plt.sca(ax_vel)
        main_dict = my_func.subset_the_dicts_merged_CI(
            gene, d['mean_dict'], d['bool_dict'], d['CI_dict']
        )
        my_func.plot_vels_and_CIs(main_dict, ax_vel, gene, d['boundary_dict'])

        # counts subplot — col 4 (col 3 left empty as a spacer, width=COUNT_GAP_WIDTH)
        ax_counts = fig.add_subplot(gs[row_idx, 4])
        plt.sca(ax_counts)
        reverse = d['boundary_dict']['G2M'] != 0   # same convention as plot_curve_count
        my_func.plot_counts(d['count_dict'], gene, ax_counts, d['boundary_dict'], reverse=reverse)

        # % time block — col 5
        ax_block   = fig.add_subplot(gs[row_idx, 5])
        spli_pct   = pct_time(d['count_dict']['spliced'][gene])
        unspli_pct = pct_time(d['count_dict']['unspliced'][gene])
        draw_count_block(
            ax_block, spli_pct, unspli_pct,
            header='% time\n(sp | un)' if row_idx == 0 else None,
        )

        # quantile strip — col 6
        ax_q  = fig.add_subplot(gs[row_idx, 6])
        q_bin = get_quantile_bin(gene, ranked_full[cell_line])
        draw_tile(
            ax_q, q_bin,
            bg_color=QUANTILE_COLORS[q_bin],
            text_color=QUANTILE_TEXT_COLORS[q_bin],
            header=strip_headers[0] if row_idx == 0 else None,
        )

        if TS_READY:
            row_ts = ts_data[cell_line].loc[gene] if gene in ts_data[cell_line].index else None

            # RPM tier strip — col 7
            ax_rpm = fig.add_subplot(gs[row_idx, 7])
            tier   = row_ts['rpm_tier'] if row_ts is not None else 'none'
            draw_tile(
                ax_rpm, RPM_TIER_LABELS[tier],
                bg_color=RPM_TIER_COLORS[tier],
                text_color=RPM_TIER_TEXT_COLORS[tier],
                header=strip_headers[1] if row_idx == 0 else None,
            )

            # Score strip — col 8
            ax_sc = fig.add_subplot(gs[row_idx, 8])
            w     = row_ts['weight'] if row_ts is not None else 0.0
            sg    = weight_to_score_group(w)
            draw_tile(
                ax_sc, f'{w:.2f}',
                bg_color=SCORE_COLORS[sg],
                text_color=SCORE_TEXT_COLORS[sg],
                header=strip_headers[2] if row_idx == 0 else None,
            )

    draw_row_dividers(fig, gs, ROW_DIVIDERS)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.995)
    out_path = os.path.join(SAVE_DIR, f'{gene}.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close('all')
    gc.collect()
    print(f'  Saved: {gene}.png')

print(f'\nAll plots saved to {SAVE_DIR}/')