#!/usr/bin/env python3
"""
mirna_condition_comparison.py

Compares variance ratio (spliced_var / unspliced_var) distributions across
conditions (Control / Dicer-EXP / Dicer-KO / AGO2-EXP / AGO2-KO) within
four miRNA targeting groups.

Gene groups are defined from the Control sample of each cell line so that
the same set of genes is compared across all conditions.  The question asked:
does knocking out miRNA machinery shift the spliced/unspliced variance ratio,
and does the effect depend on how strongly genes are targeted by miRNAs?

Outputs
-------
  {SAVE_DIR}/summary_table.csv              — per (cell_line, condition, group) absolute stats
  {SAVE_DIR}/delta_summary_table.csv        — per (cell_line, condition, group) Δ vs Control
  {SAVE_DIR}/delta_quantile_summary.csv     — same but also stratified by quantile bin
  {SAVE_DIR}/HaCat_comparison.png
  {SAVE_DIR}/A549_comparison.png
  {SAVE_DIR}/HaCat_delta.png               — per-gene Δlog10(variance ratio) vs Control
  {SAVE_DIR}/A549_delta.png
  {SAVE_DIR}/HaCat_delta_by_quantile.png   — delta split into 4 quantile panels
  {SAVE_DIR}/A549_delta_by_quantile.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl

os.chdir('/media/yohanlefol/Expansion/CC_vel/')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONDITIONS = {
    'HaCat': {
        'Control':   ('HaCat-Control',            'A_B'),
        'Dicer-EXP': ('HaCat-CRISPRi-Dicer-EXP',  'A_B'),
        'Dicer-KO':  ('HaCat-CRISPRi-Dicer-KO',   'A_B'),
        'AGO2-EXP':  ('HaCat-CRISPRi-AGO2-EXP',   'A_B'),
        'AGO2-KO':   ('HaCat-CRISPRi-AGO2-KO',    'A_B'),
    },
    'A549': {
        'Control':   ('A549-Control',              'A_B'),
        'Dicer-EXP': ('A549-CRISPRi-Dicer-EXP',   'A_B'),
        'Dicer-KO':  ('A549-CRISPRi-Dicer-KO',    'A_B'),
        'AGO2-EXP':  ('A549-CRISPRi-AGO2-EXP',    'A_B'),
        'AGO2-KO':   ('A549-CRISPRi-AGO2-KO',     'A_B'),
    },
}

CONDITION_ORDER = ['Control', 'Dicer-EXP', 'Dicer-KO', 'AGO2-EXP', 'AGO2-KO']
CONDITION_COLORS = {
    'Control':   '#888888',
    'Dicer-EXP': '#9ecae1',
    'Dicer-KO':  '#08519c',
    'AGO2-EXP':  '#fdae6b',
    'AGO2-KO':   '#8c2d04',
}

WEIGHT_THRESHOLD = -0.3

# (rpm_tier_label, score_key, display_label)
# Groups mirror create_miRNA_boxplot.py — they are NOT mutually exclusive.
GROUPS = [
    ('1000_None', '< -0.3', '>=1000 RPM\nscore < -0.3'),
    ('1000_None', '0',      '>=1000 RPM\nno target'),
    ('0_100',     '< -0.3', '0-100 RPM\nscore < -0.3'),
    ('0_100',     '0',      'no miRNA\ntarget'),
]

QUANTILES = ['0-25', '25-50', '50-75', '75-100']

DO_LOG10 = True

TS_DIR   = 'data_files/miRNA_files/precomputed'
RANK_DIR = 'data_files/data_results/rank'
SAVE_DIR = 'mirna_condition_comparison'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _group_mask(df, rpm_tier_label, score_key):
    """Boolean mask selecting genes belonging to one GROUPS entry."""
    if rpm_tier_label == '1000_None':
        if score_key == '< -0.3':
            return (df['rpm_tier'] == '1000_None') & (df['weight'] < WEIGHT_THRESHOLD)
        else:
            return df['rpm_tier'].isin(['0_100', 'none'])
    else:
        if score_key == '< -0.3':
            return (df['rpm_tier'] == '0_100') & (df['weight'] < WEIGHT_THRESHOLD)
        else:
            return df['rpm_tier'] == 'none'


def _apply_log10(vals):
    arr = np.array(vals, dtype=float)
    arr = arr[arr > 0]
    return np.log10(arr)


def load_cell_line_data(cell_line):
    """
    Load variance ratios for all conditions, with gene groups defined from Control.

    Returns
    -------
    group_data : dict
        group_data[group_label][condition] = np.array of (log10) variance ratios
    common_n : int
        Number of genes common to all conditions.
    """
    conds = CONDITIONS[cell_line]
    ctrl_cl, ctrl_rep = conds['Control']

    # Gene groups from Control's precomputed CSV
    ctrl_ts = pd.read_csv(f'{TS_DIR}/{ctrl_cl}_{ctrl_rep}_targetscan.csv')

    # Restrict to high-scoring genes in the control; compute quantile bins
    ranked   = pd.read_csv(f'{RANK_DIR}/{ctrl_cl}/{ctrl_rep}_ranked_genes.csv')
    hs_genes = set(ranked.loc[ranked['high_score'] > 0, 'gene_name'])
    ctrl_ts  = ctrl_ts[ctrl_ts['gene_name'].isin(hs_genes)].copy()

    ranked_hs = ranked[ranked['high_score'] > 0].copy()
    q = ranked_hs['spliced_low_CI'].quantile([0.25, 0.5, 0.75])
    ranked_hs['quantile_bin'] = np.select(
        [
            ranked_hs['spliced_low_CI'] <= q[0.25],
            (ranked_hs['spliced_low_CI'] > q[0.25]) & (ranked_hs['spliced_low_CI'] <= q[0.50]),
            (ranked_hs['spliced_low_CI'] > q[0.50]) & (ranked_hs['spliced_low_CI'] <= q[0.75]),
            ranked_hs['spliced_low_CI'] > q[0.75],
        ],
        QUANTILES,
    )

    # Load variance ratios from every condition (rename column to condition name)
    vr_frames = {}
    for cond, (cl, rep) in conds.items():
        path = f'{TS_DIR}/{cl}_{rep}_targetscan.csv'
        if not os.path.exists(path):
            print(f'  [skip] missing precomputed file: {path}')
            continue
        vr_frames[cond] = (
            pd.read_csv(path)[['gene_name', 'variance_ratio']]
            .rename(columns={'variance_ratio': cond})
        )

    # Merge all conditions — keep only genes present in every condition
    merged = ctrl_ts[['gene_name', 'rpm_tier', 'weight']].copy()
    for cond, df in vr_frames.items():
        merged = merged.merge(df, on='gene_name', how='inner')
    merged = merged.merge(
        ranked_hs[['gene_name', 'quantile_bin']], on='gene_name', how='left'
    )
    common_n = len(merged)
    print(f'  Genes common to all conditions: {common_n}')

    # Build per-group, per-condition value arrays (absolute)
    # Also build per-gene delta = log10(vr_cond) - log10(vr_ctrl)
    non_ctrl = [c for c in CONDITION_ORDER if c != 'Control']
    group_data  = {}
    delta_data  = {}
    for rpm_tier_label, score_key, label in GROUPS:
        mask   = _group_mask(merged, rpm_tier_label, score_key)
        subset = merged[mask].copy()
        print(f'    {label.replace(chr(10), " "):30s}: {len(subset)} genes')

        group_data[label] = {}
        for cond in CONDITION_ORDER:
            if cond not in vr_frames:
                group_data[label][cond] = np.array([])
                continue
            vals = subset[cond].dropna().values
            group_data[label][cond] = _apply_log10(vals) if DO_LOG10 else vals

        delta_data[label] = {}
        for cond in non_ctrl:
            if cond not in vr_frames or 'Control' not in vr_frames:
                delta_data[label][cond] = np.array([])
                continue
            valid = subset[(subset['Control'] > 0) & (subset[cond] > 0)]
            delta_data[label][cond] = (
                np.log10(valid[cond].values) - np.log10(valid['Control'].values)
            )

    # Build delta stratified by quantile bin
    delta_by_q = {}
    for q_bin in QUANTILES:
        q_subset = merged[merged['quantile_bin'] == q_bin]
        delta_by_q[q_bin] = {}
        for rpm_tier_label, score_key, label in GROUPS:
            mask   = _group_mask(q_subset, rpm_tier_label, score_key)
            subset = q_subset[mask].copy()
            delta_by_q[q_bin][label] = {}
            for cond in non_ctrl:
                if cond not in vr_frames or 'Control' not in vr_frames:
                    delta_by_q[q_bin][label][cond] = np.array([])
                    continue
                valid = subset[(subset['Control'] > 0) & (subset[cond] > 0)]
                delta_by_q[q_bin][label][cond] = (
                    np.log10(valid[cond].values) - np.log10(valid['Control'].values)
                )

    return group_data, delta_data, delta_by_q, common_n


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_summary_table(all_data):
    rows = []
    for cell_line, (group_data, _delta, _delta_q, _n) in all_data.items():
        for rpm_tier_label, score_key, label in GROUPS:
            clean_label = label.replace('\n', ' ')
            for cond in CONDITION_ORDER:
                arr = group_data[label].get(cond, np.array([]))
                rows.append({
                    'cell_line':  cell_line,
                    'condition':  cond,
                    'group':      clean_label,
                    'n_genes':    len(arr),
                    'median':     float(np.median(arr))          if len(arr) else np.nan,
                    'q25':        float(np.percentile(arr, 25))  if len(arr) else np.nan,
                    'q75':        float(np.percentile(arr, 75))  if len(arr) else np.nan,
                    'mean':       float(np.mean(arr))            if len(arr) else np.nan,
                    'std':        float(np.std(arr, ddof=1))     if len(arr) > 1 else np.nan,
                })
    df = pd.DataFrame(rows)
    col_order = ['cell_line', 'condition', 'group', 'n_genes',
                 'median', 'q25', 'q75', 'mean', 'std']
    return df[col_order]


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def plot_comparison(cell_line, group_data, save_path):
    n_groups    = len(GROUPS)
    n_conds     = len(CONDITION_ORDER)
    box_width   = 0.12
    box_spacing = 0.02
    group_span  = n_conds * box_width + (n_conds - 1) * box_spacing
    xlocations  = np.arange(n_groups) * (group_span + 0.4)

    mpl.rcParams['figure.dpi'] = 300
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.grid(True, linestyle='dotted')
    ax.set_axisbelow(True)

    for c_idx, cond in enumerate(CONDITION_ORDER):
        offset    = (c_idx - n_conds / 2 + 0.5) * (box_width + box_spacing)
        positions = [x + offset for x in xlocations]
        vals_per_group = [group_data[g[2]].get(cond, np.array([])) for g in GROUPS]

        ax.boxplot(
            vals_per_group,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            boxprops=dict(facecolor=CONDITION_COLORS[cond], linewidth=0.8),
            medianprops=dict(color='black', linewidth=1.4),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
            flierprops=dict(marker='.', markersize=2, alpha=0.4),
            showfliers=False,
            labels=[''] * n_groups,
        )

    ax.set_xticks(xlocations)
    ax.set_xticklabels([g[2] for g in GROUPS], fontsize=10)
    ax.set_xlabel('miRNA targeting group (defined from Control)', fontsize=11)
    ylabel = 'log₁₀(variance ratio)' if DO_LOG10 else 'variance ratio'
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(
        f'{cell_line} — spliced/unspliced variance ratio by condition and miRNA targeting',
        fontsize=12,
    )

    handles = [
        mpatches.Patch(facecolor=CONDITION_COLORS[c], edgecolor='grey',
                       linewidth=0.5, label=c)
        for c in CONDITION_ORDER
    ]
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.01, 0.5),
              title='Condition', fontsize=9, title_fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')
    print(f'  Saved: {save_path}')


# ---------------------------------------------------------------------------
# Delta summary table
# ---------------------------------------------------------------------------

def build_delta_summary_table(all_data):
    """Per-gene Δlog10(variance ratio) = log10(vr_cond) - log10(vr_ctrl)."""
    non_ctrl = [c for c in CONDITION_ORDER if c != 'Control']
    rows = []
    for cell_line, (_gd, delta_data, _delta_q, _n) in all_data.items():
        for rpm_tier_label, score_key, label in GROUPS:
            clean_label = label.replace('\n', ' ')
            for cond in non_ctrl:
                arr = delta_data[label].get(cond, np.array([]))
                rows.append({
                    'cell_line':  cell_line,
                    'condition':  cond,
                    'group':      clean_label,
                    'n_genes':    len(arr),
                    'median_delta': float(np.median(arr))         if len(arr) else np.nan,
                    'q25_delta':   float(np.percentile(arr, 25)) if len(arr) else np.nan,
                    'q75_delta':   float(np.percentile(arr, 75)) if len(arr) else np.nan,
                    'mean_delta':  float(np.mean(arr))           if len(arr) else np.nan,
                    'std_delta':   float(np.std(arr, ddof=1))    if len(arr) > 1 else np.nan,
                    'pct_positive': float(np.mean(arr > 0) * 100) if len(arr) else np.nan,
                })
    col_order = ['cell_line', 'condition', 'group', 'n_genes',
                 'median_delta', 'q25_delta', 'q75_delta',
                 'mean_delta', 'std_delta', 'pct_positive']
    return pd.DataFrame(rows)[col_order]


# ---------------------------------------------------------------------------
# Delta plot
# ---------------------------------------------------------------------------

def plot_delta(cell_line, delta_data, save_path):
    non_ctrl    = [c for c in CONDITION_ORDER if c != 'Control']
    n_groups    = len(GROUPS)
    n_conds     = len(non_ctrl)
    box_width   = 0.12
    box_spacing = 0.02
    group_span  = n_conds * box_width + (n_conds - 1) * box_spacing
    xlocations  = np.arange(n_groups) * (group_span + 0.4)

    mpl.rcParams['figure.dpi'] = 300
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.grid(True, linestyle='dotted')
    ax.set_axisbelow(True)
    ax.axhline(0, color='black', linewidth=1.2, linestyle='--', zorder=3)

    for c_idx, cond in enumerate(non_ctrl):
        offset    = (c_idx - n_conds / 2 + 0.5) * (box_width + box_spacing)
        positions = [x + offset for x in xlocations]
        vals_per_group = [delta_data[g[2]].get(cond, np.array([])) for g in GROUPS]

        ax.boxplot(
            vals_per_group,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            boxprops=dict(facecolor=CONDITION_COLORS[cond], linewidth=0.8),
            medianprops=dict(color='black', linewidth=1.4),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
            flierprops=dict(marker='.', markersize=2, alpha=0.4),
            showfliers=False,
            labels=[''] * n_groups,
        )

    ax.set_xticks(xlocations)
    ax.set_xticklabels([g[2] for g in GROUPS], fontsize=10)
    ax.set_xlabel('miRNA targeting group (defined from Control)', fontsize=11)
    ax.set_ylabel('Δ log₁₀(variance ratio)  vs  Control', fontsize=11)
    ax.set_title(
        f'{cell_line} — per-gene shift in variance ratio relative to Control',
        fontsize=12,
    )

    handles = [
        mpatches.Patch(facecolor=CONDITION_COLORS[c], edgecolor='grey',
                       linewidth=0.5, label=c)
        for c in non_ctrl
    ]
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.01, 0.5),
              title='Condition', fontsize=9, title_fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')
    print(f'  Saved: {save_path}')


# ---------------------------------------------------------------------------
# Delta-by-quantile summary table
# ---------------------------------------------------------------------------

def build_delta_quantile_summary_table(all_data):
    """Per-gene Δlog10(variance ratio) stratified by control quantile bin."""
    non_ctrl = [c for c in CONDITION_ORDER if c != 'Control']
    rows = []
    for cell_line, (_gd, _delta, delta_by_q, _n) in all_data.items():
        for q_bin in QUANTILES:
            for rpm_tier_label, score_key, label in GROUPS:
                clean_label = label.replace('\n', ' ')
                for cond in non_ctrl:
                    arr = delta_by_q[q_bin][label].get(cond, np.array([]))
                    rows.append({
                        'cell_line':    cell_line,
                        'condition':    cond,
                        'quantile_bin': q_bin,
                        'group':        clean_label,
                        'n_genes':      len(arr),
                        'median_delta': float(np.median(arr))          if len(arr) else np.nan,
                        'q25_delta':    float(np.percentile(arr, 25))  if len(arr) else np.nan,
                        'q75_delta':    float(np.percentile(arr, 75))  if len(arr) else np.nan,
                        'mean_delta':   float(np.mean(arr))            if len(arr) else np.nan,
                        'std_delta':    float(np.std(arr, ddof=1))     if len(arr) > 1 else np.nan,
                        'pct_positive': float(np.mean(arr > 0) * 100) if len(arr) else np.nan,
                    })
    col_order = ['cell_line', 'condition', 'quantile_bin', 'group', 'n_genes',
                 'median_delta', 'q25_delta', 'q75_delta',
                 'mean_delta', 'std_delta', 'pct_positive']
    return pd.DataFrame(rows)[col_order]


# ---------------------------------------------------------------------------
# Delta-by-quantile plot
# ---------------------------------------------------------------------------

def plot_delta_by_quantile(cell_line, delta_by_q, save_path):
    """2×2 grid of delta boxplots, one panel per quantile bin, shared y-axis."""
    non_ctrl    = [c for c in CONDITION_ORDER if c != 'Control']
    n_groups    = len(GROUPS)
    n_conds     = len(non_ctrl)
    box_width   = 0.12
    box_spacing = 0.02
    group_span  = n_conds * box_width + (n_conds - 1) * box_spacing
    xlocations  = np.arange(n_groups) * (group_span + 0.4)

    mpl.rcParams['figure.dpi'] = 300
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharey=True)
    axes_flat = axes.flatten()

    for ax_idx, q_bin in enumerate(QUANTILES):
        ax = axes_flat[ax_idx]
        ax.grid(True, linestyle='dotted')
        ax.set_axisbelow(True)
        ax.axhline(0, color='black', linewidth=1.2, linestyle='--', zorder=3)

        for c_idx, cond in enumerate(non_ctrl):
            offset    = (c_idx - n_conds / 2 + 0.5) * (box_width + box_spacing)
            positions = [x + offset for x in xlocations]
            vals_per_group = [
                delta_by_q[q_bin][g[2]].get(cond, np.array([]))
                for g in GROUPS
            ]

            ax.boxplot(
                vals_per_group,
                positions=positions,
                widths=box_width,
                patch_artist=True,
                boxprops=dict(facecolor=CONDITION_COLORS[cond], linewidth=0.8),
                medianprops=dict(color='black', linewidth=1.4),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                flierprops=dict(marker='.', markersize=2, alpha=0.4),
                showfliers=False,
                labels=[''] * n_groups,
            )

        ax.set_xticks(xlocations)
        ax.set_xticklabels([g[2] for g in GROUPS], fontsize=9)
        ax.set_title(f'Quantile bin: {q_bin}', fontsize=10)
        if ax_idx % 2 == 0:
            ax.set_ylabel('Δ log₁₀(variance ratio)  vs  Control', fontsize=9)

    handles = [
        mpatches.Patch(facecolor=CONDITION_COLORS[c], edgecolor='grey',
                       linewidth=0.5, label=c)
        for c in non_ctrl
    ]
    fig.legend(handles=handles, loc='lower center', ncol=len(non_ctrl),
               title='Condition', fontsize=9, title_fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f'{cell_line} — Δlog₁₀(variance ratio) by quantile bin of control expression',
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')
    print(f'  Saved: {save_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

os.makedirs(SAVE_DIR, exist_ok=True)

all_data = {}
for cell_line in CONDITIONS:
    print(f'\n=== {cell_line} ===')
    all_data[cell_line] = load_cell_line_data(cell_line)

summary_df = build_summary_table(all_data)
summary_path = os.path.join(SAVE_DIR, 'summary_table.csv')
summary_df.to_csv(summary_path, index=False)
print(f'\nSummary table → {summary_path}')
print(summary_df.to_string(index=False))

delta_df = build_delta_summary_table(all_data)
delta_path = os.path.join(SAVE_DIR, 'delta_summary_table.csv')
delta_df.to_csv(delta_path, index=False)
print(f'\nDelta summary table → {delta_path}')
print(delta_df.to_string(index=False))

delta_q_df   = build_delta_quantile_summary_table(all_data)
delta_q_path = os.path.join(SAVE_DIR, 'delta_quantile_summary.csv')
delta_q_df.to_csv(delta_q_path, index=False)
print(f'\nDelta-by-quantile table → {delta_q_path}')

print()
for cell_line in CONDITIONS:
    group_data, delta_data, delta_by_q, _ = all_data[cell_line]
    plot_comparison(cell_line, group_data,
                    os.path.join(SAVE_DIR, f'{cell_line}_comparison.png'))
    plot_delta(cell_line, delta_data,
               os.path.join(SAVE_DIR, f'{cell_line}_delta.png'))
    plot_delta_by_quantile(cell_line, delta_by_q,
                           os.path.join(SAVE_DIR, f'{cell_line}_delta_by_quantile.png'))

print('\nDone.')
