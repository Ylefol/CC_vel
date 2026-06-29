#!/usr/bin/env python3
"""
create_miRNA_boxplot.py

Grouped boxplot / line+ribbon plot comparing miRNA targeting strength
across quantile bins, loaded from pre-computed TargetScan CSVs produced
by precompute_targetscan.py.

X-axis : four quantiles (0-25, 25-50, 50-75, 75-100), binned on spliced_low_CI
Y-axis : variance ratio (spliced_var / unspliced_var)
Colors : four groups formed by crossing two RPM tiers × two score categories

Group definitions (mirrors the original pickle-based logic)
-----------------------------------------------------------
  '1000_None | < -0.3' : strong targets of highly-expressed miRNAs
  '1000_None | 0'      : genes with no targets from highly-expressed miRNAs
                         (rpm_tier in ['0_100', 'none'])
  '0_100     | < -0.3' : strong targets of lowly-expressed miRNAs only
  '0_100     | 0'      : genes with no miRNA targets at any tier (rpm_tier == 'none')
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUANTILES = ['0-25', '25-50', '50-75', '75-100']

# (rpm_tier_label, score_key, legend_label, fill_color)
GROUPS = [
    ('1000_None', '< -0.3', '1000_None  |  < -0.3', '#8B0000'),
    ('1000_None', '0',      '1000_None  |  0',       '#F4A582'),
    ('0_100',     '< -0.3', '0_100      |  < -0.3',  '#08519C'),
    ('0_100',     '0',      '0_100      |  0',        '#9ECAE1'),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _group_mask(df, rpm_tier_label, score_key):
    """Boolean mask selecting genes belonging to one GROUPS entry."""
    if rpm_tier_label == '1000_None':
        if score_key == '< -0.3':
            return (df['rpm_tier'] == '1000_None') & (df['weight'] < -0.3)
        else:  # '0' — no targets from high-RPM miRNAs
            return df['rpm_tier'].isin(['0_100', 'none'])
    else:  # '0_100'
        if score_key == '< -0.3':
            return (df['rpm_tier'] == '0_100') & (df['weight'] < -0.3)
        else:  # '0' — no targets at any tier
            return df['rpm_tier'] == 'none'


def _load_group_values(data, do_log10=False):
    """
    Build group_values from pre-computed TargetScan CSV + ranked genes CSV.

    Parameters
    ----------
    data : tuple (cell_line, rep)
        e.g. ('HaCat-Control', 'A_B')
    do_log10 : bool
        Apply log10 to variance_ratio values before returning.

    Returns
    -------
    cell_key : str
        '{cell_line}_{rep}' label used as plot title fallback.
    group_values : list[list[list[float]]]
        Indexed as group_values[group_idx][quantile_idx].
    """
    cell_line, rep = data

    ts_path = (f'data_files/miRNA_files/precomputed/'
               f'{cell_line}_{rep}_targetscan.csv')
    ts_df = pd.read_csv(ts_path)

    ranked_path = f'data_files/data_results/rank/{cell_line}/{rep}_ranked_genes.csv'
    ranked_df   = pd.read_csv(ranked_path)
    ranked_df   = ranked_df[ranked_df['high_score'] > 0].copy()

    # Quantile bins based on spliced_low_CI (consistent with original pipeline)
    q = ranked_df['spliced_low_CI'].quantile([0.25, 0.5, 0.75])
    conditions = [
        ranked_df['spliced_low_CI'] <= q[0.25],
        (ranked_df['spliced_low_CI'] > q[0.25]) & (ranked_df['spliced_low_CI'] <= q[0.50]),
        (ranked_df['spliced_low_CI'] > q[0.50]) & (ranked_df['spliced_low_CI'] <= q[0.75]),
        ranked_df['spliced_low_CI'] > q[0.75],
    ]
    ranked_df['quantile_bin'] = np.select(conditions, QUANTILES)

    merged = ts_df.merge(ranked_df[['gene_name', 'quantile_bin']], on='gene_name', how='inner')

    group_values = []
    for rpm_tier_label, score_key, _, _ in GROUPS:
        per_quantile = []
        for q_bin in QUANTILES:
            mask = (merged['quantile_bin'] == q_bin) & _group_mask(merged, rpm_tier_label, score_key)
            vals = merged.loc[mask, 'variance_ratio'].dropna().tolist()
            if do_log10:
                arr      = np.array(vals, dtype=float)
                n_drop   = (arr <= 0).sum()
                if n_drop:
                    print(f'  Warning: dropping {n_drop} value(s) ≤ 0 before log10 '
                          f'({rpm_tier_label} | {score_key}, {q_bin})')
                vals = np.log10(arr[arr > 0]).tolist()
            per_quantile.append(vals)
        group_values.append(per_quantile)

    return f'{cell_line}_{rep}', group_values


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------

def _draw_boxes(ax, group_values, xlocations, box_width, box_spacing, include_fliers):
    n_g = len(GROUPS)
    for g_idx, (vals_per_q, (_, _, _, color)) in enumerate(zip(group_values, GROUPS)):
        offset    = (g_idx - n_g / 2 + 0.5) * (box_width + box_spacing)
        positions = [x + offset for x in xlocations]
        ax.boxplot(
            vals_per_q,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            boxprops=dict(facecolor=color, linewidth=0.8),
            medianprops=dict(color='black', linewidth=1.2),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
            flierprops=dict(marker='.', markersize=2, alpha=0.5),
            showfliers=include_fliers,
            labels=[''] * len(QUANTILES),
        )


def _legend_handles():
    return [
        mpatches.Patch(facecolor=color, edgecolor='grey', linewidth=0.5, label=label)
        for _, _, label, color in GROUPS
    ]


# ---------------------------------------------------------------------------
# Single-dataset boxplot
# ---------------------------------------------------------------------------

def plot_miRNA_boxplot_v3(
        data,
        save_name=None,
        ylabel='variance_ratio',
        plot_title=None,
        include_fliers=False,
        do_log10=False,
        box_width=0.15,
        box_spacing=0.04,
):
    """
    Grouped boxplot for one sample.

    Parameters
    ----------
    data : tuple (cell_line, rep)
    save_name : str or None
    ylabel, plot_title, include_fliers, do_log10, box_width, box_spacing : see module docstring
    """
    cell_key, group_values = _load_group_values(data, do_log10)

    if plot_title is None:
        plot_title = cell_key

    n_q         = len(QUANTILES)
    n_g         = len(GROUPS)
    group_span  = n_g * box_width + (n_g - 1) * box_spacing
    xlocations  = np.arange(n_q) * (group_span + 0.35)

    mpl.rcParams['figure.dpi'] = 600
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.grid(True, linestyle='dotted')
    ax.set_axisbelow(True)

    _draw_boxes(ax, group_values, xlocations, box_width, box_spacing, include_fliers)

    ax.set_xticks(xlocations)
    ax.set_xticklabels(QUANTILES, fontsize=10)
    ax.set_xlabel('Quantile (spliced low CI)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(plot_title, fontsize=12)
    ax.legend(
        handles=_legend_handles(),
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        title='RPM threshold  |  Score',
        fontsize=9,
        title_fontsize=9,
    )

    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
        plt.close('all')
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Faceted wrapper
# ---------------------------------------------------------------------------

def plot_miRNA_boxplot_facets(
        datasets,
        save_name=None,
        ylabel='variance_ratio',
        include_fliers=False,
        do_log10=False,
        box_width=0.15,
        box_spacing=0.04,
):
    """
    Vertically stacked facets, one per dataset, with a shared y-axis.

    Parameters
    ----------
    datasets : dict {title: (cell_line, rep)} or list of (title, (cell_line, rep))
    """
    items = list(datasets.items()) if isinstance(datasets, dict) else list(datasets)

    n           = len(items)
    n_q         = len(QUANTILES)
    n_g         = len(GROUPS)
    group_span  = n_g * box_width + (n_g - 1) * box_spacing
    xlocations  = np.arange(n_q) * (group_span + 0.35)

    mpl.rcParams['figure.dpi'] = 600
    fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), sharey=True, sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (title, data) in zip(axes, items):
        _, group_values = _load_group_values(data, do_log10)
        ax.grid(True, linestyle='dotted')
        ax.set_axisbelow(True)
        _draw_boxes(ax, group_values, xlocations, box_width, box_spacing, include_fliers)
        ax.set_xticks(xlocations)
        ax.set_xticklabels(QUANTILES, fontsize=10)
        ax.set_title(title, fontsize=11)

    axes[-1].set_xlabel('Quantile (spliced low CI)', fontsize=11)
    fig.supylabel(ylabel, fontsize=11)
    fig.legend(
        handles=_legend_handles(),
        loc='center right',
        bbox_to_anchor=(1.15, 0.5),
        title='RPM threshold  |  Score',
        fontsize=9,
        title_fontsize=9,
    )

    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
        plt.close('all')
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Line + ribbon plot
# ---------------------------------------------------------------------------

def plot_miRNA_lineplot_v3(
        data,
        save_name=None,
        ylabel='variance_ratio',
        plot_title=None,
        do_log10=False,
        use_mean=False,
        ribbon_alpha=0.2,
):
    """
    Line + IQR ribbon plot for one sample.

    Parameters
    ----------
    data : tuple (cell_line, rep)
    use_mean : bool
        True → mean ± std;  False (default) → median with IQR ribbon.
    """
    cell_key, group_values = _load_group_values(data, do_log10)

    if plot_title is None:
        plot_title = cell_key

    x = np.arange(len(QUANTILES))

    mpl.rcParams['figure.dpi'] = 600
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.grid(True, linestyle='dotted')
    ax.set_axisbelow(True)

    for g_idx, (_, _, label, color) in enumerate(GROUPS):
        centres, lowers, uppers = [], [], []

        for q_idx in range(len(QUANTILES)):
            vals = np.array(group_values[g_idx][q_idx], dtype=float)

            if len(vals) == 0:
                centres.append(np.nan)
                lowers.append(np.nan)
                uppers.append(np.nan)
                continue

            if use_mean:
                sd = np.std(vals, ddof=1)
                centres.append(np.mean(vals))
                lowers.append(np.mean(vals) - sd)
                uppers.append(np.mean(vals) + sd)
            else:
                centres.append(np.median(vals))
                lowers.append(np.percentile(vals, 25))
                uppers.append(np.percentile(vals, 75))

        centres = np.array(centres)
        lowers  = np.array(lowers)
        uppers  = np.array(uppers)

        ax.plot(x, centres, color=color, linewidth=1.8, label=label, marker='o', markersize=4)
        ax.fill_between(x, lowers, uppers, color=color, alpha=ribbon_alpha)

    ax.set_xticks(x)
    ax.set_xticklabels(QUANTILES, fontsize=10)
    ax.set_xlabel('Quantile (spliced low CI)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(plot_title, fontsize=12)
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        title='RPM threshold  |  Score',
        fontsize=9,
        title_fontsize=9,
    )

    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
        plt.close('all')
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    os.chdir('/media/yohanlefol/Expansion/CC_vel/')

    hacat_datasets = {
        'HaCat-Control':          ('HaCat-Control',          'A_B'),
        'HaCat-CRISPRi-Dicer-EXP': ('HaCat-CRISPRi-Dicer-EXP', 'A_B'),
        'HaCat-CRISPRi-Dicer-KO':  ('HaCat-CRISPRi-Dicer-KO',  'A_B'),
        'HaCat-CRISPRi-AGO2-EXP':  ('HaCat-CRISPRi-AGO2-EXP',  'A_B'),
        'HaCat-CRISPRi-AGO2-KO':   ('HaCat-CRISPRi-AGO2-KO',   'A_B'),
    }
    plot_miRNA_boxplot_facets(hacat_datasets, save_name='HaCat_facets.png',do_log10=True)

    a549_datasets = {
        'A549-Control':           ('A549-Control',           'A_B'),
        'A549-CRISPRi-Dicer-EXP': ('A549-CRISPRi-Dicer-EXP', 'A_B'),
        'A549-CRISPRi-Dicer-KO':  ('A549-CRISPRi-Dicer-KO',  'A_B'),
        'A549-CRISPRi-AGO2-EXP':  ('A549-CRISPRi-AGO2-EXP',  'A_B'),
        'A549-CRISPRi-AGO2-KO':   ('A549-CRISPRi-AGO2-KO',   'A_B'),
    }
    plot_miRNA_boxplot_facets(a549_datasets, save_name='A549_facets.png',do_log10=True)

    plot_miRNA_boxplot_v3(
        data=('HaCat-Control', 'A_B'),
        save_name='HaCat-Control_A_B.png',
        plot_title='HaCat-Control A_B',
        do_log10=True
    )
