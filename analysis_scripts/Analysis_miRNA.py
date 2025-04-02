#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 10:51:51 2022

@author: yohanl
"""


#Import required functions
from snake_scripts.snake_functions import snake_utils as my_utils
from snake_scripts.snake_functions import snake_analysis_functions as my_func

from snake_scripts.snake_functions import snake_analysis_miRNA_functions as my_miR_func

import os
import numpy as np
import pandas as pd

cell_line='293t'
# cell_line=key
#Find replicates
replicates=os.listdir('data_files/confidence_intervals/'+cell_line)
replicates.remove('merged_results')
#Create layers
layers=['spliced','unspliced']


folder_to_use='A_B_C_D'

mean_dict,CI_dict,bool_dict,count_dict,boundary_dict=my_utils.get_CI_data (cell_line, layers, folder_to_use)
my_ranked_genes=pd.read_csv('data_files/data_results/rank/'+cell_line+'/'+folder_to_use+'_ranked_genes.csv')
rankable_genes=list(my_ranked_genes['gene_name'][np.where(np.asanyarray(my_ranked_genes['high_score'])>0)[0]])




# plot miRNA boxplots for rankable genes
#################
miRNA_thresh_list=['1000_None','100_1000','0_100']
save_path='all_figures/'+cell_line+'/analysis_results/'+folder_to_use+'/miRNA_rankable_genes_plots'
my_miR_func.wrapper_miRNA_boxplot_analysis(cell_line,replicates,layers,folder_to_use,'Both',miRNA_thresh_list,save_path,gene_selection=rankable_genes,single_rep=False)
# for rep in replicates:
#     my_func.wrapper_miRNA_boxplot_analysis(cell_line,replicates,layers,rep,'Both',miRNA_thresh_list,save_path,single_rep=True)

# plot miRNA boxplots for cc genes
#################
cc_genes_df=pd.read_csv('data_files/initial_data/Original_cell_cycle_genes_with_new_candidates.csv')
cc_genes_df['phase'] = cc_genes_df['phase'].map({'G2/M': 'G2M', 'M/G1': 'M_G1','G1/S':'G1_S','S':'S','G1':'G1','G2':'G2'})
all_cc_genes=list(cc_genes_df.gene)
cc_genes=[]
for gene in all_cc_genes:
    if gene in list(mean_dict['spliced'].keys()):
        cc_genes.append(gene)

miRNA_thresh_list=['1000_None','100_1000','0_100']
save_path='all_figures/'+cell_line+'/analysis_results/'+folder_to_use+'/miRNA_cc_genes_plots'
my_miR_func.wrapper_miRNA_boxplot_analysis(cell_line,replicates,layers,folder_to_use,'Both',miRNA_thresh_list,save_path,gene_selection=cc_genes,single_rep=False)
# for rep in replicates:
#     my_func.wrapper_miRNA_boxplot_analysis(cell_line,replicates,layers,rep,'Both',miRNA_thresh_list,save_path,single_rep=True)

# plot miRNA boxplots for significant genes (t_test + delay + var)
#################
significant_genes=my_func.get_sig_genes(cell_line,folder_to_use,t_test_based=True,delay_type='dec_to_0',variability_based=True)
miRNA_thresh_list=['1000_None','100_1000','0_100']
save_path='all_figures/'+cell_line+'/analysis_results/'+folder_to_use+'/miRNA_t_delay_var_genes_plots'
my_miR_func.wrapper_miRNA_boxplot_analysis(cell_line,replicates,layers,folder_to_use,'Both',miRNA_thresh_list,save_path,gene_selection=significant_genes,single_rep=False)


# plot miRNA boxplots for significant genes (delay)
#################
significant_genes=my_func.get_sig_genes(cell_line,folder_to_use,t_test_based=False,delay_type='dec_to_0',variability_based=False)
miRNA_thresh_list=['1000_None','100_1000','0_100']
save_path='all_figures/'+cell_line+'/analysis_results/'+folder_to_use+'/miRNA_delay_genes_plots'
my_miR_func.wrapper_miRNA_boxplot_analysis(cell_line,replicates,layers,folder_to_use,'Both',miRNA_thresh_list,save_path,gene_selection=significant_genes,single_rep=False)


# plot miRNA boxplots for significant genes (var)
#################
significant_genes=my_func.get_sig_genes(cell_line,folder_to_use,t_test_based=False,delay_type=None,variability_based=True)
miRNA_thresh_list=['1000_None','100_1000','0_100']
save_path='all_figures/'+cell_line+'/analysis_results/'+folder_to_use+'/miRNA_var_genes_plots'
my_miR_func.wrapper_miRNA_boxplot_analysis(cell_line,replicates,layers,folder_to_use,'Both',miRNA_thresh_list,save_path,gene_selection=significant_genes,single_rep=False)

# plot miRNA boxplots for significant genes (var)
#################
significant_genes=my_func.get_sig_genes(cell_line,folder_to_use,t_test_based=True,delay_type=None,variability_based=False)
miRNA_thresh_list=['1000_None','100_1000','0_100']
save_path='all_figures/'+cell_line+'/analysis_results/'+folder_to_use+'/miRNA_sig_t_genes_plots'
my_miR_func.wrapper_miRNA_boxplot_analysis(cell_line,replicates,layers,folder_to_use,'Both',miRNA_thresh_list,save_path,gene_selection=significant_genes,single_rep=False)

