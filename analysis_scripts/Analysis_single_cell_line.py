#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:03:56 2023

@author: yohanl
"""

#Import required functions
from snake_scripts.snake_functions import snake_utils as my_utils
from snake_scripts.snake_functions import snake_analysis_functions as my_func


#Import libraries
import os
import numpy as np
import pandas as pd


#Set-up for the desired cell line and set of results
cell_line='HaCat'
replicates=os.listdir('data_files/confidence_intervals/'+cell_line)
replicates.remove('merged_results')
#Create layers, declare folder to use (either single replicate or merged replicates)
layers=['spliced','unspliced']
folder_to_use='A_B'

# Load various results
mean_dict,CI_dict,bool_dict,count_dict,boundary_dict=my_utils.get_CI_data (cell_line, layers, folder_to_use)
my_ranked_genes=pd.read_csv('data_files/data_results/rank/'+cell_line+'/'+folder_to_use+'_ranked_genes.csv')
my_delay_df=pd.read_csv('data_files/data_results/delay_genes/'+cell_line+'/'+folder_to_use+'_delay_genes.csv')
# my_UTRs=pd.read_csv('data_files/data_results/UTR_length/'+cell_line+'/'+folder_to_use+'_UTR_length.csv')
vlm_dict=my_utils.get_vlm_values(cell_line, layers,folder_to_use )

#Find rankable genes and perform t-test statistic
rankable_genes=list(my_ranked_genes['gene_name'][np.where(np.asanyarray(my_ranked_genes['high_score'])>0)[0]])
t_test_res=pd.read_csv('data_files/data_results/rank/'+cell_line+'/'+folder_to_use+'_t_test_results.csv')

#Identify genes with padjusted < 0.01
#Remove NA from list, identify significant values, then significant genes
res = [i for i in t_test_res.padjusted if i != 'NA']
good_vals=[x for x in res if x<0.01]
significant_genes=list(t_test_res.loc[t_test_res['padjusted'] .isin(good_vals)].gene_name)


############## Plot gene velocities
#Print out the 'UNG' gene
if 'UNG'in mean_dict['spliced'].keys():
    gene_save_path='all_figures/'+cell_line+'/analysis_results/'+folder_to_use+'/gene_plots/'
    my_func.plot_layer_smooth_vel('UNG', mean_dict, bool_dict, CI_dict, count_dict,vlm_dict,boundary_dict,cell_line,save_path=gene_save_path+'layer_vel')
    my_func.plot_curve_count('UNG', mean_dict, bool_dict, CI_dict, count_dict,boundary_dict,cell_line,save_path=gene_save_path+'vel_count')


############## Plot gene delays
delay_save_path='all_figures/'+cell_line+'/analysis_results/'+folder_to_use+'/gene_delays'
#Plot delay of all genes
my_func.plot_raincloud_delay(my_delay_df,cell_line,save_path=delay_save_path,save_name='delay_all_genes')

#Plot delay of rankable genes
my_significant_delays=my_delay_df[my_delay_df["gene_name"].isin(rankable_genes)]
my_func.plot_raincloud_delay(my_significant_delays,cell_line,save_path=delay_save_path,save_name='delay_rankable_genes')

#Subset the delay dataframe with significant genes and plot
sub_delay=my_delay_df[my_delay_df['gene_name'] .isin(significant_genes)]
my_func.plot_raincloud_delay(sub_delay,cell_line,save_path=delay_save_path,save_name='delay_001_genes')


############## Raincloud plots for gene expression per cell cycle phase
genes_HaCat=pd.read_csv('data_files/data_results/rank/HaCat/A_B_t_test_results.csv')

g1_group=genes_HaCat.gene_name[genes_HaCat.phase_peak_exp=='G1']
s_group=genes_HaCat.gene_name[genes_HaCat.phase_peak_exp=='S']
g2m_group=genes_HaCat.gene_name[genes_HaCat.phase_peak_exp=='G2M']

gene_matrix=pd.read_csv('data_files/confidence_intervals/HaCat/merged_results/A_B/spliced/vlm_means.csv')

my_func.plot_phase_exp_raincloud(gene_matrix,g1_group,boundary_dict,'G1 genes','analysis_results/g1_genes.png')
my_func.plot_phase_exp_raincloud(gene_matrix,s_group,boundary_dict,'S genes','analysis_results/s_genes.png')
my_func.plot_phase_exp_raincloud(gene_matrix,g2m_group,boundary_dict,'G2M genes','analysis_results/g2m_genes.png')
