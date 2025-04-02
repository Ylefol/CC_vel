#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:03:56 2023

@author: yohanl
"""

"""Load necessary modules and functions"""


#Import required functions
from snake_scripts.snake_functions import snake_utils as my_utils
from snake_scripts.snake_functions import snake_analysis_functions as my_func


#Import libraries
import os
import numpy as np
import pandas as pd


"""State desired cell line to plot along with additional information on save paths"""

#Set-up for the desired cell line and set of results
cell_line='HaCat'
replicates=os.listdir('data_files/confidence_intervals/'+cell_line)
replicates.remove('merged_results')
#Create layers, declare folder to use (either single replicate or merged replicates)
layers=['spliced','unspliced']
folder_to_use='X'
orientation='G2M'

"""Fetch the necessary files"""

# Load various results
mean_dict,CI_dict,bool_dict,count_dict,boundary_dict=my_utils.get_CI_data (cell_line, layers, folder_to_use)
my_ranked_genes=pd.read_csv('data_files/data_results/rank/'+cell_line+'/'+folder_to_use+'_ranked_genes.csv')
my_delay_df=pd.read_csv('data_files/data_results/delay_genes/'+cell_line+'/'+folder_to_use+'_delay_genes.csv')
vlm_dict=my_utils.get_vlm_values(cell_line, layers,folder_to_use)
vlm_mean_dict=my_utils.get_vlm_values(cell_line, layers,folder_to_use,get_mean=True)


#Find rankable genes and perform t-test statistic
rankable_genes=list(my_ranked_genes['gene_name'][np.where(np.asanyarray(my_ranked_genes['high_score'])>0)[0]])
t_test_res=pd.read_csv('data_files/data_results/rank/'+cell_line+'/'+folder_to_use+'_t_test_results.csv')

#Identify genes with padjusted < 0.01
#Remove NA from list, identify significant values, then significant genes
res = [i for i in t_test_res.padjusted if i != 'NA']
good_vals=[x for x in res if x<0.01]
significant_genes=list(t_test_res.loc[t_test_res['padjusted'] .isin(good_vals)].gene_name)




"""Plots for single genes - two versions exists, each illustrating gene velocity"""

#Print out the 'UNG' gene
if 'TOP2A'in mean_dict['spliced'].keys():
    gene_save_path='all_figures/'+cell_line+'/analysis_results/'+folder_to_use+'/gene_plots/'
    my_func.plot_layer_smooth_vel('TOP2A', mean_dict, bool_dict, CI_dict, count_dict,vlm_dict,
                                  vlm_mean_dict,boundary_dict,cell_line,save_path=gene_save_path+'layer_vel')
    my_func.plot_curve_count('TOP2A', mean_dict, bool_dict, CI_dict, count_dict,boundary_dict,
                             cell_line,save_path=gene_save_path+'vel_count')


#Print out the 'UNG' gene
if 'UNG'in mean_dict['spliced'].keys():
    gene_save_path='all_figures/'+cell_line+'/analysis_results/'+folder_to_use+'/gene_plots/'
    my_func.plot_layer_smooth_vel('UNG', mean_dict, bool_dict, CI_dict, count_dict,vlm_dict,
                                  vlm_mean_dict,boundary_dict,cell_line,save_path=gene_save_path+'layer_vel')
    my_func.plot_curve_count('UNG', mean_dict, bool_dict, CI_dict, count_dict,boundary_dict,
                             cell_line,save_path=gene_save_path+'vel_count')

"""Plots (Raincloud) for gene cell cycle delay"""

delay_save_path='all_figures/'+cell_line+'/analysis_results/'+folder_to_use+'/gene_delays'
#Plot delay of rankable genes
my_rankable_delays=my_delay_df[my_delay_df["gene_name"].isin(rankable_genes)]
my_func.plot_raincloud_delay(my_rankable_delays,cell_line,save_path=delay_save_path,save_name='delay_rankable_genes')

#Subset the delay dataframe with significant genes and plot
sig_delay=my_delay_df[my_delay_df['gene_name'] .isin(significant_genes)]
my_func.plot_raincloud_delay(sig_delay,cell_line,save_path=delay_save_path,save_name='delay_001_genes')

#Comparative raincloud plot
my_func.raincloud_delay_two_files_two_categories(my_rankable_delays,sig_delay,'inc_to_+1','dec_to_0',cell_line+'_custom_delay.png')


"""Plots the pseudotime trajectory of specific REACTOME pathways *can be RAM heavy*"""
#Plot for all genes
REAC_save_path='all_figures/'+cell_line+'/analysis_results/'+folder_to_use+'/REAC_rankable'
REAC_dict=my_func.create_REAC_dict(vlm_mean_dict,rankable_genes,orientation=orientation)
my_func.create_REAC_summary_plots(REAC_dict,boundary_dict,layer='spliced',second_layer='unspliced',plot_path=REAC_save_path,orientation=orientation)

#Plot for significant genes
REAC_save_path='all_figures/'+cell_line+'/analysis_results/'+folder_to_use+'/REAC_significant'
REAC_dict=my_func.create_REAC_dict(vlm_mean_dict,significant_genes,orientation=orientation)
my_func.create_REAC_summary_plots(REAC_dict,boundary_dict,layer='spliced',second_layer='unspliced',plot_path=REAC_save_path,orientation=orientation)


"""A plot which shows matching points between velocity and expression peaks for individual genes"""


my_func.plot_spliced_velocity_expression_zero_point(mean_dict,CI_dict,bool_dict,vlm_mean_dict,boundary_dict,'UNG',plot_name='UNG_1.png')

my_func.plot_velocity_expression_zero_point(mean_dict,CI_dict,bool_dict,vlm_dict,vlm_mean_dict,boundary_dict,'TOP2A',plot_name='TOP2A_2.png')
