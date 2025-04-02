#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 09:56:42 2025

@author: yohanl
"""

from snake_scripts.snake_functions import snake_utils as my_utils
from snake_scripts.snake_functions import snake_analysis_functions as my_func
import sys
import os
import numpy as np
import pandas as pd


cell_line='HaCat'
folder_to_use='A_B'

number_of_iterations=5
#Find replicates
replicates=folder_to_use.split('_')

#Create layers
layers=['spliced','unspliced']


#Load Sc data for the gene ranking
Sc_dict={}
for layer in layers:
    Sc_dict[layer]=pd.read_csv('data_files/confidence_intervals/'+cell_line+'/merged_results/'+folder_to_use+'/'+layer+'/vel_combined_stdev.csv')

mean_dict,CI_dict,bool_dict,count_dict,boundary_dict=my_utils.get_CI_data (cell_line, layers, folder_to_use)

my_ranked_genes=my_func.create_gene_ranking(count_dict,mean_dict,CI_dict,Sc_dict,number_of_iterations)

vlm_dict=my_utils.get_vlm_values(cell_line, layers,folder_to_use)

# t_test_res=my_func.create_t_test_rank_method(my_ranked_genes,number_of_iterations,replicates,mean_dict,CI_dict,boundary_dict,vlm_dict)


layer='spliced'
CI='low_CI'

from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
from rpy2.robjects.vectors import BoolVector
stats = importr('stats')

#Param set-up
iterations_done=number_of_iterations*len(replicates)
deg_of_freedom=iterations_done-1
layer_CI=layer+'_'+CI

# Remove all zeros (non-ranked genes)
gene_input_list=my_ranked_genes[layer_CI][my_ranked_genes[layer_CI]!=0]

#Calculate p_value
my_pvals=list(stats.pt(FloatVector(gene_input_list),df=deg_of_freedom,lower_tail=BoolVector([False])))

my_pvals = []
# to test indiv genes we need to go one gene at a time and compare it with a reference. Hence the loop
# Giving the entire vector will only provide a pvalue which determines if the whole vector is significantly
# away from 0
for i, value in enumerate(gene_input_list):
    # Define the test value and reference group (excluding the current value)
    test_value = FloatVector([value])
    reference_group = FloatVector(np.delete(gene_input_list, i))  # Remove current value

    # Perform Wilcoxon Rank-Sum Test
    wilcox_result = stats.wilcox_test(test_value, reference_group,alternative='two.sided')
    p_value = wilcox_result.rx2("p.value")[0]  # Extract p-value (called rx2)
    my_pvals.append(p_value)

    #Calculate p_adjsuted values
    num_genes=len(my_pvals)
    p_adjust = list(stats.p_adjust(FloatVector(my_pvals),n=num_genes, method = 'fdr'))

#Adjust length for non-ranked genes
non_ranked=len(my_ranked_genes[layer_CI])-num_genes
non_ranked_lst=['NA']*non_ranked

#Create base dataframe with t value, p_value and padjusted value with associated
# gene names
dataframe_dict={}
dataframe_dict['t']=my_ranked_genes[layer_CI]
dataframe_dict['pvalue']=my_pvals+non_ranked_lst
dataframe_dict['padjusted']=p_adjust + non_ranked_lst
wilcox_df = pd.DataFrame(dataframe_dict)
wilcox_df.index=my_ranked_genes['gene_name']

# Find phases where layer CI begins and maxes out
found_phases=my_func.find_phase_association(wilcox_df,mean_dict,CI_dict,boundary_dict,vlm_dict,layer,CI)

#Add phases to df and return
wilcox_df['phase_peak_vel']=found_phases['peak_vel']
wilcox_df['phase_peak_exp']=found_phases['peak_exp']
wilcox_df['phase_start_vel']=found_phases['start_vel']



wilcox_df.to_csv('data_files/data_results/HaCat_A_B_wilcox.csv',index=True)

