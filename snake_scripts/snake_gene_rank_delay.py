#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 11:02:58 2021

@author: yohanl
"""

from snake_functions import snake_utils as my_utils
from snake_functions import snake_analysis_functions as my_func
import sys
import os
import numpy as np
import pandas as pd


cell_line=(sys.argv[1].split('/')[2])
folder_to_use=(sys.argv[1].split('/')[4])

number_of_iterations=int(sys.argv[2])
#Find replicates
replicates=folder_to_use.split('_')

#Create layers
layers=['spliced','unspliced']


#Load Sc data for the gene ranking
Sc_dict={}
for layer in layers:
    Sc_dict[layer]=pd.read_csv('data_files/confidence_intervals/'+cell_line+'/merged_results/'+folder_to_use+'/'+layer+'/combined_stdev.csv')

mean_dict,CI_dict,bool_dict,count_dict,boundary_dict=my_utils.get_CI_data (cell_line, layers, folder_to_use)

my_ranked_genes=my_func.create_gene_ranking(count_dict,mean_dict,CI_dict,Sc_dict,number_of_iterations)

vlm_dict=my_utils.get_vlm_values(cell_line, layers,folder_to_use )
t_test_res=my_func.create_t_test_rank_method(my_ranked_genes,number_of_iterations,replicates,mean_dict,CI_dict,boundary_dict,vlm_dict)

# my_start_phases=identify_vel_start_phases=my_func.identify_vel_start_phases(count_dict,mean_dict,CI_dict,boundary_dict)

#Identify significant genes
# significant_genes=list(my_ranked_genes['gene_name'][np.where(np.asanyarray(my_ranked_genes['high_score'])!=0)[0]])

#Retrieve expression values
# df_dict_merged=my_utils.get_vlm_values(cell_line, layers,target_folder=folder_to_use)
#Calculate peaks based on spliced values
# peak_df=my_func.calculate_peak_dict(df_dict_merged,significant_genes,boundary_dict)

my_delay_df=my_func.create_delay_dataframe(count_dict)


my_ranked_genes.to_csv(sys.argv[3],index=False)
my_delay_df.to_csv(sys.argv[4],index=False)
t_test_res.to_csv(sys.argv[5],index=False)

# t_test_res.to_csv('data_files/data_results/rank/'+cell_line+'/'+folder_to_use+'_t_test_results.csv')



