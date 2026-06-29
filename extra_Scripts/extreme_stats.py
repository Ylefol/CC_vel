#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 09:28:34 2025

@author: yohanl
"""

import pandas as pd
import numpy as np
from snake_scripts.snake_functions import snake_utils as my_utils
from snake_scripts.snake_functions import snake_analysis_functions as my_func


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

my_iter_dict=my_utils.create_iter_dict(cell_line,replicates,number_of_iterations)


import time
start_time = time.time()
ext_df=my_func.extreme_value_testing(my_iter_dict,my_ranked_genes,mean_dict,CI_dict,boundary_dict,vlm_dict)
print("--- %s seconds ---" % (time.time() - start_time))

ext_df.to_csv('extreme_res_bis_2.csv')

