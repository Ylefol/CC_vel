#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 11:03:03 2025

@author: yohanl
"""
#Import required functions
from snake_scripts.snake_functions import snake_utils as my_utils
from snake_scripts.snake_functions import snake_analysis_functions as my_func

from snake_scripts.snake_functions import snake_analysis_miRNA_functions as my_miR_func

import os
import numpy as np
import pandas as pd

cell_line='HaCat'
# cell_line=key
#Find replicates
replicates=os.listdir('data_files/confidence_intervals/'+cell_line)
replicates.remove('merged_results')
#Create layers
layers=['spliced','unspliced']


folder_to_use='A_B'

mean_dict,CI_dict,bool_dict,count_dict,boundary_dict=my_utils.get_CI_data (cell_line, layers, folder_to_use)
my_ranked_genes=pd.read_csv('data_files/data_results/rank/'+cell_line+'/'+folder_to_use+'_ranked_genes.csv')
rankable_genes=list(my_ranked_genes['gene_name'][np.where(np.asanyarray(my_ranked_genes['high_score'])>0)[0]])

my_ranked_genes=my_ranked_genes.iloc[np.where(np.asanyarray(my_ranked_genes['high_score'])>0)[0]]
quantiles=my_ranked_genes.unspliced_low_CI.quantile([0,0.25,0.5,0.75,1])




# plot miRNA boxplots for rankable genes
save_stats=pd.DataFrame(None)
#################
for i in range(2):
    if i == 0:
        log_bool=True
        y_axis='log10(variance_ratio)'
    else:
        log_bool=False
        y_axis='variance_ratio'
        
    for z in range(5):
        if z==0:
            rankable_genes=list(my_ranked_genes.gene_name)
            quant=''
        elif z==1:
            rankable_genes=list(my_ranked_genes.gene_name[my_ranked_genes.unspliced_low_CI<=quantiles[0.25]])
            quant='0-25'
        elif z==2:
            rankable_genes=list(my_ranked_genes.gene_name[(my_ranked_genes.unspliced_low_CI>quantiles[0.25]) & (my_ranked_genes.unspliced_low_CI<=quantiles[0.50])])
            quant='25-50'
        elif z==3:
            rankable_genes=list(my_ranked_genes.gene_name[(my_ranked_genes.unspliced_low_CI>quantiles[0.50]) & (my_ranked_genes.unspliced_low_CI<=quantiles[0.75])])
            quant='50-75'
        elif z==4:
            rankable_genes=list(my_ranked_genes.gene_name[my_ranked_genes.unspliced_low_CI>quantiles[0.75]])
            quant='75-100'
        
        miRNA_thresh_list=['1000_None','100_1000','0_100']
        if quant=='':
            p_title='HaCaT with rankable genes'
            save_name='HaCaT_rankable'
        else:
            p_title='HaCaT with rankable genes quantile '+quant
            save_name='quant_'+quant
        if log_bool==True:
            save_name=save_name+'_log10'
            
        save_name=save_name + ' ('+str(len(rankable_genes))+')'
        #Function - retrieve data
        my_data=my_miR_func.miRNA_analysis_data_retrieval(cell_line,replicates,layers,folder_to_use,'All',
                                                               miRNA_thresh_list,do_log10=log_bool,delay_key=None,
                                                               gene_selection=rankable_genes,single_rep=False)
        
        
        #Finalize the plot
        my_miR_func.wrapper_miRNA_boxplot_analysis_V2(input_data=my_data,miRNA_thresh_list=miRNA_thresh_list,
                                                      save_name=save_name+'.png',target_key='All',
                                                      include_fliers=log_bool,ylabel=y_axis,
                                                      plot_title=p_title)
        
        
        #Get stat results
        found_stats=my_miR_func.calculate_siginificance_between_miRNA_groups(my_data,miRNA_thresh_list)

        if len(save_stats)==0:
            save_stats=pd.DataFrame.from_dict(found_stats,orient='index',columns=[save_name])
        else:
            temp=pd.DataFrame.from_dict(found_stats,orient='index',columns=[save_name])
            save_stats=pd.merge(save_stats,temp, left_index=True,right_index=True)
            
save_stats.to_csv('HaCaT_miRNA_quant_stats.csv')
            
            
            
