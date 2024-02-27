#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:08:10 2023

@author: yohanl
"""

############## 3'UTR analysis and comparison with significant delays
#Import required functions
from snake_scripts.snake_functions import snake_analysis_miRNA_functions as my_func

#Import libraries
import pandas as pd

cell_line_dict={}
cell_line_dict['HaCat']='A_B'
cell_line_dict['jurkat']='A_B_C_D'
cell_line_dict['293t']='A_B_C_D'



#Calculate 3'UTR stuff
gtf_path='data_files/gencode.v33.annotation.gtf'

df_list=[]
for cell_line in list(cell_line_dict.keys()):
    folder_to_use=cell_line_dict[cell_line]
    
    t_test_res=pd.read_csv('data_files/data_results/rank/'+cell_line+'/'+folder_to_use+'_t_test_results.csv')
    res = [i for i in t_test_res.padjusted if i != 'NA']
    good_vals=[x for x in res if x<0.01]
    significant_genes=list(t_test_res.loc[t_test_res['padjusted'] .isin(good_vals)].gene_name)
    
    #Get delay information
    my_delay_df=pd.read_csv('data_files/data_results/delay_genes/'+cell_line+'/'+folder_to_use+'_delay_genes.csv')
    my_significant_delays=my_delay_df[my_delay_df['gene_name'] .isin(significant_genes)]
    
    my_UTRs=my_func.create_read_UTR_results(cell_line,folder_to_use,gtf_path,significant_genes)
    

    ############## Statistical comparison of 3'UTR results.
    for delay_cat in ['inc_to_+1','dec_to_0']:
        new_row=[]
        
        new_row.append(cell_line)
        new_row.append(delay_cat)
        # Spearmans for comparisons of delay with 3'UTR length for significant categories
        vals=my_func.spearman_comp_delay_with_UTR(my_UTRs,my_significant_delays,delay_cat=delay_cat)
        new_row.append(vals[0])
        new_row.append(vals[1])
        # MWU for comparison of UTR vs UTR
        vals=my_func.compare_UTR_based_on_delay(my_UTRs,my_significant_delays,delay_cat,10)
        new_row.append(vals[0])
        new_row.append(vals[1])
        vals=my_func.compare_UTR_based_on_delay(my_UTRs,my_significant_delays,delay_cat,-10)
        new_row.append(vals[0])
        new_row.append(vals[1])


        df_list.append(new_row)
        
        
col_names=['cell_lines','delay category','spearman_corr','spearman_pval','MWU_val_10','MWU_pval_10','MWU_val_-10','MWU_pval_-10']
UTR_df=pd.DataFrame(df_list,columns=col_names)
UTR_df.to_csv('data_files/data_results/UTR_delay_results.csv',index=False)