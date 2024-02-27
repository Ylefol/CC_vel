#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 09:00:22 2024

@author: yohanl
"""


#Import required functions
from snake_scripts.snake_functions import snake_utils as my_utils
from snake_scripts.snake_functions import snake_analysis_functions as my_func

from snake_scripts.snake_functions import snake_analysis_miRNA_functions as my_miR_func

import os
import numpy as np
import pandas as pd

cell_line='jurkat'
replicates=['A','B','C','D']
layers=['spliced','unspliced'] 
target_folder='A_B_C_D'
variance_param='All'
miRNA_thresh_list=['1000_None','100_1000','0_100']
save_path='test'
gene_selection='all'
single_rep=False


miR_dta_dict={}

#Define save_path if not done by user
if save_path=='' and single_rep==True:
    save_path='all_figures/'+cell_line+'/single_replicate_analysis/'+target_folder+'/miRNA_boxplot'
elif save_path=='' and single_rep==False:
    save_path='all_figures/'+cell_line+'/merged_replicate_analysis/'+target_folder+'/miRNA_boxplot'

#Target scanfile
TS_path='data_files/miRNA_files/TS_files/Predicted_Targets_Context_Scores.default_predictions.txt'

#Get main data
mean_dict,CI_dict,bool_dict,count_dict,boundary_dict=my_utils.get_CI_data(cell_line,layers,target_folder,gene_selection)

#Get vlm data
exp_values=my_utils.get_vlm_values(cell_line,layers,target_folder,get_mean=False)

#Get miR data
gene_dict,variance_dict=my_miR_func.prep_variances_for_TS(exp_values,boundary_dict,keys_to_use=variance_param)

#Convert to proper format
variance_df = pd.DataFrame.from_dict(variance_dict)
variance_df['gene_names']=variance_df.index

#Ensures that the order is the same as in gene_dict - this is very important
first_gene_dict_key=list(gene_dict.keys())[0]
true_sort = [s for s in gene_dict[first_gene_dict_key] if s in variance_df.gene_names.unique()]
variance_df = variance_df.set_index('gene_names').loc[true_sort].reset_index()

#Create dictionnary for plotting purposes
gene_exclusion_list=[]
for miRNA_thresh in miRNA_thresh_list:
    miR_dta_dict[miRNA_thresh]={}
    
    #Perform miRNA analysis
    path_miRNA='data_files/miRNA_files/categorized/'+cell_line+'_miRNA_'+str(miRNA_thresh)+'.csv'
    miRNA_pd=pd.read_csv(path_miRNA)
    miRNA_list=list(miRNA_pd.found)
    TS_dict,TS_weight_dict,sum_weight_dict,miRNA_df,text_res=my_miR_func.target_scan_analysis(TS_path,gene_dict=gene_dict,cell_line=cell_line,miR_thresh=miRNA_thresh,miRNA_list=miRNA_list)
    
    #Remove genes in necessary
    if len(gene_exclusion_list) != 0:
        gene_removal_values=variance_df.gene_names.isin(gene_exclusion_list).value_counts()
        for my_bool, cnts in gene_removal_values.items():
            if my_bool==True:
                text_res=text_res+"Removing "+str(cnts)+" genes using the provided exclusion list"
        good_idx=list(np.where(~variance_df["gene_names"].isin(gene_exclusion_list)==True)[0])
        variance_df=variance_df.iloc[good_idx]
        
        #Subset_sum_weight_dict
        for key in sum_weight_dict.keys():
            sum_weight_dict[key] = [sum_weight_dict[key][i] for i in good_idx]
    
    
    miR_dta_dict[miRNA_thresh]['sum_weight_dict']=sum_weight_dict

    
    #Update gene exclusion list
    result_df=variance_df.copy()
    result_df['weight']=sum_weight_dict[list(sum_weight_dict.keys())[0]]
    miR_col=[]
    for gene in result_df['gene_names']:
        miRNA_idx=list(np.where(miRNA_df['Gene Symbol']==gene)[0])
        miRNA_df_subset=miRNA_df.loc[miRNA_idx]
        miR_slash_list = '/'.join(miRNA_df_subset["miRNA"])
        miR_col.append(miR_slash_list)
    result_df['miRNAs']=miR_col
    miR_dta_dict[miRNA_thresh]['variance_df']=result_df
    idx_genes_with_weights=list(np.where(result_df.weight!=0)[0])
    new_gene_exclusion=result_df.iloc[idx_genes_with_weights]
    new_gene_exclusion=list(new_gene_exclusion.gene_names)
    
    if len(gene_exclusion_list) == 0:
        gene_exclusion_list=new_gene_exclusion
    else:
        for gene in new_gene_exclusion:
            gene_exclusion_list.append(gene)
            
            
    #For the particular threshold, get the genes in each miRNA score group
    for layer in layers:
        miR_dta_dict[miRNA_thresh][layer]={}
        target_genes=result_df['gene_names'].iloc[np.where(result_df['weight']==0)[0]]
        miR_dta_dict[miRNA_thresh][layer]['weight_0']=exp_values[layer][target_genes]
        
        target_genes=result_df['gene_names'].iloc[np.where((result_df['weight']<0)&(result_df['weight']>=-0.3))[0]]
        miR_dta_dict[miRNA_thresh][layer]['weight_>-0.3']=exp_values[layer][target_genes]
        
        target_genes=result_df['gene_names'].iloc[np.where(result_df['weight']<-0.3)[0]]
        miR_dta_dict[miRNA_thresh][layer]['weight_<-0.3']=exp_values[layer][target_genes]
        


target_key='All'
thresh_order=['1000_None','100_1000','0_100']
for miR_thresh in miRNA_thresh_list:
    my_miR_func.create_miRweight_boxplot(vari_df=miR_dta_dict[miR_thresh]['variance_df'],
                                         weight_dict=miR_dta_dict[miR_thresh]['sum_weight_dict'], 
                                         target_key=target_key, cell_line=cell_line,
                                         plot_name='miR_All',miR_thresh=miR_thresh,
                                         delay_cutoff=None,save_path='miR_box/'+cell_line)
    
    my_miR_func.expression_bar_plot(miR_dta_dict,miR_thresh,'spliced','miR_box/'+cell_line)
    my_miR_func.expression_bar_plot(miR_dta_dict,miR_thresh,'unspliced','miR_box/'+cell_line)

#PLOT CONCATENATION
import matplotlib.pyplot as plt
#Create the figure
fig, axes = plt.subplots(3, 3,figsize=(25, 10))
# thresh_order.reverse()
for idx,val in enumerate(thresh_order):
    
    ax = axes[0,idx]
    ax.axis('off')
    image = plt.imread('miR_box/'+cell_line+'/'+str(val)+'/miR_All.png')
    ax.imshow(image)
    ax = axes[1, idx]
    ax.axis('off')
    image = plt.imread('miR_box/'+cell_line+'/'+str(val)+'/spliced_All.png')
    ax.imshow(image)
    ax = axes[2, idx]
    ax.axis('off')
    image = plt.imread('miR_box/'+cell_line+'/'+str(val)+'/unspliced_All.png')
    ax.imshow(image)  
        
fig.tight_layout()
plt.savefig('miR_box/'+cell_line+'/Concatenation_All.png')
plt.show()

