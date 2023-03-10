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

##############Plot REAC trajectory plots
#Plot for all genes
REAC_save_path='all_figures/'+cell_line+'/analysis_results/'+folder_to_use+'/REAC_rankable'
REAC_dict=my_func.create_REAC_dict(vlm_dict,rankable_genes)
my_func.create_REAC_summary_plots(REAC_dict,boundary_dict,layer='spliced',second_layer='unspliced',plot_path=REAC_save_path)

#Plot for significant genes
REAC_save_path='all_figures/'+cell_line+'/analysis_results/'+folder_to_use+'/REAC_significant'
REAC_dict=my_func.create_REAC_dict(vlm_dict,significant_genes)
my_func.create_REAC_summary_plots(REAC_dict,boundary_dict,layer='spliced',second_layer='unspliced',plot_path=REAC_save_path)



############## Raincloud plots for gene expression per cell cycle phase
genes_HaCat=pd.read_csv('data_files/data_results/rank/HaCat/A_B_t_test_results.csv')

g1_group=genes_HaCat.gene_name[genes_HaCat.phase_peak_exp=='G1']
s_group=genes_HaCat.gene_name[genes_HaCat.phase_peak_exp=='S']
g2m_group=genes_HaCat.gene_name[genes_HaCat.phase_peak_exp=='G2M']



import statistics
dta_res_list=[]
gene_matrix=vlm_dict['spliced'] 
log10_matrix=np.log10(gene_matrix)
for file_name in os.listdir('data_files/REAC_pathways'):
    gene_list=pd.read_csv('data_files/REAC_pathways/'+file_name,header=None)
    set1 = set(list(gene_list[0]))
    set2 = set(list(vlm_dict['spliced'].keys()))
    intersect = list(set1 & set2)

    save_name=file_name.split('_')[0]
    my_func.plot_phase_exp_raincloud(gene_matrix,intersect,boundary_dict,file_name,'analysis_results/'+save_name+'.png',False,False)
    my_func.plot_phase_exp_raincloud(gene_matrix,intersect,boundary_dict,file_name+'_log10','analysis_results/'+save_name+'_log10.png',False,True)
    
    #Perform variance test
    mean_lst_norm=[]
    mean_lst_log10=[]
    
    for phase in list(['G1','S','G2M']): #Setting custom order for better representation
        #Identify start and end of boundary
        start=boundary_dict[phase]
        if phase=='G1':
            end=boundary_dict['S']-1
        elif phase=='S':
            end=len(gene_matrix.index)
        else:
            end=boundary_dict['G1']-1
            
        #Check normal variance
        sub_matrix=gene_matrix[intersect]
        sub_matrix=sub_matrix.iloc[start:end]
        sub_matrix=sub_matrix.mean(axis=0)
        mean_lst_norm.append(sub_matrix.mean())
    
        #Check log10 variance
        sub_matrix=log10_matrix[intersect]
        sub_matrix=sub_matrix.iloc[start:end]
        sub_matrix=sub_matrix.mean(axis=0)
        mean_lst_log10.append(sub_matrix.mean())
        
        
    found_var_log10=statistics.variance(mean_lst_log10)
    found_var_norm=statistics.variance(mean_lst_norm)
    dta_res_list.append([file_name,found_var_norm,found_var_log10])

res_df=pd.DataFrame(dta_res_list,columns=['REAC_name','variance_norm_exp','variance_log10_exp'])
res_df.to_csv('analysis_results/threshold_identification.csv', index=False)
    





#Function which essentially does the above, but for each cell line, or each one give
layers=['spliced']
cell_line_var_dict={}
cell_dict={'HaCat':'A_B','293t':'A_B_C_D','jurkat':'A_B_C_D'}

for cell_line in list(cell_dict.keys()):
    folder_to_use=cell_dict[cell_line]
    mean_dict,CI_dict,bool_dict,count_dict,boundary_dict=my_utils.get_CI_data (cell_line, layers, folder_to_use)
    vlm_dict=my_utils.get_vlm_values(cell_line, layers,folder_to_use )
    gene_matrix=vlm_dict['spliced'] 
    mean_lst_norm=[]
    phase_gene_df = pd.DataFrame()
    for phase in list(['G1','S','G2M']): #Setting custom order for better representation
        #Identify start and end of boundary
        start=boundary_dict[phase]
        if phase=='G1':
            end=boundary_dict['S']-1
        elif phase=='S':
            end=len(gene_matrix.index)
        else:
            end=boundary_dict['G1']-1
            
        #Check normal variance
        sub_matrix=gene_matrix
        sub_matrix=sub_matrix.iloc[start:end]
        sub_matrix=sub_matrix.mean(axis=0)
        sub_df=sub_matrix.to_frame(name=phase)
        phase_gene_df[phase]=sub_df[phase]
        # mean_lst_norm.append(sub_matrix.mean())
        
    # found_var_norm=statistics.variance(mean_lst_norm)
    
    phase_gene_df=phase_gene_df.var(axis=1)
    phase_gene_df=phase_gene_df.to_frame(name=cell_line)

    cell_line_var_dict[cell_line]=phase_gene_df[cell_line]



thresh_genes_HaCat=list(cell_line_var_dict['HaCat'][cell_line_var_dict['HaCat']<1].index)
thresh_genes_293t=list(cell_line_var_dict['293t'][cell_line_var_dict['293t']<1].index)
thresh_genes_jurkat=list(cell_line_var_dict['jurkat'][cell_line_var_dict['jurkat']<1].index)



genes_HaCat=pd.read_csv('data_files/data_results/rank/HaCat/A_B_t_test_results.csv')
genes_293t=pd.read_csv('data_files/data_results/rank/293t/A_B_C_D_t_test_results.csv')
genes_jurkat=pd.read_csv('data_files/data_results/rank/jurkat/A_B_C_D_t_test_results.csv')
DEG_file_name='compare_data/dataDEGs_no_dupCorr_up.csv'
universe = list(genes_HaCat.gene_name) + list(genes_293t.gene_name) + list(genes_jurkat.gene_name)

#Filter based on threshold here
genes_HaCat=genes_HaCat[genes_HaCat.gene_name.isin(thresh_genes_HaCat)]
genes_293t=genes_293t[genes_293t.gene_name.isin(thresh_genes_293t)]
genes_jurkat=genes_jurkat[genes_jurkat.gene_name.isin(thresh_genes_jurkat)]


cell_line_dict={'HaCat':genes_HaCat,'293t':genes_293t,'jurkat':genes_jurkat}


#Function will also filter for significant genes
res_dict=my_func.create_TCGA_comparison_stat_results(cell_line_dict,DEG_file_name,universe)
#Convert to dataframe and save
res_df=pd.DataFrame.from_dict(res_dict)
res_df.to_csv('analysis_results/TCGA_cell_line_comp_thresh_attempt_1.csv')

my_func.plot_phase_exp_raincloud(gene_matrix,g1_group,boundary_dict,'G1 genes','analysis_results/g1_genes.png',False,True)
my_func.plot_phase_exp_raincloud(gene_matrix,s_group,boundary_dict,'S genes','analysis_results/s_genes.png',False,True)
my_func.plot_phase_exp_raincloud(gene_matrix,g2m_group,boundary_dict,'G2M genes','analysis_results/g2m_genes.png',False,True)
