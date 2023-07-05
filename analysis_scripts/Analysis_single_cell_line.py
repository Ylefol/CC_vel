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
cell_line='jurkat'
replicates=os.listdir('data_files/confidence_intervals/'+cell_line)
replicates.remove('merged_results')
#Create layers, declare folder to use (either single replicate or merged replicates)
layers=['spliced','unspliced']
folder_to_use='A_B_C_D'

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




############## Plot gene velocities
#Print out the 'UNG' gene
if 'UNG'in mean_dict['spliced'].keys():
    gene_save_path='all_figures/'+cell_line+'/analysis_results/'+folder_to_use+'/gene_plots/'
    my_func.plot_layer_smooth_vel('UNG', mean_dict, bool_dict, CI_dict, count_dict,vlm_dict,vlm_mean_dict,boundary_dict,cell_line,save_path=gene_save_path+'layer_vel')
    my_func.plot_curve_count('UNG', mean_dict, bool_dict, CI_dict, count_dict,boundary_dict,cell_line,save_path=gene_save_path+'vel_count')


############## Plot gene delays
delay_save_path='all_figures/'+cell_line+'/analysis_results/'+folder_to_use+'/gene_delays'
#Plot delay of all genes
my_func.plot_raincloud_delay(my_delay_df,cell_line,save_path=delay_save_path,save_name='delay_all_genes')

#Plot delay of rankable genes
my_rankable_delays=my_delay_df[my_delay_df["gene_name"].isin(rankable_genes)]
my_func.plot_raincloud_delay(my_rankable_delays,cell_line,save_path=delay_save_path,save_name='delay_rankable_genes')

#Subset the delay dataframe with significant genes and plot
sig_delay=my_delay_df[my_delay_df['gene_name'] .isin(significant_genes)]
my_func.plot_raincloud_delay(sig_delay,cell_line,save_path=delay_save_path,save_name='delay_001_genes')


#Comparative raincloud plot
my_func.raincloud_delay_two_files_two_categories(my_delay_df,sig_delay,'inc_to_+1','dec_to_0',cell_line+'_custom_delay.png')


# spearman delays
delay_cor_res=my_func.spearman_delay_categories(sig_delay)
delay_cor_res.to_csv('data_files/data_results/sig_genes_delay_correlation_'+cell_line+'_'+folder_to_use+'.csv',index=False)

##############Plot REAC trajectory plots #Takes a while and is RAM heavy
#Plot for all genes
REAC_save_path='all_figures/'+cell_line+'/analysis_results/'+folder_to_use+'/REAC_rankable'
REAC_dict=my_func.create_REAC_dict(vlm_mean_dict,rankable_genes)
my_func.create_REAC_summary_plots(REAC_dict,boundary_dict,layer='spliced',second_layer='unspliced',plot_path=REAC_save_path)

#Plot for significant genes
REAC_save_path='all_figures/'+cell_line+'/analysis_results/'+folder_to_use+'/REAC_significant'
REAC_dict=my_func.create_REAC_dict(vlm_mean_dict,significant_genes)
my_func.create_REAC_summary_plots(REAC_dict,boundary_dict,layer='spliced',second_layer='unspliced',plot_path=REAC_save_path)



############## Raincloud plots for gene expression per cell cycle phase
genes_HaCat=pd.read_csv('data_files/data_results/rank/'+cell_line+'/'+folder_to_use+'_t_test_results.csv')

g1_group=genes_HaCat.gene_name[genes_HaCat.phase_peak_exp=='G1']
s_group=genes_HaCat.gene_name[genes_HaCat.phase_peak_exp=='S']
g2m_group=genes_HaCat.gene_name[genes_HaCat.phase_peak_exp=='G2M']

my_func.plot_phase_exp_raincloud(vlm_mean_dict['spliced'] ,g1_group,boundary_dict,'G1 genes','analysis_results/g1_genes.png',False,True)
my_func.plot_phase_exp_raincloud(vlm_mean_dict['spliced'] ,s_group,boundary_dict,'S genes','analysis_results/s_genes.png',False,True)
my_func.plot_phase_exp_raincloud(vlm_mean_dict['spliced'] ,g2m_group,boundary_dict,'G2M genes','analysis_results/g2m_genes.png',False,True)


###############Perform some tests on variance thresholding

#Plot REACTOME plots for log10 variance and normal variance.
#Aid in the identification of housekeeping genes
#Results located in 'analysis_results' folder
my_func.cell_line_var_REAC(vlm_mean_dict,boundary_dict)


#Get the variance for each cell line
layers=['spliced']
cell_dict={'HaCat':'A_B','293t':'A_B_C_D','jurkat':'A_B_C_D'}
#Returns the log10 variance - thresholding performed on log10 values
cell_var_dict=my_func.create_cell_line_variance_dictionnary(cell_dict)

#Test different thresholds
thresh=0.001
thresh_name='0001'

var_thresh_dict={}
for target_cell_line in cell_var_dict.keys():
    thresh_genes=list(cell_var_dict[target_cell_line][cell_var_dict[target_cell_line]>thresh].index)
    genes_t_res=pd.read_csv('data_files/data_results/rank/'+target_cell_line+'/'+cell_dict[target_cell_line]+'_t_test_results.csv')
    thresh_subset=genes_t_res[genes_t_res.gene_name.isin(thresh_genes)]    
    var_thresh_dict[target_cell_line]=thresh_subset
    thresh_subset.to_csv('analysis_results/'+target_cell_line+'_'+thresh_name+'_thresh.csv',index=False)


#Create REAC plots based on variance threshold
#Retrieves significant genes within variance filtered list
thresh_sig_HaCat=list(var_thresh_dict['HaCat'].gene_name[var_thresh_dict['HaCat'].padjusted<0.01]) 
REAC_save_path='all_figures/HaCat/analysis_results/A_B/REAC_thresh'+thresh_name
REAC_dict=my_func.create_REAC_dict(vlm_mean_dict,thresh_sig_HaCat)
my_func.create_REAC_summary_plots(REAC_dict,boundary_dict,layer='spliced',second_layer='unspliced',plot_path=REAC_save_path)



DEG_file_name='compare_data/dataDEGs_no_dupCorr_up.csv'
ref_key='HaCat'
thresh_var_dict=var_thresh_dict


#Create intersect
set_HaCat=set(list(thresh_var_dict['HaCat'].gene_name[thresh_var_dict['HaCat'].padjusted<0.01]))
set_293t=set(list(thresh_var_dict['293t'].gene_name[thresh_var_dict['293t'].padjusted<0.01]))
set_jurkat=set(list(thresh_var_dict['jurkat'].gene_name[thresh_var_dict['jurkat'].padjusted<0.01]))

intersect=list(set_HaCat & set_293t & set_jurkat)

my_func.DEG_comparison_thresh_var(DEG_file_name,var_thresh_dict,'HaCat',thresh_name,intersect)

###############



############## Raincloud plots to observe behaviour of known cell cycle genes

cell_lines={'HaCat':['A','B'],'293t':['A','B','C','D'],'jurkat':['A','B','C','D']}
cc_path='data_files/initial_data/Original_cell_cycle_genes_with_new_candidates.csv'

my_cc_dta=my_func.create_cell_line_replicate_cc_df(cc_path,cell_lines)

my_func.raincloud_for_cc_genes(my_cc_dta,'G1')
my_func.raincloud_for_cc_genes(my_cc_dta,'S')
my_func.raincloud_for_cc_genes(my_cc_dta,'G2/M')


my_cc_dta.to_csv('known_cc_data_expression.csv')



############## New custom delay plot


#Import required functions
from snake_scripts.snake_functions import snake_utils as my_utils
from snake_scripts.snake_functions import snake_analysis_functions as my_func

#Import libraries
import os
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
vlm_mean_dict=my_utils.get_vlm_values(cell_line, layers,folder_to_use,get_mean=True)
vlm_dict=my_utils.get_vlm_values(cell_line, layers,folder_to_use,get_mean=False)


#Make the plot - 2 versions

my_func.plot_spliced_velocity_expression_zero_point(mean_dict,CI_dict,bool_dict,vlm_mean_dict,boundary_dict,'UNG',plot_name='UNG_1.png')

my_func.plot_velocity_expression_zero_point(mean_dict,CI_dict,bool_dict,vlm_dict,vlm_mean_dict,boundary_dict,'UNG',plot_name='UNG_2.png')



