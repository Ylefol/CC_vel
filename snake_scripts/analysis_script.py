#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 11:02:58 2021

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

# Load results
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


##############Calculate length of 3'UTR for significant genes
#NOTE: This takes quite a bit of time.
gtf_path='data_files/gencode.v33.annotation.gtf'
my_UTRs=my_func.create_read_UTR_results(cell_line,folder_to_use,gtf_path,significant_genes)



############## Calculate the cross cell line overlaps and save
import itertools
#Declare the cell lines and associated folders of interest
cell_lines={}
cell_lines['HaCat']='A_B'
cell_lines['293t']='A_B_C_D'
cell_lines['jurkat']='A_B_C_D'

#Get all possible cell line comparisons
cell_line_comps=list(itertools.combinations(list(cell_lines.keys()), 2))

#Establish phases and categories of phase association
phases=['G1','S','G2M']
phase_associations=['phase_peak_vel','phase_peak_exp','phase_start_vel']

#Iterate over the cell lines, phases, phase association and perform overlap
#Results are stored in a list
df_list=[]
for comp in cell_line_comps:
    cell_res=my_func.chi_square_cell_lines(comp[0],cell_lines[comp[0]],comp[1],cell_lines[comp[1]])
    for cc_phase in phases:
        str_comp=comp[0]+'_'+comp[1]
        build_row=[str_comp,cell_res[1],cc_phase]
        phase_full_res=[]
        for phase_asso in phase_associations:
            phase_res=my_func.chi_square_cell_line_phases(comp[0],cell_lines[comp[0]],comp[1],cell_lines[comp[1]],cc_phase,phase_asso)
            build_row.append(phase_res[1])
        df_list.append(build_row)

#Convert list to pandas dataframe
col_names=['cell_lines','gene overlap','phase',phase_associations[0],phase_associations[1],phase_associations[2]]
chi_df=pd.DataFrame(df_list,columns=col_names)
#Save chi-square (overlap) results
chi_df.to_csv('data_files/data_results/chi_square_results.csv',index=False)
    



############## 3'UTR analysis and comparison with significant delays
#Import required functions
from snake_scripts.snake_functions import snake_utils as my_utils
from snake_scripts.snake_functions import snake_analysis_functions as my_func

#Import libraries
import os
import numpy as np
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
