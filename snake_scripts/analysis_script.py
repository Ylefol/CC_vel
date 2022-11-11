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
# import scipy.stats as stats
# import matplotlib.pyplot as plt

cell_line='HaCat'
# cell_line=key
#Find replicates
replicates=os.listdir('data_files/confidence_intervals/'+cell_line)
replicates.remove('merged_results')
#Create layers
layers=['spliced','unspliced']


folder_to_use='A_B'
# folder_to_use=cell_lines[key]

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

if 'UNG'in mean_dict['spliced'].keys():
    gene_save_path='all_figures/'+cell_line+'/analysis_results/'+folder_to_use+'/gene_plots/'
    my_func.plot_layer_smooth_vel('UNG', mean_dict, bool_dict, CI_dict, count_dict,vlm_dict,boundary_dict,cell_line,save_path=gene_save_path+'layer_vel')
    my_func.plot_curve_count('UNG', mean_dict, bool_dict, CI_dict, count_dict,boundary_dict,cell_line,save_path=gene_save_path+'vel_count')

delay_save_path='all_figures/'+cell_line+'/analysis_results/'+folder_to_use+'/gene_delays'
#Plot delay of all genes
my_func.plot_raincloud_delay(my_delay_df,cell_line,save_path=delay_save_path,save_name='delay_all_genes')

#Plot delay of rankable genes
my_significant_delays=my_delay_df[my_delay_df["gene_name"].isin(rankable_genes)]
my_func.plot_raincloud_delay(my_significant_delays,cell_line,save_path=delay_save_path,save_name='delay_rankable_genes')

#Subset the delay dataframe with significant genes and plot
sub_delay=my_delay_df[my_delay_df['gene_name'] .isin(significant_genes)]
my_func.plot_raincloud_delay(sub_delay,cell_line,save_path=delay_save_path,save_name='delay_001_genes')


#Plot REAC results
REAC_save_path='all_figures/'+cell_line+'/analysis_results/'+folder_to_use+'/REAC_rankable'
REAC_dict=my_func.create_REAC_dict(vlm_dict,rankable_genes)
my_func.create_REAC_summary_plots(REAC_dict,boundary_dict,layer='spliced',second_layer='unspliced',plot_path=REAC_save_path)

#Plot REAC results significant genes
REAC_save_path='all_figures/'+cell_line+'/analysis_results/'+folder_to_use+'/REAC_significant'
REAC_dict=my_func.create_REAC_dict(vlm_dict,significant_genes)
my_func.create_REAC_summary_plots(REAC_dict,boundary_dict,layer='spliced',second_layer='unspliced',plot_path=REAC_save_path)



#Custom wrapper to create statistics for gene overlap
import itertools
cell_lines={}
cell_lines['HaCat']='A_B'
cell_lines['293t']='A_B_C_D'
cell_lines['jurkat']='A_B_C_D'
cell_line_comps=list(itertools.combinations(list(cell_lines.keys()), 2))

phases=['G1','S','G2M']
phase_associations=['phase_peak_vel','phase_peak_exp','phase_start_vel']

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

col_names=['cell_lines','gene overlap','phase',phase_associations[0],phase_associations[1],phase_associations[2]]

chi_df=pd.DataFrame(df_list,columns=col_names)
chi_df.to_csv('chi_square_results.csv',index=False)
    

#Plot layer/vel and vel/count for all cc genes


# #Print all CC genes
# for phase in list(pd.unique(cc_genes_df['phase'])):
#     for gene in list(cc_genes_df[cc_genes_df['phase']==phase]['gene']):
#         if gene in mean_dict['spliced'].keys():
#             my_func.plot_layer_smooth_vel(gene, mean_dict, bool_dict, CI_dict, count_dict,vlm_dict,boundary_dict,cell_line,save_path='cc_genes/'+cell_line+'/vel/'+phase,single_rep=False)
#             my_func.plot_curve_count(gene, mean_dict, bool_dict, CI_dict, count_dict,boundary_dict,cell_line,save_path='cc_genes/'+cell_line+'/count/'+phase)



delay_save_path='all_figures/'
#Plot delay of all genes

#Plot delay of rankable genes
my_significant_delays=my_delay_df[my_delay_df["gene_name"].isin(rankable_genes)]

#Subset the delay dataframe with significant genes and plot
sub_delay=my_delay_df[my_delay_df['gene_name'] .isin(significant_genes)]


new_delay_df=my_delay_df.copy()
new_delay_df['inc_to_+1_sig']=sub_delay['inc_to_+1']
new_delay_df['dec_to_0_sig']=sub_delay['dec_to_0']

delay_df=new_delay_df

#Log transform the data
dta1 = my_utils.log10_dta(delay_df,'inc_to_+1')
dta2 = my_utils.log10_dta(delay_df,'dec_to_0')
dta3 = my_utils.log10_dta(delay_df,'inc_to_+1_sig')
dta4 = my_utils.log10_dta(delay_df,'dec_to_0_sig')
    
    
import seaborn as sns
import matplotlib.pyplot as plt
import ptitprince as pt
import gc
import matplotlib as mpl

plot_name=''
save_name='delay_custom'
save_path='all_figures/'
num_genes=len(dta1)
if plot_name=='':
    plot_name="Raincloud plots gene delay ("+str(num_genes)+" genes)"
if save_name=='':
    save_name='Raincloud_plot.png'
    
data_to_plot = [dta1, dta2, dta3, dta4]

#Set the style/background of the plot
sns.set(style="whitegrid",font_scale=2)

#Create the plot
f, ax = plt.subplots(figsize=(15, 5))
ax=pt.RainCloud(data = data_to_plot, palette = 'Set2', bw = 0.3, width_viol = .6, ax = ax, orient = 'v',offset=0.12)
# set style for the axes
labels = ['increase\nto +1', 'decrease\nto 0', 'increase\nto +1\nsignificant', 'decrease\nto 0\nsignificant']
for ax in [ax]:
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticklabels(labels)
    
plt.title(plot_name)
plt.ylabel('log10(cell delay)')
#Save the plot
if save_path=='':
    plot_path="all_figures/"+cell_line+"/merged_replicates/gene_delays"
else:
    plot_path=save_path
if not os.path.exists(plot_path):
    os.makedirs(plot_path, exist_ok=True)
plt.savefig(os.path.join(plot_path,save_name+'.png'),bbox_inches='tight')
plt.clf()
plt.close("all")
# plt.show()
gc.collect()

#Resets the 'theme' for the plots as to not interfere with downstream plots
mpl.rc_file_defaults()
    
