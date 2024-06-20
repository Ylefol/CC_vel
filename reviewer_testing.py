#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 13:26:23 2024

@author: yohanl
"""


import scanpy as sc
import numpy as np
from snake_scripts.snake_functions import snake_reassignment_functions as my_func
import os
import shutil

import sys

#This read the whole file into memory
adata = sc.read_loom('data_files/initial_data/loom_files/HaCat_A.loom',X_name="")

samp_var='A'
cell_line_var='HaCat'

filter_numbers={}
filter_numbers['cells']=[len(adata.obs)]
filter_numbers['genes']=[len(adata.var)]


#Create all folders
main_path=my_func.set_up_folders_reassignment_script(cell_line_var,samp_var)

storage_folder='remove_when_done'

barcode_list = []
if "data_files/initial_data/barcodes/"+cell_line_var+"_"+samp_var+"_barcodes.txt"!='None':
    for i in open("data_files/initial_data/barcodes/"+cell_line_var+"_"+samp_var+"_barcodes.txt","r").readlines(): 
        barcode_list.append(i.rstrip())

if len(barcode_list)>0:
    adata = my_func.filter_loom_barcode(adata,barcode_list)

filter_numbers['cells'].append(len(adata.obs)-len(barcode_list))


adata.var_names_make_unique()

# gtf="data_files/gencode.v33.annotation.gtf"
# MT_list=pp.create_chrM_list(gtf)

path_MT="data_files/initial_data/chrM_unique.txt"
MT_list=my_func.get_chrM_list(path_MT)

#Filter cells based on minimum gene_count per cell
sc.pp.filter_cells(adata,min_genes=500)
filter_numbers['cells'].append(len(adata.obs))

#Filter genes based on minimum number of cells expressing the gene
sc.pp.filter_genes(adata,min_cells=5)
filter_numbers['genes'].append(len(adata.var))

#Filter cells based on number of reads per cell 
sc.pp.filter_cells(adata,min_counts=4000)
filter_numbers['cells'].append(len(adata.obs))

# my_func.scanpy_pp_plots(adata,MT_list,True,True,True,path=storage_folder,sub_folder='pre-pp')

#Get threshold for unspliced and mirochondrial filters
max_unspli=20
min_unspli=10
mito_thresh=0

#Filter cells based on maximum unspliced percentage
my_func.filter_based_on_spliced_unspliced_ratio(adata,'unspliced',max_percent=max_unspli)
filter_numbers['cells'].append(len(adata.obs))

#Filter cells based on minimum unspliced percentage
my_func.filter_based_on_spliced_unspliced_ratio(adata,'unspliced',min_percent=min_unspli)
filter_numbers['cells'].append(len(adata.obs))

#Filter cells based on minimum intron count
my_func.filter_adata_using_min_intron_count(adata,1)
filter_numbers['genes'].append(len(adata.var))

#Filter cells based on MT percentage
my_func.filter_MT_percent(adata,MT_list,mito_thresh)
filter_numbers['cells'].append(len(adata.obs))

#Create and format filter table
import pandas as pd
filter_numbers=pd.DataFrame.from_dict(filter_numbers, orient='index').T
filter_numbers['cells']=filter_numbers['cells'].astype(dtype ='int64')
filter_numbers['genes']=filter_numbers['genes'].fillna(0).astype(dtype ='int64')
filter_numbers.to_csv('all_figures/'+cell_line_var+'/'+samp_var+'/filter_numbers.csv')

# my_func.scanpy_pp_plots(adata,MT_list,True,True,True,path=storage_folder,sub_folder='post-pp')


# adata.write_loom(snakemake.output[0], write_obsm_varm=False)

CC_path="data_files/initial_data/Original_cell_cycle_genes_with_new_candidates.csv"
# CC_path="data_files//Original_cell_cycle_genes.csv"


my_func.my_score_genes_cell_cycle_improved(adata=adata,layer_choice='spliced', CC_path=CC_path)

sc.pp.normalize_total(adata, target_sum=1e6) #RPM normalized for all layers
sc.pp.log1p(adata)
adata.raw = adata#necessary?

#Create a copy to 'fish up' the non-CC genes after ordering
adata_pre_selection = adata.copy()
my_func.check_cols_and_rows(adata_pre_selection)

adata=my_func.selection_method(adata,highly_variable=False,CC_path=CC_path)

#Removes any genes with 0 reads
my_func.check_cols_and_rows(adata)

sc.tl.pca(adata,svd_solver='arpack')
sc.pp.regress_out(adata,['n_counts'])

sc.tl.pca(adata,svd_solver='arpack')
# my_func.plot_phase_bar_dist(adata, 20, return_data=False, plot_path=main_path+"/figures/pca_bar_line_plots/regressed")


adata_check=adata.copy()

my_func.score_filter_reassign(adata_check,0.10)
# my_func.plot_phase_bar_dist(adata_check, 20, return_data=False, plot_path=main_path+"/figures/pca_bar_line_plots/score_filtered")

my_func.coordinate_filter_reassign(adata_check,0.10)
# my_func.plot_phase_bar_dist(adata_check, 20, return_data=False, plot_path=main_path+"/figures/pca_bar_line_plots/coordinate_filtered")

#Sets angles and pseudotime order, then PCAs it
############

adata_check.obs['angles'] = my_func.compute_angles(adata_check.obsm['X_pca'][:,:2])
adata_check = adata_check[adata_check.obs['angles'].argsort(),:]
adata_check.obs['order'] = np.arange(len(adata_check.obs))

#Calculate and plot the normal distribution of each phase on the angles avaialable
#This determines the angles at which each phase starts and ends
#Also determines the orientation of the dataset
dict_crossover,orientation=my_func.normal_dist_boundary_wrapper(adata_check,save_path='all_figures/')


if orientation=='G1':
    adata_check.uns['phase_boundaries'] = {'g1_start': dict_crossover['G1_G2M'], 's_start': dict_crossover['S_G1'], 'g2m_start': dict_crossover['S_G2M']}
else:
    adata_check.uns['phase_boundaries'] = {'g1_start': dict_crossover['S_G1'], 's_start': dict_crossover['S_G2M'], 'g2m_start': dict_crossover['G1_G2M']}

#finds the angles using the .uns data from the dict_crossover information
g1_angle, s_angle, g2m_angle=my_func.find_angle_boundaries(adata_check)

#Reassigns the phases based on the found angles
my_func.phase_angle_assignment(adata_check,g1_angle,s_angle,g2m_angle)

#Shift the data based on G1-G2M crossover to set that as the zero point
adata_check=my_func.shift_data(adata_check, dict_crossover['S_G2M'], direction = 'negative', reverse=False)



#Reassignement section
###########
adata.obs['angles'] = my_func.compute_angles(adata.obsm['X_pca'][:,:2])
adata = adata[adata.obs['angles'].argsort(),:]
adata.obs['order'] = np.arange(len(adata.obs))


sc.pl.pca(adata,color=['phase','angles'],show=True,save=None)

G1_in_S=adata[(adata.obs['phase']=='G1') & (adata.obs['angles'] >s_angle) & (adata.obs['angles']<g2m_angle)]
G1_in_G2M=adata[(adata.obs['phase']=='G1') & (adata.obs['angles'] >g2m_angle)]


# HANDLE FROM HERE
####
 
S_adata=adata[(adata.obs['phase']!='G2M') & (adata.obs['angles'] >s_angle) & (adata.obs['angles']<g2m_angle)]
sc.tl.rank_genes_groups(S_adata, 'phase', method='t-test', key_added = "t-test")
df1 = sc.get.rank_genes_groups_df(S_adata, group='S', key='t-test', pval_cutoff=0.01,log2fc_min=1)
df2 = sc.get.rank_genes_groups_df(S_adata, group='S', key='t-test', pval_cutoff=0.01,log2fc_max=-1)
S_S_genes=pd.concat([df1, df2], ignore_index = True)

df1 = sc.get.rank_genes_groups_df(S_adata, group='G1', key='t-test', pval_cutoff=0.01,log2fc_min=1)
df2 = sc.get.rank_genes_groups_df(S_adata, group='G1', key='t-test', pval_cutoff=0.01,log2fc_max=-1)
S_G1_genes=pd.concat([df1, df2], ignore_index = True)


G2M_adata=adata[(adata.obs['phase']!='S') & (adata.obs['angles'] >g2m_angle)]
sc.tl.rank_genes_groups(G2M_adata, 'phase', method='t-test', key_added = "t-test")

df1 = sc.get.rank_genes_groups_df(G2M_adata, group='G2M', key='t-test', pval_cutoff=0.01,log2fc_min=1)
df2 = sc.get.rank_genes_groups_df(G2M_adata, group='G2M', key='t-test', pval_cutoff=0.01,log2fc_max=-1)
G2M_G2M_genes=pd.concat([df1, df2], ignore_index = True)

df1 = sc.get.rank_genes_groups_df(G2M_adata, group='G1', key='t-test', pval_cutoff=0.01,log2fc_min=1)
df2 = sc.get.rank_genes_groups_df(G2M_adata, group='G1', key='t-test', pval_cutoff=0.01,log2fc_max=-1)
G2M_G1_genes=pd.concat([df1, df2], ignore_index = True)


# create a excel writer object
with pd.ExcelWriter("G1_infiltration.xlsx") as writer:
    S_S_genes.to_excel(writer, sheet_name="S_cluster_S_genes", index=False)
    S_G1_genes.to_excel(writer, sheet_name="S_cluster_G1_genes", index=False)
    G2M_G2M_genes.to_excel(writer, sheet_name="G2M_cluster_G2M_genes", index=False)
    G2M_G1_genes.to_excel(writer, sheet_name="G2M_cluster_G1_genes", index=False)