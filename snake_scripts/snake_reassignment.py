#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 08:57:45 2020

@author: yohan
"""
import scanpy as sc
import numpy as np
from snake_functions import snake_reassignment_functions as my_func
import os
import shutil

import sys

#This read the whole file into memory
adata = sc.read_loom(sys.argv[1],X_name="")

samp_var=(list(sys.argv[1])[-6])
cell_line_var=(sys.argv[1].split('_')[-2])
cell_line_var=cell_line_var.split('/')[1]

filter_numbers={}
filter_numbers['cells']=[len(adata.obs)]
filter_numbers['genes']=[len(adata.var)]


#Create all folders
main_path=my_func.set_up_folders_reassignment_script(cell_line_var,samp_var)


barcode_list = []
if sys.argv[2]!='None':
    for i in open(sys.argv[2],"r").readlines(): 
        barcode_list.append(i.rstrip())

if len(barcode_list)>0:
    adata = my_func.filter_loom_barcode(adata,barcode_list)

filter_numbers['cells'].append(len(adata.obs)-len(barcode_list))


adata.var_names_make_unique()

# gtf="data_files/gencode.v33.annotation.gtf"
# MT_list=pp.create_chrM_list(gtf)

path_MT=sys.argv[3]
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

my_func.scanpy_pp_plots(adata,MT_list,True,True,True,path=main_path,sub_folder='Pre')

#Get threshold for unspliced and mirochondrial filters
max_unspli,min_unspli,mito_thresh=my_func.get_unspli_and_mito_thresholds(cell_line_var,samp_var)

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

my_func.scanpy_pp_plots(adata,MT_list,True,True,True,path=main_path,sub_folder='Post')


# adata.write_loom(snakemake.output[0], write_obsm_varm=False)

CC_path=sys.argv[4]
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

my_func.perform_scanpy_pca(adata,compute=True,exclude_gene_counts=False,exclude_CC=False,save_path=main_path,sub_folder='initial')
sc.pp.regress_out(adata,['n_counts'])

my_func.perform_scanpy_pca(adata,compute=True,exclude_gene_counts=False,exclude_CC=False, save_path=main_path,sub_folder='regressed')
# my_func.plot_phase_bar_dist(adata, 20, return_data=False, plot_path=main_path+"/figures/pca_bar_line_plots/regressed")


adata_check=adata.copy()

my_func.score_filter_reassign(adata_check,0.10)
my_func.perform_scanpy_pca(adata_check,compute=False,exclude_gene_counts=True,exclude_CC=False,save_path=main_path,sub_folder='score_filtered')
# my_func.plot_phase_bar_dist(adata_check, 20, return_data=False, plot_path=main_path+"/figures/pca_bar_line_plots/score_filtered")

my_func.coordinate_filter_reassign(adata_check,0.10)
my_func.perform_scanpy_pca(adata_check,compute=False,exclude_gene_counts=True,exclude_CC=False,save_path=main_path,sub_folder='coordinate_filtered')
# my_func.plot_phase_bar_dist(adata_check, 20, return_data=False, plot_path=main_path+"/figures/pca_bar_line_plots/coordinate_filtered")

#Sets angles and pseudotime order, then PCAs it
############

adata_check.obs['angles'] = my_func.compute_angles(adata_check.obsm['X_pca'][:,:2])
adata_check = adata_check[adata_check.obs['angles'].argsort(),:]
adata_check.obs['order'] = np.arange(len(adata_check.obs))

#Calculate and plot the normal distribution of each phase on the angles avaialable
#This determines the angles at which each phase starts and ends
#Also determines the orientation of the dataset
dict_crossover,orientation=my_func.normal_dist_boundary_wrapper(adata_check,save_path=(main_path+"/figures/norm_dist_plots/"))


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

my_func.perform_scanpy_pca(adata_check,compute=False,exclude_gene_counts=True,exclude_CC=False,save_path=main_path,sub_folder='filtered_reassigned')


#Reassignement section
###########
adata.obs['angles'] = my_func.compute_angles(adata.obsm['X_pca'][:,:2])
adata = adata[adata.obs['angles'].argsort(),:]
adata.obs['order'] = np.arange(len(adata.obs))


my_func.phase_angle_assignment(adata,g1_limit=g1_angle,s_limit=s_angle,g2m_limit=g2m_angle)

#Find the new shift using the appropriate angle
if orientation=='G1':
    shift_idx=np.where(adata.obs['angles']==g2m_angle)[0][0]
else:
    shift_idx=np.where(adata.obs['angles']==s_angle)[0][0]
adata=my_func.shift_data(adata, shift_idx, direction = 'negative', reverse=False)

#Re-establishes the new starting order due to the above shift
g2m_start=np.where(adata.obs['phase']=='G2M')[0][0]
g1_start=np.where(adata.obs['phase']=='G1')[0][0]
s_start=np.where(adata.obs['phase']=='S')[0][0]
adata.uns['phase_boundaries'] = {'g1_start': g1_start, 's_start': s_start, 'g2m_start': g2m_start}


#Finds new angle boundaries and plots PCA
ang_boundaries=np.array([])
for p in adata.obs.phase:
    if p == 'G1':
        ang_boundaries=np.append(ang_boundaries,g1_angle)
    elif p=='S':
        ang_boundaries=np.append(ang_boundaries,s_angle)
    else:   #it is G2M
        ang_boundaries=np.append(ang_boundaries,g2m_angle)
adata.obs["angle_boundaries"]=ang_boundaries
my_func.perform_scanpy_pca(adata,compute=False,exclude_gene_counts=True,exclude_CC=False,save_path=main_path,sub_folder='reassigned')


#QC checks if the CC gene expressions are highest in cells categorized in their phase
########
# my_func.compare_marker_genes_per_phase_mod(adata, CC_path, phase_choice="G1", do_plots=False)
# my_func.compare_marker_genes_per_phase_mod(adata, CC_path, phase_choice="S", do_plots=False)
# my_func.compare_marker_genes_per_phase_mod(adata, CC_path, phase_choice="G2M", do_plots=False)

#Saves the orientation in the adata object
ori_arr=np.asarray([orientation]*len(adata.obs))
adata.obs["orientation"]=ori_arr[0]

# adata.write_loom("loom_files//HaCat_hsa_phase_reassignment_CC_A_new_cand.loom", write_obsm_varm=True)

sort_order = adata.obs_names
adata_pre_selection=adata_pre_selection[sort_order,:]
adata_pre_selection.obs=adata.obs
adata_pre_selection.uns = adata.uns

adata.write_loom(sys.argv[5], write_obsm_varm=False)
adata_pre_selection.write_loom(sys.argv[6], write_obsm_varm=False)


#Since scanpy forces the creation of a 'figures' folder, the above code is adapted to that
#Here we re-organize the folders with figures to remove the unecessary 'figures' folder
#the results are instead moved to their appropriate location represented by
#the cell line followed by the replicate in the 'all_figures' folder

path_fig='all_figures/'+cell_line_var+'/'+samp_var+'/figures'
path_no_fig='all_figures/'+cell_line_var+'/'+samp_var
folders_in_figure=os.listdir(path_fig)

for folder in folders_in_figure:
    original = path_fig+'/'+folder
    target = path_no_fig
    
    #If folder already exists, remove it to be able to replace with current iteration
    if folder in os.listdir(target):
        shutil.rmtree(path_no_fig+'/'+folder)
    shutil.move(original,target)
    
shutil.rmtree(path_fig)


