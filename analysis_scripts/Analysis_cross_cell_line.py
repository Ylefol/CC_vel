#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:07:15 2023

@author: yohanl
"""
from snake_scripts.snake_functions import snake_analysis_functions as my_func
import pandas as pd

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
    print(comp)
    cell_res=my_func.chi_square_cell_lines(comp[0],cell_lines[comp[0]],comp[1],cell_lines[comp[1]])
    for cc_phase in phases:
        str_comp=comp[0]+'_'+comp[1]
        build_row=[str_comp,cell_res[1],cc_phase]
        phase_full_res=[]
        print(cc_phase)
        for phase_asso in phase_associations:
            print(phase_asso)
            phase_res=my_func.chi_square_cell_line_phases(comp[0],cell_lines[comp[0]],comp[1],cell_lines[comp[1]],cc_phase,phase_asso)
            build_row.append(phase_res[1])
        df_list.append(build_row)

#Convert list to pandas dataframe
col_names=['cell_lines','gene overlap','phase',phase_associations[0],phase_associations[1],phase_associations[2]]
chi_df=pd.DataFrame(df_list,columns=col_names)
#Save chi-square (overlap) results
chi_df.to_csv('data_files/data_results/chi_square_results.csv',index=False)


############## Gene overlap between cell lines + compare with lists
#Load the three files for significant genes
genes_HaCat=pd.read_csv('data_files/data_results/rank/HaCat/A_B_t_test_results.csv')
genes_293t=pd.read_csv('data_files/data_results/rank/293t/A_B_C_D_t_test_results.csv')
genes_jurkat=pd.read_csv('data_files/data_results/rank/jurkat/A_B_C_D_t_test_results.csv')

res = [i for i in genes_HaCat.padjusted if i != 'NA']
good_vals=[x for x in res if x<0.01]
significant_genes_HaCat=list(genes_HaCat.loc[genes_HaCat['padjusted'] .isin(good_vals)].gene_name)

res = [i for i in genes_293t.padjusted if i != 'NA']
good_vals=[x for x in res if x<0.01]
significant_genes_293t=list(genes_293t.loc[genes_293t['padjusted'] .isin(good_vals)].gene_name)

res = [i for i in genes_jurkat.padjusted if i != 'NA']
good_vals=[x for x in res if x<0.01]
significant_genes_jurkat=list(genes_jurkat.loc[genes_jurkat['padjusted'] .isin(good_vals)].gene_name)


set1 = set(significant_genes_HaCat)
set2 = set(significant_genes_293t)
set3 = set(significant_genes_jurkat)
intersect = list(set1 & set2 & set3)

with open('data_files/intersect_three_cell_lines.txt', 'w') as f:
    f.write('\n'.join(intersect))


############## Calculate observed and expected versus a DEG file (used TCGA analysis originally)
#NOTE, THIS IS NOT THE VARIANCE FILTERED VERSION

#Universe contains all genes that will be compared and more (due to non-significant cell line genes)
universe = list(genes_HaCat.gene_name) + list(genes_293t.gene_name) + list(genes_jurkat.gene_name)

genes_intersect=genes_HaCat[genes_HaCat.gene_name.isin(intersect)]

cell_line_dict={'HaCat':genes_HaCat,'293t':genes_293t,'jurkat':genes_jurkat,'intersect':genes_intersect}
DEG_file_name='compare_data/dataDEGs_no_dupCorr_up.csv'



#Function will also filter for significant genes
res_dict=my_func.create_TCGA_comparison_stat_results(cell_line_dict,DEG_file_name,universe)
#Convert to dataframe and save
res_df=pd.DataFrame.from_dict(res_dict)
res_df.to_csv('analysis_results/TCGA_cell_line_comp.csv')




















