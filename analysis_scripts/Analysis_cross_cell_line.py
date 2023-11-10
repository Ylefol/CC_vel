#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:07:15 2023

@author: yohanl
"""
from snake_scripts.snake_functions import snake_analysis_functions as my_func


#Declare the cell lines and associated folders of interest
cell_lines={}
cell_lines['HaCat']='A_B'
cell_lines['293t']='A_B_C_D'
cell_lines['jurkat']='A_B_C_D'


"""perform and save chi_square overlap results using t-test results"""
#By leaving all parameters as default, this will perform the overlap with just the 
#T-test results at a adjusted p value of 0.01
my_func.wrapper_chi_square_overlap(cell_lines)


"""Raincloud plots to observe known cell cycle gene behaviour across all cell lines"""

cell_lines={'HaCat':['A','B'],'293t':['A','B','C','D'],'jurkat':['A','B','C','D']}
cc_path='data_files/initial_data/Original_cell_cycle_genes_with_new_candidates.csv'

my_cc_dta=my_func.create_cell_line_replicate_cc_df(cc_path,cell_lines)

my_func.raincloud_for_cc_genes(my_cc_dta,'G1')
my_func.raincloud_for_cc_genes(my_cc_dta,'S')
my_func.raincloud_for_cc_genes(my_cc_dta,'G2/M')


my_cc_dta.to_csv('known_cc_data_expression.csv')




"""Identify genes which are significant based on variability thresholds, establish
the intersect of the relevant cell lines, compare these results with TCGA DEGs"""
#Create and save the variability thresholds
thresh_var_dict=my_func.identify_cell_line_variability_thresholds(cell_lines)

#Identify genes with significant negative threshold
HaCat_genes=my_func.get_sig_genes('HaCat','A_B',t_test_based=False,delay_type='dec_to_0',variability_based=False)
c293t_genes=my_func.get_sig_genes('293t','A_B_C_D',t_test_based=False,delay_type='dec_to_0',variability_based=False)
jurkat_genes=my_func.get_sig_genes('jurkat','A_B_C_D',t_test_based=False,delay_type='dec_to_0',variability_based=False)

#Remove genes with significant negative delay
thresh_var_dict['HaCat']=thresh_var_dict['HaCat'][thresh_var_dict['HaCat'].gene_name.isin(HaCat_genes)]
thresh_var_dict['293t']=thresh_var_dict['293t'][thresh_var_dict['293t'].gene_name.isin(c293t_genes)]
thresh_var_dict['jurkat']=thresh_var_dict['jurkat'][thresh_var_dict['jurkat'].gene_name.isin(jurkat_genes)]

#Create and save intersect
set_HaCat=set(list(thresh_var_dict['HaCat'].gene_name[thresh_var_dict['HaCat'].padjusted<0.01]))
set_293t=set(list(thresh_var_dict['293t'].gene_name[thresh_var_dict['293t'].padjusted<0.01]))
set_jurkat=set(list(thresh_var_dict['jurkat'].gene_name[thresh_var_dict['jurkat'].padjusted<0.01]))

intersect=list(set_HaCat & set_293t & set_jurkat)
with open('data_files/intersect_three_cell_lines.txt', 'w') as f:
    f.write('\n'.join(intersect))


DEG_file_name='compare_data/dataDEGs_no_dupCorr_up.csv'
ref_key='HaCat'

my_func.DEG_comparison_thresh_var(DEG_file_name,thresh_var_dict,'HaCat','automated',intersect)




"""Identify and plot Delay thresholds
The two functions below use Gaussian Mixture Modelling to split the trimodal distribution
of cell line delays. This is performed for all cell lines and all four delay categories.
The results are then plotted into a single file and saved to the main directory"""

gmm_dictionnary=my_func.create_GMM_dict(cell_line_dictionnary=cell_lines,gmm_n_comp=3,
                                        use_sig_genes=True,log10_transform=False,random_state=123)

my_func.plot_GMM_res_three_cell_lines(gmm_dict=gmm_dictionnary,plot_save_name='non_log10_gmm_results.png')


"""perform and save chi_square overlap results. This time using all available
filtering methods. Specifically - t-test based adjusted p value below 0.01, a 
variability greater than the mean of the median (of the REACTOME pathways) and a 
cell cycle delay (specifically in the decrease to 0 category) that does not go
below the identified threshold via the gaussian mixture modeling technique."""

my_func.wrapper_chi_square_overlap(cell_line_dict=cell_lines,use_t_sig=True,
                                   delay_cat='dec_to_0', use_var_sig=False,
                                   csv_save_name='data_files/data_results/chi_square_t_delay_results.csv')



"""Get and save significant genes to a text file. Significance using all three 
methods of significance"""

for cell_line in cell_lines.keys():
    retrieved_genes=my_func.get_sig_genes(cell_line,cell_lines[cell_line],
                                          t_test_based=True,delay_type='dec_to_0',
                                          variability_based=True)
    file_name=cell_line+'_all_sig.txt'
    with open(file_name, 'w') as f:
        for line in retrieved_genes:
            f.write(line+'\n')






"""Create an excel file for gene status, equivalent to supplementary file 1 in
the manuscript"""
my_func.create_gene_status_excel(cell_lines)



