#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 08:46:37 2022

@author: yohanl
"""
from snake_functions import snake_utils as my_utils
from snake_functions import snake_merged_results_functions as my_func

import sys
import os
import pandas as pd
import numpy as np
import shutil


#Params defined by user
cell_line_var=(sys.argv[4].split('/')[2])
replicates=(sys.argv[4].split('/')[4])


numer_of_iterations=int(sys.argv[2])
z_value_inputed=float(sys.argv[3])

#Default params
layers=['unspliced','spliced']
main_path='data_files/confidence_intervals/'+cell_line_var


#Folder name is the name of the replicate or merged replicates inputed
folder_name=replicates

#Function ensures that if merged replicates were inputed, they are identified as such
#Single replicates will remain in list format
replicates=replicates.split('_')


#Define path to results and create the necessary folder
path_to_results="data_files/confidence_intervals/"+cell_line_var+"/merged_results/"+folder_name
my_utils.create_folder(path_to_results)

#Merger of replicates and CI calculations is done separately on a per layer basis
layer_bool_dict={}
for layer in layers:
    #Merge replicates of the layers
    vel_gene_dict,smallest_reps=my_func.merge_replicates(main_path,replicates=replicates,layer=layer,target_merge='velocity') 

    #Perform calculations for gene velocity means and confidence intervals
    vel_gene_dict,vel_up_CI,vel_low_CI,vel_stand_dev_dict=my_func.calculate_dta_mean_and_confidence_intervals(gene_dict=vel_gene_dict,replicates=replicates,do_CI=True,num_iter=numer_of_iterations,z_val=z_value_inputed)
    
    #Merge vlm replicates per layer
    vlm_gene_dict,smallest_reps=my_func.merge_replicates(main_path,replicates=replicates,layer=layer,target_merge='expression') 
    
    #Perform calculations for gene vlm means and confidence intervals
    vlm_gene_dict,vlm_up_CI,vlm_low_CI,vlm_stand_dev_dict=my_func.calculate_dta_mean_and_confidence_intervals(gene_dict=vlm_gene_dict,replicates=replicates,do_CI=True,num_iter=numer_of_iterations,z_val=z_value_inputed)
    
    ####################
    #Merge vlm replicates per layer
    exp_mean_gene_dict,smallest_reps=my_func.merge_replicates(main_path,replicates=replicates,layer=layer,target_merge='mean_expression') 
    
    #Perform calculations for gene vlm means and confidence intervals
    exp_mean_gene_dict,exp_mean_up_CI,exp_mean_low_CI,exp_mean_stand_dev_dict=my_func.calculate_dta_mean_and_confidence_intervals(gene_dict=exp_mean_gene_dict,replicates=replicates,do_CI=True,num_iter=numer_of_iterations,z_val=z_value_inputed)
    ####################
    
    
    #Create bool dict – indicates where confidence intervals are above or below 0
    merged_rep_bool=my_func.create_bool_dictionnary(gene_dict=vel_gene_dict,up_CIs=vel_up_CI,low_CIs=vel_low_CI)
    
    #Create folder to save results
    my_utils.create_folder(path_to_results+'/'+layer)

    #save bool dict
    bool_df=pd.DataFrame.from_dict(merged_rep_bool)
    bool_df.to_csv(path_to_results+'/'+layer+"/bool.csv",index=False)
    
    #Create the delay dataframes
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    val_df=pd.DataFrame.from_dict(vel_up_CI)
    for gene in val_df.columns:
        #Finds positive and negative values before setting all to zero
        neg_vals=np.where(vel_up_CI[gene]<0)[0]
        pos_vals=np.where(vel_low_CI[gene]>0)[0]
        
        #Set all values to 0, will then be swapped as non-zeros if necessary
        val_df[gene][0:]=0
        if len(neg_vals)!=0:
            val_df[gene][neg_vals]=-1
        if len(pos_vals)!=0:
            val_df[gene][pos_vals]=1
    
    #Save delay dataframe
    val_df.to_csv(path_to_results+'/'+layer+"/vel_counts.csv",index=False)
    
    #Save lower CI for velocity
    vel_low_df=pd.DataFrame.from_dict(vel_low_CI)
    vel_low_df.to_csv(path_to_results+'/'+layer+"/vel_low_CI.csv",index=False)
    
    #Save upper CI for velocity
    vel_up_df=pd.DataFrame.from_dict(vel_up_CI)
    vel_up_df.to_csv(path_to_results+'/'+layer+"/vel_up_CI.csv",index=False)
    
    #Save mean velocity values
    vel_gene_df=pd.DataFrame.from_dict(vel_gene_dict)
    vel_gene_df.to_csv(path_to_results+'/'+layer+"/vel_means.csv",index=False)
    
    #Save combined standard deviation for velocity
    vel_stand_dev_df=pd.DataFrame.from_dict(vel_stand_dev_dict)
    vel_stand_dev_df.to_csv(path_to_results+'/'+layer+"/vel_combined_stdev.csv",index=False)
    
    #Save lower CI for vlm
    vlm_low_df=pd.DataFrame.from_dict(vlm_low_CI)
    vlm_low_df.to_csv(path_to_results+'/'+layer+"/exp_low_CI.csv",index=False)
    
    #Save upper CI for vlm
    vlm_up_df=pd.DataFrame.from_dict(vlm_up_CI)
    vlm_up_df.to_csv(path_to_results+'/'+layer+"/exp_up_CI.csv",index=False)
    
    #Save mean vlm values
    vlm_gene_df=pd.DataFrame.from_dict(vlm_gene_dict)
    vlm_gene_df.to_csv(path_to_results+'/'+layer+"/exp_means.csv",index=False)
    
    #Save combined standard deviation for vlm
    vlm_stand_dev_df=pd.DataFrame.from_dict(vlm_stand_dev_dict)
    vlm_stand_dev_df.to_csv(path_to_results+'/'+layer+"/exp_combined_stdev.csv",index=False)


    ####################
    #Save lower CI for mean expression
    vlm_low_df=pd.DataFrame.from_dict(exp_mean_low_CI)
    vlm_low_df.to_csv(path_to_results+'/'+layer+"/smooth_expression_low_CI.csv",index=False)
    
    #Save upper CI for mean expression
    vlm_up_df=pd.DataFrame.from_dict(exp_mean_up_CI)
    vlm_up_df.to_csv(path_to_results+'/'+layer+"/smooth_expression_up_CI.csv",index=False)
    
    #Save mean mean expression
    vlm_gene_df=pd.DataFrame.from_dict(exp_mean_gene_dict)
    vlm_gene_df.to_csv(path_to_results+'/'+layer+"/smooth_expression_means.csv",index=False)
    
    #Save combined standard deviation for mean expression
    vlm_stand_dev_df=pd.DataFrame.from_dict(exp_mean_stand_dev_dict)
    vlm_stand_dev_df.to_csv(path_to_results+'/'+layer+"/smooth_expression_combined_stdev.csv",index=False)


#Extracts boundary data based on the smallest replicate and saves it to the merged location
#We do this as it is the smallest replicates boundary data that will serve for plots
#This is also done here as past this point, there is no way of discerning which replicate
#Was the smallest since they have been balanced in accordance to the smallest.
boundary_dta = pd.read_csv('data_files/boundary_data/'+cell_line_var+'_'+smallest_reps[0]+'_boundaries.csv',header=None,index_col=0)
boundary_dta.to_csv(path_to_results+"/cell_boundaries.csv",header=False)


# #Delete iteratoins from machine
# for rep in replicates:
#     delete_path_1='data_files/confidence_intervals/'+cell_line_var+'/'+rep
#     delete_path_2='data_files/confidence_intervals/'+cell_line_var+'/'+rep

#     shutil.rmtree(delete_path_1)
#     shutil.rmtree(delete_path_2)















