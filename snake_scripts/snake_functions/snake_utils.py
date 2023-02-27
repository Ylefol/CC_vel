#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 09:04:40 2021

@author: yohanl
"""

import os
import pyranges as pr
import numpy as np
import pandas as pd

def create_folder(path):
    """
    Function which checks if a directory and it's path exist, if it does not,
    it creates the path/directory.

    Parameters
    ----------
    path : string
        The path/directory to be created.

    Returns
    -------
    None.

    """
    path=path.split('/')
    for idx, elem in enumerate(path):
        if idx != 0:
            end_path='/'.join(path[0:idx+1])
        else:
            end_path=elem
        if os.path.exists(end_path)==False:
            os.mkdir(end_path)
            
            


def balance_dataframes(df_dict):
    """
    Balances a dataframe in terms of genes and cells. In the event that cells
    have to be balanced, the dataframe with additional cells will have cells
    removed in even increments. Genes are balanced by removing genes which are
    not present in both dataframes.

    Parameters
    ----------
    df_dict : Dictionnary
        A dictionnary containing the dataframes that are to be balanced.

    Returns
    -------
    final_df_dict : dictionnary
        A dictionnary containing the balanced dataframes.

    """
    col_list=df_dict[next(iter(df_dict))].columns
    length_list=[]
    for key in df_dict.keys():
        length_list.append(len(df_dict[key].index))
        col_list=col_list.intersection(df_dict[key].columns)
    smallest_df=min(length_list)
    for key in df_dict.keys():
        idx=np.round(np.linspace(0, len(df_dict[key].index) -1, smallest_df)).astype(int)
        df_dict[key]=df_dict[key].iloc[idx]
        df_dict[key]=df_dict[key][col_list]
        
    final_df_dict={}
    for key in df_dict:
        final_df_dict[key]={}
        for gene in df_dict[key]:
            final_df_dict[key][gene]=df_dict[key][gene].to_numpy()
            
    return final_df_dict

def find_smallest_replicate(df_dict):
    """
    Function which identifies the replicate with the least amount of cells
    

    Parameters
    ----------
    df_dict : dictionnary
        dictionnary containing the dataframes with the mean values of each replicate.

    Returns
    -------
    found_rep : string
        The smallest replicate

    """
    length_val=None
    for key in df_dict.keys():
        if length_val == None:
            length_val=len(df_dict[key].index)
            found_rep=key
        else:
            if length_val > len(df_dict[key].index):
                length_val=len(df_dict[key].index)
                found_rep=key
    found_rep=found_rep.split('_')[1]
    return found_rep

def merge_mean_layers(df_dict,layers):
    """
    Function which merges the mean values of the replicates.
    Function essentially takes the mean of the means.

    Parameters
    ----------
    df_dict : dictionnary
        dictionnary containing the blaanced dataframes for each replicate.
    layers : list
        list containing the layers (spliced and unspliced.

    Returns
    -------
    merged_means : dictionnary
        The merged means.

    """
    merged_means={}
    for layer in layers:
        key_list=[]
        for key in df_dict.keys():
            if key.split('_')[0]==layer:
                key_list.append(key)
        
        my_gene_dict={}
        for gene in df_dict[key_list[0]].keys():
            my_gene_dict[gene]=[]
            for idx,key in enumerate(key_list):
                if len(my_gene_dict[gene])==0:
                    my_gene_dict[gene]=np.array([df_dict[key][gene]])
                else:
                    my_gene_dict[gene]=np.append(my_gene_dict[gene],np.array([df_dict[key][gene]]),axis=0)
        for gene in my_gene_dict.keys():
            my_gene_dict[gene]=my_gene_dict[gene].mean(axis=0)
        
        merged_means[layer]=my_gene_dict
    return merged_means


def get_CI_data(cell_line, layers, target_merger,gene_selection='all'):
    df_dict={}
    CI_dict={}
    bool_dict={}
    count_dict={}
    for layer in layers:
        bool_dict[layer]=pd.read_csv('data_files/confidence_intervals/'+cell_line+'/merged_results/'+target_merger+'/'+layer+'/bool.csv')
        count_dict[layer]=pd.read_csv('data_files/confidence_intervals/'+cell_line+'/merged_results/'+target_merger+'/'+layer+'/counts.csv')
    
        CI_dict[layer]={}
        for CI in ['low_CI','up_CI']:
            CI_dict[layer][CI]=pd.read_csv('data_files/confidence_intervals/'+cell_line+'/merged_results/'+target_merger+'/'+layer+'/'+CI+'.csv')

        df_dict[layer]=pd.read_csv('data_files/confidence_intervals/'+cell_line+'/merged_results/'+target_merger+'/'+layer+'/means.csv')
    
    
    if gene_selection != 'all':
        for layer in layers:
            df_dict[layer] = { gene: df_dict[layer][gene] for gene in gene_selection }
            bool_dict[layer] = { gene: bool_dict[layer][gene] for gene in gene_selection }
            count_dict[layer] = { gene: count_dict[layer][gene] for gene in gene_selection }
            
            for CI in ['low_CI','up_CI']:
                CI_dict[layer][CI] = { gene: CI_dict[layer][CI][gene] for gene in gene_selection }
    
    
    cell_boundaries=pd.read_csv('data_files/confidence_intervals/'+cell_line+'/merged_results/'+target_merger+'/cell_boundaries.csv',header=None,index_col=0)
    boundary_dict={}
    for phase in cell_boundaries.index:
        boundary_dict[phase]=cell_boundaries.loc[[phase]].values[0][0]

    return df_dict,CI_dict,bool_dict,count_dict,boundary_dict


def get_CI_data_old (cell_line, replicates, layers, target_merger, single_rep=False):
    """
    Function which retrieves confidence interval related data. It retrieves the means of the merged replicates
    the confidence intervals of the merged replicates, the boolean values for each cell that indicate
    if a cell is 'accepted' based on the confidence interval, the count values, and the boundary
    information indicating the start and end of each cell cycle phase.
    
    Function written by Yohan Lefol

    Parameters
    ----------
    cell_line : string
        The cell line being used.
    replicates : string
        The replicates being used.
    layers : string
        The layers used (almost always spliced and unspliced).
    target_merger : string
        The folder of the merged replicates (ex: A_B).
    single_rep : boolean
        Boolean indicating if the dat comes from the merged CIs or individual CIs
        This changes the path where the data is retrieved as well as if the couts need to be calculated

    Returns
    -------
    df_dict : dictionnary
        dictionnary of dataframes containing the mean values of the replicates.
    CI_dict : dictionnary
        dictionnary containing the upper and lower confidence intervals of the replicates.
    bool_dict : dictionnary
        dictionnary containing the boolean values (accept value or reject) based on confidence intervals.
    count_dict : dictionnary
        dictionnary containing the count values (+1,0,-1) for the replicates.
    boundary_dict : dictionnary
        dicitonnary containing the cell boundaries for G1, S, and G2M.


    """
    df_dict={}
    CI_dict={}
    bool_dict={}
    count_dict={}
    for layer in layers:
        if single_rep==False:
            bool_dict[layer]=pd.read_csv('data_files/confidence_intervals/'+cell_line+'/merged_results/'+target_merger+'/'+layer+'/bool_'+layer+'.csv')
            count_dict[layer]=pd.read_csv('data_files/confidence_intervals/'+cell_line+'/merged_results/'+target_merger+'/'+layer+'/counts_'+layer+'.csv')
    
        CI_dict[layer]={}
        for CI in ['low_CI','up_CI']:
            if single_rep==True:
                CI_dict[layer][CI]=pd.read_csv('data_files/confidence_intervals/'+cell_line+'/'+target_merger+'/'+layer+'/'+CI+'.csv')
            else:
                CI_dict[layer][CI]=pd.read_csv('data_files/confidence_intervals/'+cell_line+'/merged_results/'+target_merger+'/'+layer+'/'+CI+'.csv')
        
        
        if single_rep==True:
            df_dict[str(layer+'_'+target_merger)]=pd.read_csv('data_files/confidence_intervals/'+cell_line+'/'+target_merger+'/'+layer+'/means.csv')
        else:
            df_dict[str(layer+'_'+target_merger)]=pd.read_csv('data_files/confidence_intervals/'+cell_line+'/merged_results/'+target_merger+'/'+layer+'/combined_mean.csv')
    
    
    if single_rep==True:
        cell_boundaries = pd.read_csv('data_files/boundary_data/'+cell_line+'_'+target_merger+'_boundaries.csv',header=None,index_col=0)
    else:
        smallest_rep = find_smallest_replicate(df_dict)
        cell_boundaries = pd.read_csv('data_files/boundary_data/'+cell_line+'_'+smallest_rep+'_boundaries.csv',header=None,index_col=0)

    
    
    boundary_dict={}
    for phase in cell_boundaries.index:
        boundary_dict[phase]=cell_boundaries.loc[[phase]].values[0][0]
    

    
    #Calculate count_dict and bool_dict
    if single_rep==True:
        count_dict={}
        bool_dict={}
        for layer in layers:
            count_dict[layer]={}
            bool_dict[layer]={}
            val_df=CI_dict[layer]['up_CI'].copy()
            for gene in CI_dict[layer]['up_CI'].keys():
            
                neg_vals=np.where(CI_dict[layer]['up_CI'][gene]<0)[0]
                pos_vals=np.where(CI_dict[layer]['low_CI'][gene]>0)[0]
                
                #Set all values to 0, will then be swapped as non-zeros if necessary
                val_df[gene][0:]=0
                if len(neg_vals)!=0:
                    val_df[gene][neg_vals]=-1
                if len(pos_vals)!=0:
                    val_df[gene][pos_vals]=1
                val_df[gene]=pd.Series(val_df[gene])
            count_dict[layer]=val_df
            bool_df=val_df.replace(-1,True)
            bool_df=bool_df.replace(1,True)
            bool_df=bool_df.replace(0,False)
            bool_dict[layer]=bool_df
        #Calculate boolean values    
        
 
        
        
    else:
        bool_dict=balance_dataframes(bool_dict)
        for key in CI_dict.keys():
            CI_dict[key]=balance_dataframes(CI_dict[key])
    # for key in bool_dict.keys():
    #     bool_dict[key]=balance_dataframes(bool_dict[key])
    
    #Format velocity keys
    df_dict['spliced'] = df_dict.pop('spliced_'+target_merger)
    df_dict['unspliced'] = df_dict.pop('unspliced_'+target_merger)
    
    return df_dict,CI_dict,bool_dict,count_dict,boundary_dict

def get_vlm_values(cell_line,layers,target_folder):
    """
    Simple fetch function which retrieves the spliced and unspliced values for each gene 
    of a replicates cell line. The retrieved values can be merged if required
    
    Function written by Yohan Lefol

    Parameters
    ----------
    cell_line : string
        String indicating the cell line being used.
    layers : list of strings
        Indicates the layers (Usually spliced and unspliced).
    target_folder : string
        String indicating the folder where the values are located

    Returns
    -------
    df_dict : dictionnary
        A dictionnary containing the retireved values.

    """
    df_dict={}
    for layer in layers:
        df_dict[layer]=pd.read_csv('data_files/confidence_intervals/'+cell_line+'/merged_results/'+target_folder+'/'+layer+'/vlm_means.csv')
        
    return df_dict


def log10_dta(delay_df,key):
    """
    performs the log10 transformation of a key of a dictionnary or pandas dataframe.
    The function accounts for the possibility of 0s and negative values in the 
    list to be log transformes. The function log10 transforms and absolute value version 
    of the list, then goes through the list and swaps -INF with 0s as well as
    provides the negative values for indexes which orignially held negative values.
    
    Function written by Yohan Lefol

    Parameters
    ----------
    delay_df : pandas dataframe or dictionnary
        dictionnary or dataframe containing a list of values ot be log10 transformed.
    key : string
        the string to the key or column of the delay_df to be log10 transformed.

    Returns
    -------
    new_dta : numpy nd array
        The log10 transformed data.

    """
    if key==None:
        og_arr=np.asarray(delay_df)
    else:
        og_arr=np.asarray(delay_df[key])
    test_dta=np.log10(abs(og_arr))
    
    new_dta=[]
    for idx,val in enumerate(test_dta):
        if val==-np.inf:
            new_dta.append(0)
        elif og_arr[idx]<0:
            new_dta.append(-val)
        else:
            new_dta.append(val)
            
    new_dta=np.asarray(new_dta)
    return new_dta


def moving_average(array, window_size=200,orientation=None) :
    """
    A function which smoothes data using the moving average technique.
    In the case of G1 orientation, an offset will need to be corrected
    
    Function written by Yohan Lefol

    Parameters
    ----------
    array : numpy nd.array
        The array containing the data that will be smoothed.
    window_size : int, optional
        The amounf of values used at once for the smoothing. The default is 200.
    orientation : string, optional
        either G1 or G2M to indicate the orientation of the data. The default is None.

    Returns
    -------
    moving_averages : numpy nd.array
        The array containing the smoothed data.

    """
    # #for it to be circular, need to add the n first points to end
    array=np.concatenate([array,array[:window_size-1]])
    numbers_series=pd.Series(array)
    windows = numbers_series.rolling(window_size)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    without_nans = moving_averages_list[window_size - 1:]

    moving_averages=np.asarray(without_nans)
    if orientation=='G1':
        moving_averages=np.roll(moving_averages,int(window_size/2))

        
    return moving_averages



#%%Extras ?

def create_chrM_list(gtf_path):
    """
    Uses pyranges to extract all mitochondrial genes from a gtf file
    return the list of mitochondrial genes found
    Caution: loading gtf files can be a bit heavy on the RAM

    Parameters
    ----------
    gtf_path : string
        path to the Gene Transfer File.

    Returns
    -------
    MT_genes_list : list
        The list of mitochondrial genes found in the gtf file.

    """
    #Reads the gtf file using PyRanges
    gr=pr.read_gtf(gtf_path)
    
    #Subsets the pyrange file to only include the mitochondrial chromosome
    gr=gr["chrM"]
    
    MT_genes_list=[]
    #Removes the 'MT-' and ensures that the final list contains unique gene_names
    for i in gr.gene_name:
        x=i.split('-')
        if x[1] not in MT_genes_list:
            MT_genes_list.append(x[1])
    
    return MT_genes_list