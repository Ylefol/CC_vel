#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 11:53:14 2021

@author: yohanl
"""

import pandas as pd
import numpy as np
import pyranges as pr
import scipy.stats as stats

import statistics
from math import sqrt

import itertools
import os
import gc

# import snake_utils as my_utils
#The 'snake_analysis_functions' can be called either through snakemake or 
#through an analysis script. This changes the directore in which the utils
#will be located. This block account for either option
try:
    from snake_functions import snake_utils as my_utils
except:
    from snake_scripts.snake_functions import snake_utils as my_utils


import ptitprince as pt
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import ConnectionPatch

#%% Processing and sorting



def create_gene_ranking(count_dict,mean_dict,CI_dict,Sc_dict,number_of_iters):
    """
    Function which establishes a gene ranking based on a z-score. All genes are 
    included in the resulting dataframe, however only genes which have both a 
    negative and positive spliced and unspliced velocity are qualified to be ranked,
    otherwise a value (and rank) of 0 will be given.
    
    If a gene qualifies to be ranked, a score is given for four categories,
    a low CI and up CI for spliced and unspliced. The score is the minimum or maximum CI
    (if upper or lower CI respetively) divided by the velocity value of that location substracted by
    the found CI.
    
    A high score is given based on the highest value of the 4 categories for each gene.
    
    The dataframe is then sorted (descending) using this high score
    
    Function written by Yohan Lefol

    Parameters
    ----------
    count_dict : dictionnary
        A dictionnary containing the count values for all genes at spliced and unspliced.
    mean_dict : dictionnary
        A dictionnary containing the velocity values for each gene at spliced and unspliced.
    CI_dict : dictionnary
        A dictionnary containing the upper and lower confidence interval for each gene at spliced and unspliced.
    Sc_dict : dictionnary
        A dictionnary containing the merged standard deviations used in the calculation of the t-test
    number_of_iters : int
        The number of iterations performed on the data, this is used as the number of observations for the calculation
        of the t-test

    Returns
    -------
    rank_df : pandas dataframe
        A data frame containing the category score and high score for each gene.

    """

    #Establish which genes qualify for ranking
    genes_to_rank=[]
    for gene in count_dict['spliced'].keys():
        if 1.0 in list(count_dict['spliced'][gene]) and -1.0 in list(count_dict['spliced'][gene]):
            if 1.0 in list(count_dict['unspliced'][gene]) and -1.0 in list(count_dict['unspliced'][gene]):
                genes_to_rank.append(gene)
    
    #Calculates the score and finds the location for at each layer, for each confidence interval, for each gene, 
    rank_dict={}
    rank_dict['gene_name']=list(CI_dict['spliced']['up_CI'].keys())
    for layer in CI_dict.keys():
        for CI in CI_dict[layer].keys():
            rank_dict[layer+'_'+CI]=[]
            for gene in CI_dict[layer][CI].keys():
                if gene in genes_to_rank: #If qualified for ranking, calculate score
                    #Get total cell number and standard deviation
                    # stand_dev=np.std(mean_dict[layer][gene])
                    
                    if CI=='low_CI':  
                        target_peak=max(mean_dict[layer][gene])

                    else:#CI is up_CI
                        target_peak=min(mean_dict[layer][gene])
                    
                    #Find idx to extract correct combined standard dev (different per cell)
                    target_idx = np.where(mean_dict[layer][gene] == target_peak)[0][0]
                    stand_dev=Sc_dict[layer][gene][target_idx]
                    
                    my_score=target_peak/(stand_dev/sqrt(number_of_iters))
                    rank_dict[layer+'_'+CI].append(my_score)
                else: #Gene is not qualified for ranking
                    rank_dict[layer+'_'+CI].append(0)
    
    #Convert to pandas
    rank_df=pd.DataFrame.from_dict(rank_dict)
    #Create the high score column
    rank_df['high_score']=rank_df[['spliced_low_CI','spliced_up_CI','unspliced_low_CI','unspliced_up_CI']].apply(np.max,axis=1)
    
    #Sort the dataframe by high score before returning
    rank_df=rank_df.sort_values('high_score',ascending=False)

    return rank_df


def identify_vel_start_phases(count_dict,mean_dict,CI_dict,boundary_dict):
    """
    Function which identifies the phase in which a velocity starts changing.
    The function will iterate over genes which can be ranked, that is genes which
    have a confirmed positive and negative velocity for both the unspliced 
    and spliced values.
    
    For each confidence interval (confirmed positive and negative velocity for 
    spliced and unspliced), the function identifies the moment (phase) where the 
    beginning of either the positive or negative velocity occurs.
    
    The function then records this into a dataframe and returns it. Genes which do
    not qualify for ranking are given an NA as their found phase.
    
    Function written by Yohan Lefol

    Parameters
    ----------
    count_dict : dictionnary
        A dictionnary containing the count values for all genes at spliced and unspliced.
    mean_dict : dictionnary
        A dictionnary containing the velocity values for each gene at spliced and unspliced.
    CI_dict : dictionnary
        A dictionnary containing the upper and lower confidence interval for each gene at spliced and unspliced.
    boundary_dict : dictionnary
        A dictionnary indicating the cell boundaries for the cell cycle phases.

    Returns
    -------
    rank_df : pandas dataframe
        A data frame containing the category score, category phase and high score for each gene.

    """
    #Establish which genes qualify for ranking
    genes_to_rank=[]
    for gene in count_dict['spliced'].keys():
        if 1.0 in list(count_dict['spliced'][gene]) and -1.0 in list(count_dict['spliced'][gene]):
            if 1.0 in list(count_dict['unspliced'][gene]) and -1.0 in list(count_dict['unspliced'][gene]):
                genes_to_rank.append(gene)
    
    #Create list of phases to find location of score
    boundary_dict_sort={k: v for k, v in sorted(boundary_dict.items(), key=lambda item: item[1])}
    if list(boundary_dict_sort.keys())[0]=='G2M' and list(boundary_dict_sort.keys())[1]=='G1':
        G2M_list=['G2M']*boundary_dict['G1']
        G1_list=['G1']*(boundary_dict['S']-boundary_dict['G1'])
        S_list=['S']*(len(mean_dict['spliced'][list(mean_dict['spliced'].keys())[0]])-boundary_dict['S'])
        phase_list=G2M_list+G1_list+S_list
    else:
        G2M_list=['G2M']*boundary_dict['S']
        S_list=['S']*(boundary_dict['G1']-boundary_dict['S'])
        G1_list=['G1']*(len(mean_dict['spliced'][list(mean_dict['spliced'].keys())[0]])-boundary_dict['G1'])
        phase_list=G2M_list+S_list+G1_list
        
    #Calculates the score and finds the location for at each layer, for each confidence interval, for each gene, 
    phase_dict={}
    phase_dict['gene_name']=list(CI_dict['spliced']['up_CI'].keys())
    for layer in CI_dict.keys():
        for CI in CI_dict[layer].keys():
            phase_dict[layer+'_'+CI+'_phase']=[]
            for gene in CI_dict[layer][CI].keys():
                if gene in genes_to_rank:
                    if CI=='low_CI':  
                        phase_search_val=max(mean_dict[layer][gene])   
                        phase_search_idx=np.where(CI_dict[layer][CI][gene]>0)[0][0] #Take first val above 0
                    else:#CI is up_CI
                        phase_search_val=min(mean_dict[layer][gene])
                        phase_search_idx=np.where(CI_dict[layer][CI][gene]<0)[0][0] #Take first val below 0
                    
                    #Find location
                    target_phase=phase_list[np.where(mean_dict[layer][gene]==phase_search_val)[0][0]]
                    target_phase=phase_list[phase_search_idx]
                    
                    phase_dict[layer+'_'+CI+'_phase'].append(target_phase)
                else: #Gene is not qualified for ranking
                    phase_dict[layer+'_'+CI+'_phase'].append('NA')

    #Convert to pandas
    phase_df=pd.DataFrame.from_dict(phase_dict)
    return phase_df

def find_gaps(num_cells,index_array_1,inc,index_array_2=None):
    """
    Function that creates a dictionnary of gap spacings. The dictionnary contains
    two keys (start and end). It looks through index_array_1 and records when
    the sequence of index no longer matches (ex: 1,2,3,7,8,9) would result
    in a gap with 'start' as 4 and 'end' as 6. 
    
    index_list_2 is used a precaution in case the gap overlaps with the start point
    Assume a list of 0 to 10, index_list_1 may be (6,7,8,9) which would indicate
    a gap starting at 10 and going to 5. This is possible as we assume the
    dataset to be circular.
    
    This event is unlikely to occur but it is possible
    
    Function written by Yohan Lefol

    Parameters
    ----------
    num_cells : integer
        The number of cells in the array.
    index_array_1 : np.ndarray
        A list of indexes that will be analyzed for gaps.
    inc : int
        The number of allowed differences before a gap is 'started'.
    index_array_2 : np.ndarray, optional
        Used to check if the gap goes beyond the start/end point. The default is None.

    Returns
    -------
    gap_dict : dictionnary
        A dictionnary containing two keys (start and end) which represent the
        start and end values of all gaps found in index_list_1.

    """
    index_array_1=sorted(index_array_1)
    gap_dict={}
    gap_dict['start']=[]
    gap_dict['end']=[]
    good_val_found=False
    for idx,val in enumerate(index_array_1):
        if idx==len(index_array_1)-1:#end of list
            if val-inc!=index_array_1[idx-inc]:#Gap on last num
                if index_array_1[0]==1 and index_array_1[-1]==num_cells:
                    gap_dict['start'].append(index_array_1[idx])
                    for dex,item in enumerate(index_array_1):
                        if val+inc!=index_array_1[idx+inc]:#Gap found
                            gap_dict['end'].append(index_array_1[idx])
            else:
                if index_array_1[0]==1 and index_array_1[-1]==num_cells:
                    for dex,item in enumerate(index_array_1):
                        if val+inc!=index_array_1[idx+inc]:#Gap found
                            gap_dict['end'].append(index_array_1[idx])
                else:
                    gap_dict['end'].append(index_array_1[idx])
                            
            break
        if val+inc==index_array_1[idx+inc]:
            if good_val_found==False:#First one of series detected
                gap_dict['start'].append(index_array_1[idx])
            good_val_found=True
        else:
            if good_val_found==True:#Else it isn't a gap since there was no good val before hand
                gap_dict['end'].append(index_array_1[idx])
                good_val_found=False
    if index_array_2 !=None:
        for idx,val in enumerate(gap_dict['start']):
            if val-1 in index_array_2[0]: #Means the start point was spliced, no bueno
                #Need to remove it
                del gap_dict['start'][idx]
                del gap_dict['end'][idx]
    
    return gap_dict


def filter_delay_array(gap_array,comp_array):
    """
    Function which filters delay/gap arrays for the delay calculation.
    The function checks if the found starting and end point of an array overlap
    with another array. The function is used to check if an array of cells for spliced 
    values overlaps with an array of cells for unspliced and vice-versas
    
    If the function finds no overlap, it removes that segment of the array. This either 
    leaves the array with cell indexes whihc overlap with the comparison array, or empty
    if no overlap is found.
    
    Function written by Yohan Lefol

    Parameters
    ----------
    gap_array : numpy.ndarray
        The array which will be filtered.
    comp_array : numpy.ndarray
        The array used to check for overlap in the 'gap_array'.

    Returns
    -------
    gap_array : numpy.ndarray
        The processed/filtered array.

    """
    #Find all gaps (breaks in the sequence of indexes)
    my_gaps=find_gaps(len(gap_array), gap_array[0],1)
    
    #Prepare lists for removal of start and end values if needed
    # rem_start_lst=[]
    # rem_end_lst=[]
    for idx in range(len(my_gaps['start'])):#Iterate over all possible 'starts'
        gap_start=my_gaps['start'][idx]
        gap_end=my_gaps['end'][idx]
        loc_start=np.where(gap_array[0]==gap_start)[0][0]
        loc_end=np.where(gap_array[0]==gap_end)[0][0]
        split_gap_arr=gap_array[0][loc_start:loc_end]#Create an array of only the values within the iterated start and end
        for num in split_gap_arr:
            if num in comp_array[0]:#If there is an overlap, this array (start and end) is validated
                break
            if num==split_gap_arr[-1]:#Reached the end, no overlap, remove indexes of start and end
                gap_array=np.delete(gap_array, list(range(loc_start,loc_end+1)), axis=1)
                
                #Quality control. In some cases, if there is a single cell peak
                #It will not be seen by gaps, and therefore be left in the array
                #this will remove that single cell peak
                if len(gap_array[0])==1:
                    gap_array=np.delete(gap_array,0,axis=1)
                # rem_start_lst.append(gap_start)
                # rem_end_lst.append(gap_end)
    
    # #Removal of start and end values here to not affect for loop
    # if len(rem_start_lst)>0:
    #     my_gaps['start'].remove(rem_start_lst)
    #     my_gaps['end'].remove(rem_end_lst)
    
    #Return start and end values along with filtered array
    return (gap_array)

def find_delays(count_val,spli_arr,unspli_arr):
    """
    Function which calculates the delay for either +1 value or -1 values.
    
    Two delays can be found for each count value. his function creates the necessary 
    subsets of indexes for spliced and unspliced, calls the function to filter them,
    then calculates the increase delay and decrease delay.
    
    Function written by Yohan Lefol
    
    Parameters
    ----------
    count_val : int
        Either +1 or -1, it indicates which types of delays are being searched for
        if +1 it will find increase to +1 and decrease to 0. If it is -1 it will find
        decrease to -1 and increase to .
    spli_arr : numpy.ndarray
        The array containing the filtered spliced cells.
    unspli_arr : numpy.ndarray
        The array containing the filtered unspliced cells.

    Returns
    -------
    found_inc : int
        The found value for increase (either to 0 or +1).
    found_dec : int
        The found value for decrease (either to 0 or -1).

    """
    spli_count=np.where(spli_arr==count_val)
    unspli_count=np.where(unspli_arr==count_val)
    
    #Only one index will reveal nothing, therefore terminate now
    if len(spli_count[0])<=1 or len(unspli_count[0])<=1:
        return(0,0)
    
    #Find relevant gaps, and filter the array
    spli_count=filter_delay_array(spli_count, unspli_count)
    unspli_count=filter_delay_array(unspli_count, spli_count)
    
    #Only one increase should exist from now on, so gota find the delay
    if len(spli_count[0])==0 or len(unspli_count[0])==0:#If one is zero, the other will also be zero
        return(0,0)
    else:
        flip_spli_pos_one=np.flip(spli_count)
        flip_unspli_pos_one=np.flip(unspli_count)
        if count_val==1:
            found_inc=spli_count[0][0]-unspli_count[0][0]
            found_dec=flip_spli_pos_one[0][0]-flip_unspli_pos_one[0][0]
        else:#If count val is -1, the increase and decrease calculation flip
            found_dec=spli_count[0][0]-unspli_count[0][0]
            found_inc=flip_spli_pos_one[0][0]-flip_unspli_pos_one[0][0]
    return found_inc,found_dec



def create_delay_dataframe(count_dict):
    """
    Function which creates a delay dataframe. A delay dataframe shows the 
    delay between an shift in unspliced followed by the same shift in spliced. 
    This results in a positive delay. Thereverse (shift in spliced followed by unspliced)
    creates a negative delay.
    
    Function written by Yohna Lefol

    Parameters
    ----------
    count_dict : dictionnary
        A dictionnary containing the count values (+1,0,-1) for each gene, at each cell,
        for each layer (spliced and unspliced).

    Returns
    -------
    delay_df : pandas dataframe.
        A dataframe containing the four relevant delay values for each gene of the count dict.

    """
    my_delay_dict={}
    my_delay_dict['gene_name']=count_dict['spliced'].keys()
    my_delay_dict['inc_to_0']=[]
    my_delay_dict['inc_to_+1']=[]
    my_delay_dict['dec_to_0']=[]
    my_delay_dict['dec_to_-1']=[]
    for gene in my_delay_dict['gene_name']:
        spli_arr=np.asarray(count_dict['spliced'][gene])
        unspli_arr=np.asarray(count_dict['unspliced'][gene])
        
        #If they don't match on a zero (might occur in single replicate analyses)
        if len(np.where((spli_arr==0)&(unspli_arr==0))[0])>0:
            roll_val=np.where((spli_arr==0)&(unspli_arr==0))[0][0]
        elif len(np.where((spli_arr==1)&(unspli_arr==1))[0])>0:
            roll_val=np.where((spli_arr==1)&(unspli_arr==1))[0][0]
        elif len(np.where((spli_arr==-1)&(unspli_arr==-1))[0])>0:
            roll_val=np.where((spli_arr==-1)&(unspli_arr==-1))[0][0]
        else:#No matches -- skip the gene
            continue
        spli_arr=np.roll(spli_arr,-roll_val)
        unspli_arr=np.roll(unspli_arr,-roll_val)
        
        #check 0-1
        inc_found,dec_found=find_delays(1,spli_arr,unspli_arr)
        my_delay_dict['inc_to_+1'].append(inc_found)
        my_delay_dict['dec_to_0'].append(dec_found)
        
        #check 0 to -1
        inc_found,dec_found=find_delays(-1,spli_arr,unspli_arr)
        my_delay_dict['inc_to_0'].append(inc_found)
        my_delay_dict['dec_to_-1'].append(dec_found)
        
        
    delay_df=pd.DataFrame.from_dict(my_delay_dict)
    return delay_df



def create_REAC_dict(exp_values,significant_genes,orientation='G1'):
    """
    Function which reads text files in a REAC_pathways folder (REACTOME folde). The text 
    files should contain the names of the genes within the reactome pathway of the 
    text files name
    
    The function will identify all common genes between the pathways and the expression
    data provided, it will then filter out any genes which are not significant. 
    It will then calculate the mean expression curve for each of the genes.
    The genes are stored in a dictionnary first split by REACTOME name, then by
    spliced and unspliced layers

    Parameters
    ----------
    exp_values : dictionnary
        Dictionnary containing the expression values of each gene for spliced and unspliced.
    significant_genes : list
        A list of genes which are considered significant, only genes that are significant
        are added to the final dictionnary.
    orientation : TYPE, optional
        DESCRIPTION. The default is 'G1'.

    Returns
    -------
    mean_exp_dict : dictionary
        Dictionary split by REAC name, the spliced and unspliced. The spliced and 
        unspliced layer will contain their respective 2D array where columns (axis 0) 
        are genes and rows (axis 1) are cells. Gene names are not preserved.

    """
    mean_exp_dict={}
    for REAC in os.listdir('data_files/REAC_pathways'):
        if str.endswith(REAC,'.txt')==True:
            name_REAC=str.split(REAC,'.')[0]
            gene_list=pd.read_csv('data_files/REAC_pathways/'+REAC,header=None)
            first_gene=True
            for gene in gene_list[0]:
                if gene in exp_values['spliced'] and gene in significant_genes:
                    # my_x_axis=np.arange(0,len(exp_values['spliced'][gene]))
                    # spli_mean_array,unspli_mean_array=smooth_layer_no_vlm(x_axis=my_x_axis,bin_size=100,window_size=200,spliced_array=exp_values['spliced'][gene],unspliced_array=exp_values['unspliced'][gene],orientation=orientation)
                    spli_mean_array=exp_values['spliced'][gene]
                    unspli_mean_array=exp_values['unspliced'][gene]
                    if first_gene == True:
                        spli_tot_array=spli_mean_array
                        unspli_tot_array=unspli_mean_array
                        first_gene=False
                    else:
                        spli_tot_array=np.vstack((spli_tot_array, spli_mean_array))
                        unspli_tot_array=np.vstack((unspli_tot_array, unspli_mean_array))
            
            if (first_gene==False):#Check if any genes were found
                mean_exp_dict[name_REAC]={}
                mean_exp_dict[name_REAC]['spliced']=spli_tot_array
                mean_exp_dict[name_REAC]['unspliced']=unspli_tot_array
    return mean_exp_dict



def calculate_peak_dict(exp_values,significant_genes,boundary_dict,layer_to_use='spliced',orientation='G1'):
    """
    Function which identifies the expression peaks of each significant gene based 
    on the genes mean expression throughout the cell cycle.

    Parameters
    ----------
    exp_values : dictionnary
        Dictionnary containing the expression values of each gene for spliced and unspliced.
    significant_genes : list
        A list of genes which are considered significant, only genes that are significant
        are added to the final dictionnary.
    boundary_dict : dictionnary
        A dictionnary containing the cell cycle boundaries in terms of cell number
    layer_to_use : string, optional
        Either spliced or unspliced to identify the layer of interest. The default is 'spliced'.
    orientation : string, optional
        Either G1 or G2M to indicate the orientation of the cell cycle. The default is 'G1'.

    Returns
    -------
    final_df : TYPE
        dataframe containing the gene name, associated peak expression, and the 
        phase ot which it has been mapped to.

    """
    exp_peak_dict={}
    for gene in significant_genes:
        # x_for_param=np.arange(0,len(exp_values['spliced'][gene]))
        # spli_mean_array,unspli_mean_array=smooth_layer_no_vlm(x_axis=x_for_param,bin_size=100,window_size=200,spliced_array=exp_values['spliced'][gene],unspliced_array=exp_values['unspliced'][gene],orientation=orientation)
        spli_mean_array=exp_values['spliced']
        unspli_mean_array=exp_values['unspliced']
        if layer_to_use=='spliced':
            arr_interest=spli_mean_array
        elif layer_to_use=='unspliced':
            arr_interest=unspli_mean_array
        
        found_index=np.where(arr_interest==max(arr_interest))[0][0]
        if found_index >=boundary_dict['G2M'] and found_index<boundary_dict['G1']:
            found_phase='G2M'
        elif found_index >= boundary_dict['G1'] and found_index<boundary_dict['S']:
            found_phase='G1'
        else:
            found_phase='S'
        
        exp_peak_dict[gene]=[max(arr_interest),found_phase]

    final_df=pd.DataFrame.from_dict(exp_peak_dict,orient='index',columns=['peak_expression', 'phase'])
    final_df[['gene_name']]=final_df.index
    final_df=final_df[['gene_name','peak_expression','phase']]
    return final_df


def create_cell_line_replicate_cc_df(cc_path,cell_line_dict):
    """
    Function which creates a pandas dataframe containing all known cell cycle genes
    for the G1, S, and G2/M phases. It then extracts the spliced values for each of the
    submitted cell line / replicates and adds those gene values to the dataframe.

    Parameters
    ----------
    cc_path : string
        location of the known cell cycle genes file.
    cell_line_dict : dictionnary
        dictionnary where key names are cell lines and values are lists of replicate names.

    Returns
    -------
    A pandas dataframe containing cell cycle gene names, associated cell cycle phase,
    and expression data (spliced) for the queried cell lines and replicates.

    """
    cc_df=pd.read_csv(cc_path)
    cc_df=cc_df[['gene','phase']]
    cc_df.columns=['Gene','phase']
    cc_df=cc_df[cc_df.phase.isin(['S','G2/M','G1'])]
    
    new_cc_df=cc_df.copy()
    for cell_line in list(cell_line_dict.keys()):
        for replicate in cell_line_dict[cell_line]:
            
            #Load relevant data to cell line and replicate
            import scanpy as sc
            adata = sc.read_loom('data_files/phase_reassigned/CC_'+cell_line+'_'+replicate+'.loom',X_name="")
            spli_df=adata.to_df(layer='spliced')
                        
            
            col_name=cell_line+'_'+replicate
            spli_df=spli_df.mean().to_frame()
            spli_df['Gene']=list(spli_df.index)
            spli_df.columns=[col_name,'Gene']
            spli_df.index.name=''
            
            new_cc_df=pd.merge(new_cc_df,spli_df,how='outer')
    return(new_cc_df)

def create_cell_line_variance_dictionnary(cell_line_dict,target_layer='spliced'):
    """
    Function which calculates the variance for selected cell lines. The function takes in
    a dictionary where the key is the name of the cell line and the value is the 
    folder (merged replicates) to be used to calculate the variance

    Parameters
    ----------
    cell_line_dict : dictionnary
        Dictionnary containing cell line name (key) and folder to use (value).
    target_layer : list, optional
        list with the layers to use.. The default is ['spliced'].

    Returns
    -------
    Dictionnary containing the variance values for the selected cell lines.

    """
    cell_line_var_dict={}
    for cell_line in list(cell_line_dict.keys()):
        folder_to_use=cell_line_dict[cell_line]
        mean_dict,CI_dict,bool_dict,count_dict,boundary_dict=my_utils.get_CI_data (cell_line, [target_layer], folder_to_use)
        # vlm_dict=my_utils.get_vlm_values(cell_line, target_layer,folder_to_use )
        vlm_mean_dict=my_utils.get_vlm_values(cell_line, [target_layer],folder_to_use,get_mean=True)
    
        gene_matrix=np.log10(vlm_mean_dict[target_layer])
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
    return(cell_line_var_dict)


def cell_line_var_REAC(vlm_mean_dict,boundary_dict):
    """
    Function which creates REACTOME plots using variance, both normal variance and 
    log10 variance. The function is intended to aid in identifying the best form of
    variance to use in order to filter for house keeping genes

    Parameters
    ----------
    vlm_mean_dict : dictionnary
        dictionnary containing the smoothed expression values for spliced data.

    Returns
    -------
    None.

    """
    dta_res_list=[]
    gene_matrix=vlm_mean_dict['spliced'] 
    log10_matrix=np.log10(gene_matrix)
    for file_name in os.listdir('data_files/REAC_pathways'):
        gene_list=pd.read_csv('data_files/REAC_pathways/'+file_name,header=None)
        set1 = set(list(gene_list[0]))
        set2 = set(list(vlm_mean_dict['spliced'].keys()))
        intersect = list(set1 & set2)
    
        save_name=file_name.split('_')[0]
        plot_phase_exp_raincloud(gene_matrix,intersect,boundary_dict,file_name,'analysis_results/'+save_name+'.png',False,False)
        plot_phase_exp_raincloud(gene_matrix,intersect,boundary_dict,file_name+'_log10','analysis_results/'+save_name+'_log10.png',False,True)
        
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
    

#%% Plotting calculations

def minimal_xticks(start, end):
    """
    Determines the minimum number of x ticks possible
    
    Function originates from DentateGyrus notebook for Velocyto

    Parameters
    ----------
    start : int
        The first tick.
    end : int
        The last tick.

    Returns
    -------
    None.

    """
    end_ = np.around(end, -int(np.log10(end))+1)
    xlims = np.linspace(start, end_, 5)
    xlims_tx = [""]*len(xlims)
    xlims_tx[0], xlims_tx[-1] = f"{xlims[0]:.0f}", f"{xlims[-1]:.02f}"
    plt.xticks(xlims, xlims_tx)#,fontsize=15)

    
def minimal_yticks(start, end):
    """
    Determines the minimum number of y ticks possible
    
    Function originates from DentateGyrus notebook for Velocyto

    Parameters
    ----------
    start : int
        The first tick.
    end : int
        The last tick.

    Returns
    -------
    None.

    """
    end_ = np.around(end, -int(np.log10(end))+1)
    ylims = np.linspace(start, end_, 5)
    ylims_tx = [""]*len(ylims)
    ylims_tx[0], ylims_tx[-1] = f"{ylims[0]:.0f}", f"{ylims[-1]:.02f}"
    plt.yticks(ylims, ylims_tx)#,fontsize=15)


def smooth_layer_no_vlm(x_axis,bin_size, window_size, spliced_array, unspliced_array, orientation):
    """
    Function which calculates the smoothed date of a spliced and unspliced array.
    This function already exists, but depends on a velocyto object. This function is
    a non velocyto object dependant version.
    
    Function written by Yohan Lefol

    Parameters
    ----------
    x_axis : numpy ndarray or list
        A 0 to n list or array showing the number of cells in the dataset.
    bin_size : int
        The size of the bins used for smoothing.
    window_size : int
        The size of the window used for the moving average.
    spliced_array : numpy ndarray
        the array containing the spliced values.
    unspliced_array : numpy ndarray
        The array containing the unspliced values.
    orientation : string
        Either G1 or G2M to indicate the orientation of the data based on the cell cycle.

    Returns
    -------
    spli_mean_array : numpy ndarray
        The smoothed spliced data.
    unspli_mean_array : numpy ndarray
        The smoothed unspliced data.

    """
    num_bin=int(len(x_axis)/bin_size)
    spli_mean_list=[]
    unspli_mean_list=[]
    for order in range(num_bin):
        if order==0:    #First iteration
            spli_mean_list.append(np.mean(spliced_array[0:bin_size]))
            unspli_mean_list.append(np.mean(unspliced_array[0:bin_size]))
        else:  
            spli_mean_list.append(np.mean(spliced_array[order*bin_size:(order+1)*bin_size]))              
            unspli_mean_list.append(np.mean(unspliced_array[order*bin_size:(order+1)*bin_size]))
    
    
    last_index_check=num_bin*bin_size
    if len(x_axis)%bin_size==0:
        last_index_check=int(last_index_check-(bin_size/2))

    last_val_spli=np.mean(spliced_array[last_index_check:len(x_axis)])
    last_val_unspli=np.mean(unspliced_array[last_index_check:len(x_axis)])
    
    spli_mean_list.append(last_val_spli)
    unspli_mean_list.append(last_val_unspli)
    
    spli_mean_list.insert(0,last_val_spli)
    unspli_mean_list.insert(0,last_val_unspli)

    # These two for loops extend the means as desired.
    for idx,val in enumerate(spli_mean_list):
        if idx==0:#First iterattion:
            spli_mean_array=np.linspace(start=val,stop=spli_mean_list[idx+1],num=bin_size)
        else:
            if idx!=len(spli_mean_list)-1:
                if idx == len(spli_mean_list)-2:#Last iteration
                    spli_mean_array=np.concatenate([spli_mean_array,np.linspace(start=val,stop=spli_mean_list[idx+1],num=len(x_axis)-(num_bin*bin_size))])
                else:
                    spli_mean_array=np.concatenate([spli_mean_array,np.linspace(start=val,stop=spli_mean_list[idx+1],num=bin_size)])
    for idx,val in enumerate(unspli_mean_list):
        if idx==0:#First iterattion:
            unspli_mean_array=np.linspace(start=val,stop=unspli_mean_list[idx+1],num=bin_size)
        else:
            if idx != len(unspli_mean_list)-1:
                if idx == len(unspli_mean_list)-2:
                    unspli_mean_array=np.concatenate([unspli_mean_array,np.linspace(start=val,stop=unspli_mean_list[idx+1],num=len(x_axis)-(num_bin*bin_size))])
                else:
                    unspli_mean_array=np.concatenate([unspli_mean_array,np.linspace(start=val,stop=unspli_mean_list[idx+1],num=bin_size)])
    
    spli_mean_array=my_utils.moving_average(spli_mean_array,window_size=window_size,orientation=orientation)
    unspli_mean_array=my_utils.moving_average(unspli_mean_array,window_size=window_size,orientation=orientation)

    return spli_mean_array,unspli_mean_array

def plot_ax_lines_phase_portrait_no_vlm(x_axis,orientation,boundary_dict,v_lines=True,y_axis_loc=0):
    """
    Function which plots axis lines and colored horizontal lines to illustrate
    the different cell cycle phases of the dataset
    
    Function written by Yohan Lfeol

    Parameters
    ----------
    x_axis : list or numpy ndarray
        A 0 to n list or array showing the number of cell sin the dataste.
    orientation : string
        Either G1 or G2M to indicate the orientation of the data in regards to the cell cycle.
    boundary_dict : dictionnary
        Dictionnary containing the cell cycle boundaries for each phase (G1, S, G2M).
    v_lines : boolean
        Indicates if the vertical black lines between cc phases should be plotted
    y_axis_loc : int
        The location on the y axis where the horizontal lines will be printed

    Returns
    -------
    None.

    """
    
    #Plot vertical ax lines to clearly split the plot into the three cell cycles
    if v_lines==True:
        for key,order in boundary_dict.items():
            plt.axvline(order[0],c='k',lw=2)
    
    #Sorts based on the order
    phase_order=sorted(boundary_dict,key=lambda k: boundary_dict[k][0])
    #Extracts the start and end point of each phase, then plots them as horizontal lines 
    #At the 0 point of the plot (bottom of the plot)
    for inc,p in enumerate(phase_order):
        if orientation == 'G1':
            if p=='G1':
                color_used=boundary_dict['G2M'][1]
            elif p== 'S':
                color_used=boundary_dict['G1'][1]
            else:
                color_used=boundary_dict['S'][1]
        if orientation=='G2M':
            if p=='G1':
                color_used=boundary_dict['S'][1]
            elif p== 'S':
                color_used=boundary_dict['G2M'][1]
            else:
                color_used=boundary_dict['G1'][1] 
        if inc==0:
            plt.hlines(y_axis_loc,0,boundary_dict[p][0], colors=color_used, linestyles='solid',lw=6)
        else:
            plt.hlines(y_axis_loc,boundary_dict[phase_order[inc-1]][0],boundary_dict[p][0], colors=color_used, linestyles='solid',lw=6)
        
        if inc==2 and boundary_dict[p][0]<np.max(x_axis):
            # if orientation == 'G1':
            plt.hlines(y_axis_loc,boundary_dict[phase_order[inc]][0],np.max(x_axis), colors=boundary_dict[p][1], linestyles='solid',lw=6)
            # if orientation == 'G2M':
            #     plt.hlines(0,boundary_dict[phase_order[inc]][0],np.max(vlm.ca['new_order']), colors=boundary_dict[phase_order[0]][1], linestyles='solid',lw=8)

def create_gap_dict(bool_df,gene_name):
    """
    Function which finds the start and end of sets of booleans for a gene of interest
    The function finds and returns the data in a plottable format

    Function written by Yohan Lefol

    Parameters
    ----------
    bool_df : dictionnary
        dictionnary containing the boolean values for the merged replicates.
    gene_name : string
        The gene of interest.

    Returns
    -------
    gap_dict : dictionnary
        A dictionnary containing the index for the start and end of both True and False.

    """
    false_array=np.where(np.asarray(bool_df[gene_name])==False)
    true_array=np.where(np.asarray(bool_df[gene_name])==True)
    
    gap_dict={}
    gap_dict['start_true']=[]
    gap_dict['start_false']=[]
    gap_dict['end_true']=[]
    gap_dict['end_false']=[]
    if len(false_array[0])==0:
        gap_dict['start_true'].append(0)
        gap_dict['end_true'].append(np.max(true_array))
    elif len(true_array[0])==0:
        gap_dict['start_false'].append(0)
        gap_dict['end_false'].append(np.max(false_array))
    else:
        temp_list=[]
        for i in false_array[0]:
            if i+1 in true_array[0]:
                temp_list.append(i)
                gap_dict['start_false'].append(min(temp_list))
                gap_dict['end_false'].append(max(temp_list))
                temp_list=[]
            else:
                temp_list.append(i)
            if i>np.max(true_array):
                gap_dict['start_false'].append(i)
                gap_dict['end_false'].append(np.max(false_array))
                break
                
        temp_list=[]
        for i in true_array[0]:
            if i+1 in false_array[0]:
                temp_list.append(i)
                gap_dict['start_true'].append(min(temp_list))
                gap_dict['end_true'].append(max(temp_list))
                temp_list=[]
            else:
                temp_list.append(i)
            if i>np.max(false_array):
                gap_dict['start_true'].append(i)
                gap_dict['end_true'].append(np.max(true_array))
                break
            
    return gap_dict


def subset_the_dicts_merged_CI(gene_name,df_dict,bool_dict,CI_dict):
    """
    Function which stores all the relevant gene information into a single dictionnary
    for coding convenience
    
    Function written by Yohan Lefol
    

    Parameters
    ----------
    gene_name : string
        the gene of interest.
    df_dict : dictionnary
        dictionnary containing the merged means of the replicates.
    bool_dict : dictionnary
        dictionnary containing the boolean values of the merged replicates.
    CI_dict : dictionnary
        dictionnary containing the upper and lower confidence intervals for the merged replicates.

    Returns
    -------
    rep_dict : dictionnary
        A dictionnary containing the values of interest for the gene of interest.

    """
    rep_dict={}
    rep_dict['spli_data']=df_dict['spliced'][gene_name]
    rep_dict['unspli_data']=df_dict['unspliced'][gene_name]
    rep_dict['gap_bool_spli']=create_gap_dict(bool_dict['spliced'],gene_name)
    rep_dict['gap_bool_unspli']=create_gap_dict(bool_dict['unspliced'],gene_name)
    rep_dict['spli_low_CI']=CI_dict['spliced']['low_CI'][gene_name]
    rep_dict['unspli_low_CI']=CI_dict['unspliced']['low_CI'][gene_name]
    rep_dict['spli_up_CI']=CI_dict['spliced']['up_CI'][gene_name]
    rep_dict['unspli_up_CI']=CI_dict['unspliced']['up_CI'][gene_name]

    return(rep_dict)

#%% Target scan functions

def prep_variances_for_TS(mean_dict,boundary_dict,keys_to_use='All'):
    """
    Function which calculates the variance for both spliced and unspliced values
    of merged replicates. This serves as pre-processing for the target scan (TS)
    analysis downstream.
    
    Function written by Yohan Lefol

    Parameters
    ----------
    mean_dict : dictionnary
        dictionnary containing the mean values of the merged replicates.
    boundary_dict : dictionnary
        dictionnary containing the cell boundaries of G1,S, and G2M.
    keys_to_use : string, optional
        Indication of the divisions done on the means. The variance is calculated
        for these divisions. The default is 'All'.

    Returns
    -------
    gene_dict : dictionnary
        A dictionnary containing the name of the genes belonging to the groups.
    variance_dict : dictionnary
        Dictionnary containing the variances.

    """
    
    variance_dict={}
    gene_dict={}
    phase_length_dict={}
    
    gene_name=next(iter(mean_dict['spliced']))
    
    if keys_to_use != 'All' and keys_to_use != 'Both' and keys_to_use != 'Phases':
        print("The entered 'keys_to_use' parameter was no bueno")
        return None,None
    
    
    if keys_to_use=='Phases' or keys_to_use=='Both':
        if boundary_dict['G2M']==0:#G1 orientation
            phase_length_dict['G2M'] = [0,boundary_dict['G1']]
            phase_length_dict['S'] = [boundary_dict['S'],len(mean_dict['spliced'][gene_name])]
            phase_length_dict['G1'] = [boundary_dict['G1'],boundary_dict['S']]
        else:#G2M orientation
            phase_length_dict['S'] = [0,boundary_dict['G1']]
            phase_length_dict['G2M'] = [boundary_dict['G2M'],len(mean_dict['spliced'][gene_name])]
            phase_length_dict['G1'] = [boundary_dict['G1'],boundary_dict['G2M']]
    if keys_to_use=='All' or keys_to_use=='Both':
        phase_length_dict['All'] = [0,len(mean_dict['spliced'][gene_name])]

    
    for phase in phase_length_dict.keys():
       idx_start=phase_length_dict[phase][0]
       idx_end=phase_length_dict[phase][1]
       variance_dict[phase]={}
       for gene in mean_dict['spliced'].keys():
            vari_unspli = mean_dict['unspliced'][gene][idx_start:idx_end].var()
            vari_spli = mean_dict['spliced'][gene][idx_start:idx_end].var()
            final_vari = vari_spli/vari_unspli
            variance_dict[phase][gene]=final_vari

    #Prepare a dictionnary with just the names of the genes in each key
    for key in variance_dict.keys():
        gene_dict[key]=[x for x in list(variance_dict[key].keys()) if str(x) != 'nan']
        
    return gene_dict,variance_dict


#%% Plotting functions

def plot_raincloud_delay(delay_df,cell_line,plot_name='',save_path='',save_name=''):
    """
    Function which plots four raincloud plots illustrating the different types of
    delays found in unspliced and spliced datasets.
    
    Raincloud plots were made following the tutorial seen here 
    (https://colab.research.google.com/drive/10UObYNGsepQgaCswi6l1cOy0CxQdr3Ki?usp=sharing#scrollTo=clVUB4O0y3Ki)
     
     source article: Raincloud plots: a multi-platform tool for robust data visualization
     
     Function written by Yohan Lefol, adapted from the above link
         

    Parameters
    ----------
    delay_dict : dictionnary
        dictionnary containing the values for the 4 types of delays.
    cell_line : string
        string indicating the cell line used, variable is used for save location purposes.
    plot_name : string, optional
        Title shown on the plot. The default is ''.
    save_path : string, optional
        path to a save location. The default is ''.

    Returns
    -------
    None.

    """
    
    #Log transform the data
    dta1 = my_utils.log10_dta(delay_df,'inc_to_0')
    dta2 = my_utils.log10_dta(delay_df,'inc_to_+1')
    dta3 = my_utils.log10_dta(delay_df,'dec_to_0')
    dta4 = my_utils.log10_dta(delay_df,'dec_to_-1')
    
    

    
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
    labels = ['increase\nto 0', 'increase\nto +1', 'decrease\nto 0', 'decrease\nto-1']
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
    
    
    spearman_and_plot_delays(delay_df,'inc_to_+1','dec_to_0',save_path=plot_path,save_name=save_name+'_active_transcription_correlation.png')
    spearman_and_plot_delays(delay_df,'inc_to_0','dec_to_-1',save_path=plot_path,save_name=save_name+'_no_transcription_correlation.png')


def raincloud_delay_two_files_two_categories(delay_file_1,delay_file_2,cat_1,cat_2,save_name='custom_delay.png'):
    """
    Function which plots two categories of delays from two files. This is to better compare
    two categories of two different origins. For example, delays representing active
    transcription for all genes and for significant genes

    Parameters
    ----------
    delay_file_1 : pandas dataframe
        Contains delay data.
    delay_file_2 : pandas dataframe
        Contains delay data.
    cat_1 : string
        string for a category present in both delay files.
    cat_2 : string
        string for a category present in both delay files..
    save_name : string, optional
        save path and name of the plot. The default is 'custom_delay.png'.

    Returns
    -------
    None.

    """
    #Log transform the data
    dta1 = my_utils.log10_dta(delay_file_1,cat_1)
    dta2 = my_utils.log10_dta(delay_file_1,cat_2)
    dta3 = my_utils.log10_dta(delay_file_2,cat_1)
    dta4 = my_utils.log10_dta(delay_file_2,cat_2)
        
    data_to_plot = [dta1, dta2, dta3, dta4]
    
    #Set the style/background of the plot
    sns.set(style="whitegrid",font_scale=2)
    
    #Create the plot
    f, ax = plt.subplots(figsize=(15, 5))
    ax=pt.RainCloud(data = data_to_plot, palette = ['#e89674','#95a3c3','#e89674','#95a3c3'], bw = 0.3, width_viol = .6, ax = ax, orient = 'v',offset=0.12)
    # set style for the axes
    labels = ['', '', '', '']
    for ax in [ax]:
        ax.xaxis.set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticklabels(labels)
    
    plot_name='left - '+str(len(dta1))+' genes | right - '+str(len(dta3))+' genes'
    plt.title(plot_name)
    plt.ylabel('log10(cell delay)')
    #Save the plot
    plt.savefig(os.path.join(save_name),bbox_inches='tight')
    
def plot_layer_plot(ax,df_dict,df_dict_mean,boundary_dict,gene_name,orientation,return_spli_ax=False):
    """
    Function which plots what has been called the 'layer plot'. This plot shows 
    all the spliced and unspliced value of each cell for a specific gene. Spliced 
    and unspliced values are shown in different colors. The mean curve of the spliced 
    and unspliced cells is also shown: A colored bar at the bottom of the plot 
    indicates in which phase of the cell cycle those cells are.
    
    Function written by Yohan Lefol

    Parameters
    ----------
    ax : matplotlib subplot object
        coorinates within a grid for plotting.
    df_dict : dictionnary
        dictionnary containing the spliced and unspliced values for the targeted gene.
    df_dict_mean : dictionnary
        dictionnary containing the mean spliced and unspliced values for the targeted gene.
    boundary_dict : dictionnary
        dictionnary contianing the cell boundaries for the cell cycle.
    gene_name : string
        The name of the gene to be plotted.
    orientation : string
        Either G1 or G2M to indicate the orientation of the dataset in regards
        to the cell cycle.
    return_spli_ax : boolean, optional
        Boolean indicating if the spliced boolean should be returned. May be usefull
        for plotting purposes

    Returns
    -------
    None.

    """
    #Copies the boundary dict as to not overwrite it
    #Adds color code to the boundary dict
    layer_boundaries=boundary_dict.copy()
    
    colors_dict = {'G1':np.array([52, 127, 184]),
      'S':np.array([37,139,72]),
      'G2M':np.array([223,127,49]),}
    colors_dict = {k:v/256 for k, v in colors_dict.items()}
    
    layer_boundaries['S']=[layer_boundaries['S'],colors_dict['S']]
    layer_boundaries['G2M']=[layer_boundaries['G2M'],colors_dict['G2M']]
    layer_boundaries['G1']=[layer_boundaries['G1'],colors_dict['G1']]
    
    #Creates the x axis (number of cells)
    my_x_axis=np.arange(0,len(df_dict['spliced'][gene_name]))
    
    #Plots the unspliced cells on the left hand y axis
    #Customizes the figure
    
    ax.scatter(my_x_axis, df_dict['unspliced'][gene_name], alpha=0.7, c="#b35806", s=5, label="unspliced",marker='.')
    ax.set_ylim(0, np.max(df_dict['unspliced'][gene_name])*1.02)
    minimal_yticks(0, np.max(df_dict['unspliced'][gene_name])*1.02)
    
    #Creates a separate y axis (ax2)
    ax_2 = ax.twinx()
    #Plots the spliced cells on the secondary y axis
    ax_2.scatter(my_x_axis, df_dict['spliced'][gene_name], alpha=0.7, c="#542788", s=5, label="spliced",marker='.')
    ax_2.set_ylim(0, np.max(df_dict['spliced'][gene_name])*1.02)
    minimal_yticks(0, np.max(df_dict['spliced'][gene_name])*1.02)
    
    #Some plot formatting
    ax.set_ylabel("unspliced",labelpad=-25,fontsize=15)
    ax_2.set_ylabel("spliced",labelpad=-25,fontsize=15)
    plt.xlim(0,np.max(my_x_axis))
    p = np.min(my_x_axis)
    P = np.max(my_x_axis)
    # ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks(np.linspace(p,P,5), [f"{p:.0f}", "","","", f"{P:.0f}"])#,fontsize=15)
    ax.tick_params(axis='x')#,labelsize=15)
    
    #Calculate the spliced and unspliced mean curved (smoothed) values
    # spli_mean_array,unspli_mean_array=smooth_layer_no_vlm(x_axis=my_x_axis,bin_size=100,window_size=200,spliced_array=df_dict['spliced'][gene_name],unspliced_array=df_dict['unspliced'][gene_name],orientation='G1')
    
    spli_mean_array=df_dict_mean['spliced'][gene_name]
    unspli_mean_array=df_dict_mean['unspliced'][gene_name]
    
    bin_order_axis=np.arange(start=0,stop=len(my_x_axis))
    
    #Plot the spliced and unspliced mean curves on their respective axes
    ax_2.plot(bin_order_axis, spli_mean_array, c="#2a1344",linewidth=3,label='mean_spliced')
    ax.plot(bin_order_axis, unspli_mean_array, c="#7d3d04",linewidth=3,label='mean_unspliced')

    #Add ax lines and colored horizontal lines to illustrate the cell cycle phases
    plot_ax_lines_phase_portrait_no_vlm(my_x_axis,orientation,layer_boundaries)
    
    #If the orientation is G2M, reverse the x axis
    if orientation == 'G2M':
        plt.gca().invert_xaxis()
        ax.set_xlabel("order (reversed)",labelpad=-10)#,fontsize=20)
    
    if return_spli_ax==True:
        return(ax_2)

def plot_vels_and_CIs(main_dict,subplot_coordinates,plot_title,boundary_dict,single_rep=False):
    """
    Function which plots the mean values, along with the confidence intervals,
    as well as the boolean values in accordance with the confidence intervals.
    The plot also indicates the different phases of the cell cycle.
    
    Function written by Yohan Lefol

    Parameters
    ----------
    main_dict : dictionnary
        Dictionnary containing all the information (means, CI, boolean) for the gene
        being plotted.
    subplot_coordinates : matplotlib subplot object
        coorinates within a grid for plotting.
    plot_title : string
        The title given to the plot.
    boundary_dict : dictionnary
        Dictionnary containing the boundaries for G1, S, and G2M.
    single_rep : Boolean
        Indicate if the gene is being plotted from the values of a single replicate
        or merged replicates, this affects the presence of confidence intervals
        
    Returns
    -------
    None.

    """
    ax=subplot_coordinates
    #Original spliced color - #542788
    #Original unspliced color - #b35806
    ax.plot(range(len(main_dict['spli_data'])),main_dict['spli_data'],c="#B59AD6",zorder=3)
    ax.plot(range(len(main_dict['unspli_data'])),main_dict['unspli_data'],c="#CE9665",zorder=3)
    
    if single_rep==False:
        ax.plot(range(len(main_dict['spli_low_CI'])), main_dict['spli_low_CI'], c="k",linewidth=0.8,ls='-',zorder=3)
        ax.plot(range(len(main_dict['unspli_low_CI'])), main_dict['unspli_low_CI'], c="k",linewidth=0.8,ls='-',zorder=3)
        ax.plot(range(len(main_dict['spli_up_CI'])), main_dict['spli_up_CI'], c="k",linewidth=0.8,ls='--',zorder=3)
        ax.plot(range(len(main_dict['unspli_up_CI'])), main_dict['unspli_up_CI'], c="k",linewidth=0.8,ls='--',zorder=3)
    
    # min_y_val,max_y_val=0,0
    # if min_y_val>np.min(main_dict['spli_data']):
    #     min_y_val=np.min(main_dict['spli_data'])
    # if max_y_val<np.max(main_dict['spli_data']):
    #     max_y_val=np.max(main_dict['spli_data'])
    
    #Even if single rep is true, the min max will be the same for CI or mean, might
    #As well keep it this way
    min_y_val=np.min(main_dict['spli_low_CI'])
    if min_y_val > np.min(main_dict['unspli_low_CI']):
        min_y_val=np.min(main_dict['unspli_low_CI'])
    
    max_y_val=np.max(main_dict['spli_up_CI'])
    if max_y_val < np.max(main_dict['unspli_up_CI']):
        max_y_val=np.max(main_dict['unspli_up_CI'])
        
    plt.ylim(min_y_val*1.05, max_y_val*1.05)
    plt.xlim(0,len(main_dict['spli_data'])-1)
    p = np.min(0)
    P = np.max(len(main_dict['spli_data'])-1)
    ax.spines['top'].set_visible(False)
    plt.xticks(np.linspace(p,P,5), [f"{p:.0f}", "","","", f"{P:.0f}"])
    
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    p = min([np.min(main_dict['spli_data']),np.min(main_dict['unspli_data'])])
    P = max([np.max(main_dict['spli_data']),np.max(main_dict['unspli_data'])])
    
    plt.yticks([p,P], visible=True, rotation="vertical",va="center")

    

    #This plots the gray boxes for the confidence intervals
    # for gap_type in ['gap_bool_spli','gap_bool_unspli']:
    #     for idx,start_val in enumerate(main_dict[gap_type]['start_false']):
    #         if gap_type=='gap_bool_spli':
    #             y_val=0.015*max_y_val
    #         else:
    #             y_val=-(0.015*max_y_val)
    #         end_val=main_dict[gap_type]['end_false'][idx]+1
    #         gap_distance=abs(start_val-end_val)
    #         x_arr=np.arange(start_val,end_val,1)
    #         y_arr=np.asarray([y_val]*gap_distance)
    #         plt.fill_between(x_arr,y_arr,color='darkgray',lw=2.5,zorder=2, alpha=1)
    
    #Plots the thicker lines indicating velocity significance
    for gap_type in ['gap_bool_spli','gap_bool_unspli']:
        for idx,start_val in enumerate(main_dict[gap_type]['start_true']):
            end_val=main_dict[gap_type]['end_true'][idx]
            if gap_type=='gap_bool_spli':
                ax.plot(range(len(main_dict['spli_data']))[start_val:end_val],main_dict['spli_data'][start_val:end_val],c="#542788",lw=2,zorder=3)
            else:
                ax.plot(range(len(main_dict['unspli_data']))[start_val:end_val],main_dict['unspli_data'][start_val:end_val],c="#b35806",lw=2,zorder=3)
                
    #This plots the phase boundaries
    colors_dict = {'G1':np.array([52, 127, 184]),'S':np.array([37,139,72]),'G2M':np.array([223,127,49]),}
    colors_dict = {k:v/256 for k, v in colors_dict.items()}
    num_cells = len(main_dict['spli_data'])
    
    for key,order in boundary_dict.items():
        plt.axvline(order,c='k',lw=2)
    
    if boundary_dict['G2M']==0:
        plt.hlines(min_y_val*1.05,0,boundary_dict['G1'], colors=colors_dict['G2M'], linestyles='solid',lw=6)
        plt.hlines(min_y_val*1.05,boundary_dict['G1'],boundary_dict['S'], colors=colors_dict['G1'], linestyles='solid',lw=6)
        plt.hlines(min_y_val*1.05,boundary_dict['S'],num_cells, colors=colors_dict['S'], linestyles='solid',lw=6)
    else:
        plt.hlines(min_y_val*1.05,0,boundary_dict['G1'], colors=colors_dict['S'], linestyles='solid',lw=6)
        plt.hlines(min_y_val*1.05,boundary_dict['G1'],boundary_dict['G2M'], colors=colors_dict['G1'], linestyles='solid',lw=6)
        plt.hlines(min_y_val*1.05,boundary_dict['G2M'],num_cells, colors=colors_dict['G2M'], linestyles='solid',lw=6)
        
        plt.gca().invert_xaxis()

    
    plt.hlines(0,0,len(main_dict['spli_data']),colors='black',linestyles='solid',lw=0.5)
    # plt.title(plot_title)


def plot_layer_smooth_vel(gene_name, mean_dict, bool_dict, CI_dict, counts_dict,vlm_dict,vlm_mean_dict,boundary_dict,cell_line,save_path='',single_rep=False):
    """
    A wrapper function which creates a two plot figure:
    It plots the layer plot on the lef thand side and the curve/velocity plot
    on the right hand side. The function exists in a version where a velocyto
    object is required for the plotting, this is an adaptation where the velocyto object
    is no longer needed.
    
    Function written by Yohan Lefol

    Parameters
    ----------
    gene_name : string
        The name of the gene to be plotted.
    mean_dict : dictionnary
        Dictionnary containing the velocity values.
    bool_dict : dictionnary
        Dictionnary containing the boolean values (if a confidence interval
        shows that a velocity is truly positive or negative).
    CI_dict : dictionnary
        Dictionnary containing the confidence interval values.
    counts_dict : dictionnary
        Dictionnary containing the count values (either +1,-1, or 0) for the velocity.
    vlm_dict : dictionnary
        Dictionnary containing the spliced and unpsliced values (non-velocity).
    vlm_mean_dict : dictionnary
        Dictionnary containing the spliced and unpsliced values (non-velocity) means.
    boundary_dict : dictionnary
        Dictionnary indicating the cell cycle boundaries in terms of cell number for each phase..
    cell_line : string
        String indicating the cell line being used.
    save_path : string, optional
        Path to which the figure will be saved. The default is ''.
    single_rep : boolean, optional
        Indicates if the plot is for a single replicate or not. If it is for a single replicate,
        confidence intervals are removed. The default is False.

    Returns
    -------
    None.

    """
    if boundary_dict['G2M']==0:
        orientation='G1'
    else:
        orientation='G2M'
    
    plt.figure(None, (9.00,3.92), dpi=600)

    gs = plt.GridSpec(1,2)
    
    #Plot the layer plot on the left hand side
    ax = plt.subplot(gs[0]) 
    plot_layer_plot(ax,vlm_dict,vlm_mean_dict,boundary_dict,gene_name,orientation)
    
    #Plot the velocity curve plot on the right hand side
    ax = plt.subplot(gs[1])
    main_dict=subset_the_dicts_merged_CI(gene_name,mean_dict,bool_dict,CI_dict)
    plot_vels_and_CIs(main_dict,ax,plot_title=gene_name,boundary_dict=boundary_dict,single_rep=single_rep)
    
    #Build the elements for the legend
    from matplotlib.lines import Line2D
    handles,labels=ax.get_legend_handles_labels()
    spli_leg=Line2D([0], [0],color='#542788', linewidth=1, linestyle='solid',label='spliced')
    unspli_leg=Line2D([0], [0],color='#b35806', linewidth=1, linestyle='solid',label='unspliced')   
    lines_up=Line2D([0], [0],color='black', linewidth=0.5, linestyle='--',label='upper_CI')
    lines_down=Line2D([0], [0],color='black', linewidth=0.5, linestyle='solid',label='lower_CI')
    # gray_spli=Line2D([0], [0],color='darkgray', linewidth=4, linestyle='solid',label='above 0 -- spliced')
    # gray_unspli=Line2D([0], [0],color='darkgray', linewidth=4, linestyle='solid',label='below 0 -- unspliced')
    
    
    #Create the legend elements for the phase boundaries
    colors_dict = {'G1':np.array([52, 127, 184]),'S':np.array([37,139,72]),'G2M':np.array([223,127,49]),}
    colors_dict = {k:v/256 for k, v in colors_dict.items()}
    
    G2M_bar=Line2D([0], [0],color=colors_dict['G2M'], linewidth=4, linestyle='solid',label='G2/M cells')
    G1_bar=Line2D([0], [0],color=colors_dict['G1'], linewidth=4, linestyle='solid',label='G1 cells')
    S_bar=Line2D([0], [0],color=colors_dict['S'], linewidth=4, linestyle='solid',label='S cells')
    
    #Add necessary legend elements to the legend
    handles.append(spli_leg) 
    handles.append(unspli_leg) 
    # handles.append(gray_spli) 
    # handles.append(gray_unspli) 
    handles.append(G2M_bar) 
    handles.append(G1_bar) 
    handles.append(S_bar) 
    
    if single_rep==False:    
        handles.append(lines_up) 
        handles.append(lines_down) 
    
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    #Use gene name as plot title
    plt.suptitle(gene_name,ha='center',fontsize=20, y=1.05)
    
    #SAve the plot using gene name and cell line as location unless specified otherwise
    if save_path=='':
        plot_path="all_figures/"+cell_line+"/merged_replicates/layer_vel_figs"
    else:
        plot_path=save_path
    name=gene_name.replace('.','_')#If a period is in the gene name, replace with underscore
    if not os.path.exists(plot_path):
        os.makedirs(plot_path, exist_ok=True)
    plt.savefig(os.path.join(plot_path,name),bbox_inches='tight')
    plt.clf()
    plt.close("all")
    # plt.show()
    gc.collect()
    

def plot_counts(counts_dict,gene_name,subplot_coordinates,boundary_dict,reverse):
    """
    Function which plots the count values of a gene
    
    Function written by Yohan Lefol

    Parameters
    ----------
    counts_dict : dictionnary
        Dictionnary containing the count values.
    gene_name : string
        The name of the gene being plotted.
    subplot_coordinates : matplotlib subplot object
        coorinates within a grid for plotting.
    boundary_dict : dictionnary
        A dictionnary containing the cell cycle boundaries in terms of cell number
    reverse : Boolean
        A Boolean indicating if the x axis should be reversed or not

    Returns
    -------
    None.

    """
    spli_counts=counts_dict['spliced']
    unspli_counts=counts_dict['unspliced']
    ax=subplot_coordinates
    ax.plot(range(len(spli_counts[gene_name])),spli_counts[gene_name],c="#542788")
    ax.plot(range(len(unspli_counts[gene_name])),unspli_counts[gene_name],c="#b35806")
    plt.ylim(-1.25, 1.25)
    plt.xlim(0,len(spli_counts[gene_name]))
    plt.hlines(0,0,len(spli_counts[gene_name]),colors='black',linestyles='solid',lw=0.5)
    # plt.title(gene_name)
    
    p = np.min(0)
    P = np.max(len(spli_counts[gene_name])-1)
    plt.xticks(np.linspace(p,P,5), [f"{p:.0f}", "","","", f"{P:.0f}"])
    plt.yticks([-1,1], visible=False, rotation="horizontal")
    
    
    #This plots the phase boundaries
    colors_dict = {'G1':np.array([52, 127, 184]),'S':np.array([37,139,72]),'G2M':np.array([223,127,49]),}
    colors_dict = {k:v/256 for k, v in colors_dict.items()}
    num_cells = len(counts_dict['spliced'])
    
    if boundary_dict['G2M']==0:
        plt.hlines(-1.25,0,boundary_dict['G1'], colors=colors_dict['G2M'], linestyles='solid',lw=6)
        plt.hlines(-1.25,boundary_dict['G1'],boundary_dict['S'], colors=colors_dict['G1'], linestyles='solid',lw=6)
        plt.hlines(-1.25,boundary_dict['S'],num_cells, colors=colors_dict['S'], linestyles='solid',lw=6)
    else:
        plt.hlines(-1.25,0,boundary_dict['G1'], colors=colors_dict['S'], linestyles='solid',lw=6)
        plt.hlines(-1.25,boundary_dict['G1'],boundary_dict['G2M'], colors=colors_dict['G1'], linestyles='solid',lw=6)
        plt.hlines(-1.25,boundary_dict['G2M'],num_cells, colors=colors_dict['G2M'], linestyles='solid',lw=6)
    
    if reverse==True:
        plt.gca().invert_xaxis()
    # ax.legend(loc="best")
    # plot_path="my_figures/count_figs"
    # name=gene_name
    # if not os.path.exists(plot_path):
    #     os.makedirs(plot_path, exist_ok=True)
    # plt.savefig(os.path.join(plot_path,name),bbox_inches='tight')
    # plt.clf()
    # plt.close("all")
    # plt.show()
    # gc.collect()



def plot_curve_count(gene_name, mean_dict, bool_dict, CI_dict, counts_dict,boundary_dict,cell_line,save_path=''):
    """
    Function that plots and saves a grid figure which illustrates the spliced and
    unspliced values of the merged replicates, along with the confidence interval lines,
    the cell cycle phase boundaries. The seconda plot of the grid is a count
    plot which shows the count values of the merged replicates.
    
    Function written by Yohan Lefol

    Parameters
    ----------
    gene_name : string
        The name of the gene to be plotted.
    mean_dict : dictionnary
        dictionnary containing the merged replicate values.
    bool_dict : dictionnary
        dictionnary containing the boolean values of the merged replicates,
        it indicates if a cells value is within the confidence interval or not.
    CI_dict : dictionnary
        The upper and lower confidence intervals for the merged replicates.
    counts_dict : dictionnary
        The count values for the merged replicates.
    boundary_dict : dictionnary
        The cell boundaries for G1, S, and G2M.
    cell_line : string
        The name of the cell line, value is used to determine the save location
        of the plot.
    save_path : string
        The path to which the plot will be saved

    Returns
    -------
    None.

    """
    plt.figure(None, (9.00,3.92), dpi=600)

    gs = plt.GridSpec(1,2)
    
    ax = plt.subplot(gs[0]) 
    main_dict=subset_the_dicts_merged_CI(gene_name,mean_dict,bool_dict,CI_dict)
    plot_vels_and_CIs(main_dict,ax,plot_title=gene_name,boundary_dict=boundary_dict)
      
    ax = plt.subplot(gs[1])
    if boundary_dict['G2M']==0:#Indicates G1 orientation
        plot_counts(counts_dict,gene_name,ax,boundary_dict,reverse=False)
    else:#Indicates G2M orientation
        plot_counts(counts_dict,gene_name,ax,boundary_dict,reverse=True)
    from matplotlib.lines import Line2D

    handles,labels=ax.get_legend_handles_labels()
    spli_leg=Line2D([0], [0],color='#542788', linewidth=1, linestyle='solid',label='spliced')
    unspli_leg=Line2D([0], [0],color='#b35806', linewidth=1, linestyle='solid',label='unspliced')   
    lines_up=Line2D([0], [0],color='black', linewidth=0.5, linestyle='--',label='upper_CI')
    lines_down=Line2D([0], [0],color='black', linewidth=0.5, linestyle='solid',label='lower_CI')
    # gray_spli=Line2D([0], [0],color='darkgray', linewidth=4, linestyle='solid',label='above 0 -- spliced')
    # gray_unspli=Line2D([0], [0],color='darkgray', linewidth=4, linestyle='solid',label='below 0 -- unspliced')
    
    
    #This plots the phase boundaries
    colors_dict = {'G1':np.array([52, 127, 184]),'S':np.array([37,139,72]),'G2M':np.array([223,127,49]),}
    colors_dict = {k:v/256 for k, v in colors_dict.items()}
    
    G2M_bar=Line2D([0], [0],color=colors_dict['G2M'], linewidth=4, linestyle='solid',label='G2/M cells')
    G1_bar=Line2D([0], [0],color=colors_dict['G1'], linewidth=4, linestyle='solid',label='G1 cells')
    S_bar=Line2D([0], [0],color=colors_dict['S'], linewidth=4, linestyle='solid',label='S cells')
    
    handles.append(spli_leg) 
    handles.append(unspli_leg) 

    # handles.append(gray_spli) 
    # handles.append(gray_unspli) 
    handles.append(G2M_bar) 
    handles.append(G1_bar) 
    handles.append(S_bar) 
    
    handles.append(lines_up) 
    handles.append(lines_down) 
    
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.suptitle(gene_name,ha='center',fontsize=20, y=1.05)
    if save_path=='':
        plot_path="all_figures/"+cell_line+"/merged_replicates/curve_count_figs"
    else:
        plot_path=save_path
    name=gene_name.replace('.','_')#If a period is in the gene name, replace with underscore
    if not os.path.exists(plot_path):
        os.makedirs(plot_path, exist_ok=True)
    plt.savefig(os.path.join(plot_path,name),bbox_inches='tight')
    plt.clf()
    plt.close("all")
    plt.show()
    gc.collect()


def create_REAC_summary_plots(value_dict,boundary_dict,layer='spliced',second_layer=None,orientation='G1',plot_path='REAC_folder/'):
    """
    Function which plots the genes of a REACTOME pathway as well as the mean gene
    expression for the pathway.
    
    

    Parameters
    ----------
    value_dict : dictionnary
        Dictionnary containing the expression values of each gene for spliced and unspliced.
    boundary_dict : dictionnary
        The cell boundaries for G1, S, and G2M.
    layer : string, optional
        The layer to be plotted, either spliced or unspliced. The default is 'spliced'.
    second_layer : string, optional
        The second layer to be added to the plot if needed, it will appear below the first.
        The default is None
    orientation : string, optional
        Either G1 or G2M to indicate the orientation of the cell cycle. The default is 'G1'.
    plot_path : string, optional
        The save location of the plots being created. The default is 'REAC_folder/'.

    Returns
    -------
    None.

    """
    
    #Create color dict
    col_dict={}
    col_dict['spliced']=["#9577b8","#2a1344"]
    col_dict['unspliced']=["#bd8553","#7d3d04"]

    #Copies the boundary dict as to not overwrite it
    #Adds color code to the boundary dict
    layer_boundaries=boundary_dict.copy()
    
    colors_dict = {'G1':np.array([52, 127, 184]),
      'S':np.array([37,139,72]),
      'G2M':np.array([223,127,49]),}
    colors_dict = {k:v/256 for k, v in colors_dict.items()}
    
    layer_boundaries['S']=[layer_boundaries['S'],colors_dict['S']]
    layer_boundaries['G2M']=[layer_boundaries['G2M'],colors_dict['G2M']]
    layer_boundaries['G1']=[layer_boundaries['G1'],colors_dict['G1']]
    
    #Calculate the scaled genes and mean line
    for REAC in value_dict.keys():
        
        #Find total genes in the pathway
        REAC_file=pd.read_csv('data_files/REAC_pathways/'+REAC+'.txt',header=None)
        total_genes=len(REAC_file[0])
        
        #Remove underscore from names
        REAC_name=str.replace(REAC, '_', ' ')
        my_y=np.atleast_2d(value_dict[REAC][layer])
        my_y=my_y/my_y.sum(axis=1, keepdims=True)
        mean_line=np.mean(my_y,axis=0)
        my_x_axis=np.arange(0,len(mean_line))
        
        #Do the plot
        ax = plt.subplot(111)
        ax.plot(my_x_axis, my_y.T, c=col_dict[layer][0],alpha=0.4)
        ax.plot(my_x_axis, mean_line, c=col_dict[layer][1],linewidth=3)
        
        if second_layer != None:
            second_val=np.atleast_2d(value_dict[REAC][second_layer])
            second_val=second_val/second_val.sum(axis=1, keepdims=True)
            second_mean=np.mean(second_val,axis=0)
            second_mean=second_mean-np.max(second_val)*1.05
            second_val=second_val.T-np.max(second_val)*1.05
            ax.plot(my_x_axis, second_val, c=col_dict[second_layer][0],alpha=0.4)
            ax.plot(my_x_axis, second_mean, c=col_dict[second_layer][1],linewidth=3)
            ax.set_ylim(-np.max(my_y)*1.02, np.max(my_y)*1.02)
            
        else:
            ax.set_ylim(0, np.max(my_y)*1.02)
        #Add ax lines and colored horizontal lines to illustrate the cell cycle phases
        plot_ax_lines_phase_portrait_no_vlm(my_x_axis,orientation,layer_boundaries)
        
        ax.axes.yaxis.set_ticks([])
        # my_func.minimal_yticks(0, np.max(my_y)*1.02)
        
        ax.set_xlim(0,np.max(my_x_axis))
        p = np.min(my_x_axis)
        P = np.max(my_x_axis)
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        plt.xticks(np.linspace(p,P,5), [f"{p:.0f}", "","","", f"{P:.0f}"])#,fontsize=15)
        ax.tick_params(axis='x')#,labelsize=15)
        
        ax.set_ylabel("scaled expression value",fontsize=15)
        ax.set_xlabel("order",fontsize=15)
        plt.title(REAC_name+' ('+str(len(my_y.T[0]))+'/'+str(total_genes)+')',fontsize=15)
        
        #If the orientation is G2M, reverse the x axis
        if orientation == 'G2M':
            plt.gca().invert_xaxis()
            plt.set_xlabel("order (reversed)")#,fontsize=20)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path, exist_ok=True)
        plt.savefig(os.path.join(plot_path,REAC),dpi=300)
        plt.clf()
        plt.close("all")
        # plt.show()
        gc.collect()


def plot_phase_exp_raincloud(exp_matrix,target_genes,phase_boundaries,plot_title='',save_name='',scale=False,log10=False):
    """
    Function which plots a rainfall plot split into three cell cycle phases. The function
    takes in the primary expression matrix, a list of gene subsets (to filter the matrix),
    as well as the phase boundaries, plot title, and the name of the file for saving the plot.

    Parameters
    ----------
    exp_matrix : pandas dataframe
        A dataframe containing the expression results with cells as rows and genes as columns
    target_genes : pandas series
        A series of genes to subset the expression matrix with
    phase_boundaries : dictionnary
        The cell boundaries for G1, S, and G2M.
    plot_name : string, optional
        Title shown on the plot. The default is ''.
    save_path : string, optional
        path to a save location. The default is ''.
    scale : boolean, optional
        boolean indicating if data should be scaled (following log transformation
        if applicable).
    log10 : boolean, optional
        boolean indicating if data should be log10 transformed

    Returns
    -------
    None.

    """
    #Start by setting up the data, one df per phase
    data_to_plot=[]
    for phase in list(['G1','S','G2M']): #Setting custom order for better representation
        #Identify start and end of boundary
        start=phase_boundaries[phase]
        if phase=='G1':
            end=phase_boundaries['S']-1
        elif phase=='S':
            end=len(exp_matrix.index)
        else:
            end=phase_boundaries['G1']-1
        
        sub_matrix=exp_matrix[target_genes]
        sub_matrix=sub_matrix.iloc[start:end]
        sub_matrix=sub_matrix.mean(axis=0)
        if log10==True:
            sub_dta= my_utils.log10_dta(sub_matrix,None)
        else:
            sub_dta=sub_matrix.to_numpy()
        if scale==True:
            sub_dta=sub_dta/sub_dta.sum(axis=0, keepdims=True)
        data_to_plot.append(sub_dta)

    #Set the style/background of the plot
    sns.set(style="whitegrid",font_scale=2)

    #Create the plot
    f, ax = plt.subplots(figsize=(15, 5))
    ax=pt.RainCloud(data = data_to_plot, palette = 'Set2', bw = 0.3, width_viol = .6, ax = ax, orient = 'v',offset=0.12)
    # set style for the axes
    labels = ['G1', 'S', 'G2/M']
    for ax in [ax]:
        ax.xaxis.set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticklabels(labels)
        
    plt.title(plot_title)
    plt.ylabel('log10(phase expression)')

    plt.savefig(save_name,bbox_inches='tight')
    plt.clf()
    plt.close("all")
    # plt.show()
    gc.collect()

    #Resets the 'theme' for the plots as to not interfere with downstream plots
    mpl.rc_file_defaults()


def raincloud_for_cc_genes(main_df,target_phase):
    """
    Function which plots raincloud plots. This function expects a very specific file
    generated by the 'create_cell_line_replicate_cc_df' function. The target phase 
    is specified and this function will plot a raincloud plot of all cell line and 
    replicates for the genes associated to the specified function. 
    The function will save the plot to the main directory.

    Parameters
    ----------
    main_df : pandas dataframe
        pandas dataframe containing gene names, associated cell cycle phase, and spliced
        expression results for each cell line + replicate to be plotted.
    target_phase : string
        Either G1, S, or G2/M.

    Returns
    -------
    None.

    """
    #Start by setting up the data, one df per phase
    data_to_plot=[]
    exp_matrix=main_df[main_df.phase==target_phase]
    exp_matrix=exp_matrix.drop(columns=['Gene','phase'])
    for col in exp_matrix.columns:
        data_to_plot.append(exp_matrix[col].dropna())
    
    #Set the style/background of the plot
    sns.set(style="whitegrid",font_scale=2)
    # Create an array with the colors you want to use
    colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']
    # Set your custom color palette
    sns.set_palette(sns.color_palette(colors))
    
    #Create the plot
    f, ax = plt.subplots(figsize=(12,6)  , dpi=100)
    ax=pt.RainCloud(data = data_to_plot, palette = colors, ax = ax, orient = 'v',offset=0.12)
    # set style for the axes
    labels = list(exp_matrix.columns)
    for ax in [ax]:
        ax.xaxis.set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticklabels(labels)
    plt.xticks(rotation=45)
    plt.title('cell cycle genes associated to '+target_phase)
    plt.ylabel('spliced expression values')
    
    if target_phase=='G2/M':
        target_phase='G2M'
    plt.savefig('raincloud_cc_genes_'+target_phase+'.png',bbox_inches='tight')
    plt.clf()
    plt.close("all")
    # plt.show()
    gc.collect()
    
    #Resets the 'theme' for the plots as to not interfere with downstream plots
    mpl.rc_file_defaults()


def plot_spliced_velocity_expression_zero_point(mean_dict,CI_dict,bool_dict,vlm_mean_dict,boundary_dict,gene_name,plot_name='',custom_alpha_zero_point=0.1):
    """
    Function which plots the spliced velocity as well as spliced expression data. The plot also illustrates the 
    location of peak expression (up and down) as well as where velocity is uncertain (zero points).
    
    This plot also shows the order of the cells/data points based on their assigned cell cycle phase.
    
    Plot is then saved.

    Parameters
    ----------
    mean_dict : dictionnary
        Dictionnary containing the spliced velocity data.
    CI_dict : dictionnary
        Dictionnary containing the confidence interval data for spliced velocity.
    bool_dict : dictionnary
        Dictionnary containing booleans which indicate where the spliced velocity is uncertain.
    vlm_mean_dict : dictionnary
        Dictionnary containing the smoothed spliced and unspliced values for the expression data.
    boundary_dict : dictionnary
        Dictionnary containing the cell cycle phase boundaries.
    gene_name : string
        String of the gene name that is to be plotted.
    plot_name : string, optional
        plot path and name. The default is ''.
    custom_alpha_zero_point : float, optional
        The alpha value given to the zero point range. Depending on the x axis, various
        different alphas may be beneficial due to line overlapping. The default is 0.1.

    Returns
    -------
    None.

    """
    #Figure set up
    plt.figure(None, (10.00,3.92), dpi=600)
    gs = plt.GridSpec(1,2)
    ax = plt.subplot(gs[0]) 
    
    #Create the dictionnary for the velocity curves (vel, CI, and boolean)
    main_dict=subset_the_dicts_merged_CI(gene_name,mean_dict,bool_dict,CI_dict)
    
    #### create/plot velocity curve
    #Plot the velocity curve and the confidence interval provided it is a merged replicate
    #The lines plotted here have a fainter color
    ax.plot(range(len(main_dict['spli_data'])),main_dict['spli_data'],c='#542788',zorder=3)
    ax.plot(range(len(main_dict['spli_low_CI'])), main_dict['spli_low_CI'], c="k",linewidth=0.8,ls='-',zorder=3)
    ax.plot(range(len(main_dict['spli_up_CI'])), main_dict['spli_up_CI'], c="k",linewidth=0.8,ls='--',zorder=3)

    #### velocity y axis formatting
    min_y_val=np.min(main_dict['spli_low_CI'])
    max_y_val=np.max(main_dict['spli_up_CI'])
    #Plots the horizontal zero line on the y axis of velocity
    plt.hlines(0,0,len(main_dict['spli_data']),colors='black',linestyles='solid',lw=0.5)
    #setup y axis for velocity - requires scientific notatoion
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    p = np.min(main_dict['spli_data'])
    P = np.max(main_dict['spli_data'])
    plt.yticks([p,0,P], visible=True, rotation="vertical",va="center")
    ax.set_ylabel("spliced velocity",labelpad=0,fontsize=15)
    
    #Calculate y axis spacing
    y_axis_spacing=min_y_val-(min_y_val*1.05)
    plt.ylim(min_y_val-y_axis_spacing, max_y_val+y_axis_spacing)
    
    plt.draw() #Need to draw to fetch the labels
    #Get labels and overwrite scientific notation format for the 0
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels[1]=0
    ax.set_yticklabels(labels)
    
    #### x axis set-up
    my_x_axis=np.arange(0,len(main_dict['spli_data']))
    plt.xlim(0,len(my_x_axis))
    p = np.min(0)
    P = np.max(len(my_x_axis))
    ax.spines['top'].set_visible(False)
    plt.xticks(np.linspace(p,P,5), [f"{p:.0f}", "","","", f"{P:.0f}"])
    #Retrieve tick labels and make last tick of x axis aligned to the right
    ticklabels = ax.get_xticklabels()
    ticklabels[-1].set_ha("right")
    
    #Get spliced values
    spli_array=vlm_mean_dict['spliced'][gene_name]

    #### Plot expression on y axis
    #Creates a separate y axis (ax_2)
    ax_2 = ax.twinx()
    
    #Plot spliced data
    ax_2.plot(my_x_axis, spli_array, c="#542788",linewidth=2,alpha=0.5)
    #y axis formatting
    min_y_val=np.min(spli_array)
    max_y_val=np.max(spli_array)
    y_axis_spacing=min_y_val-min_y_val*1.05
    plt.ylim((min_y_val+y_axis_spacing), max_y_val*1.05)
    #setup y axis for velocity - requires scientific notatoion
    # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    plt.yticks([min_y_val,max_y_val], visible=True, rotation="vertical",va="center")
    ax_2.set_ylabel("spliced expression",labelpad=-10,fontsize=15)
    
    
    #### Zero point ranges
    for idx,val in enumerate(main_dict['gap_bool_spli']['start_false']):
        start_range=val
        end_range=main_dict['gap_bool_spli']['end_false'][idx]
        for i in range(start_range,end_range):
            con = ConnectionPatch(xyA=(i,vlm_mean_dict['spliced'][gene_name][i]), xyB=(i,0), coordsA="data", coordsB="data", axesA=ax_2, axesB=ax, color="red",alpha=custom_alpha_zero_point)
            ax.add_artist(con)
    
    #### expression peaks (min and max)
    location_max=np.where(vlm_mean_dict['spliced'][gene_name]==np.max(vlm_mean_dict['spliced'][gene_name]))[0][0]
    con_max = ConnectionPatch(xyA=(location_max,vlm_mean_dict['spliced'][gene_name][location_max]), xyB=(location_max,0), coordsA="data", coordsB="data", axesA=ax_2, axesB=ax, color="darkred",alpha=1,lw=1)
    ax.add_artist(con_max)
    
    location_min=np.where(vlm_mean_dict['spliced'][gene_name]==np.min(vlm_mean_dict['spliced'][gene_name]))[0][0]
    con_min = ConnectionPatch(xyA=(location_min,vlm_mean_dict['spliced'][gene_name][location_min]), xyB=(location_min,0), coordsA="data", coordsB="data", axesA=ax_2, axesB=ax, color="darkred",alpha=1,lw=1)
    ax.add_artist(con_min)
    
    
    #### Plot cell cycle color guide
    #Set up new dict
    layer_boundaries=boundary_dict.copy()
    #Set-up color codes
    colors_dict = {'G1':np.array([52, 127, 184]),'S':np.array([37,139,72]),'G2M':np.array([223,127,49]),}
    colors_dict = {k:v/256 for k, v in colors_dict.items()}
    #Create/adjust new dict
    layer_boundaries['S']=[layer_boundaries['S'],colors_dict['S']]
    layer_boundaries['G2M']=[layer_boundaries['G2M'],colors_dict['G2M']]
    layer_boundaries['G1']=[layer_boundaries['G1'],colors_dict['G1']]
    #Identify orientation
    if boundary_dict['G2M']==0:
        orientation='G1'
    else:
        orientation='G2M'
    #create the colored horizontal lines for cell cycle phase
    plot_ax_lines_phase_portrait_no_vlm(my_x_axis,orientation,layer_boundaries,v_lines=False,y_axis_loc=(min_y_val+y_axis_spacing))
    
    
    #### Create plot legend
    from matplotlib.lines import Line2D
    spli_leg=Line2D([0], [0],color='#542788', linewidth=2, linestyle='solid',label='spliced velocity')
    unspli_leg=Line2D([0], [0],color='#542788', linewidth=2,alpha=0.5, linestyle='solid',label='spliced expression')   
    lines_up=Line2D([0], [0],color='black', linewidth=1, linestyle='--',label='upper_CI')
    lines_down=Line2D([0], [0],color='black', linewidth=1, linestyle='solid',label='lower_CI')
    
    G2M_bar=Line2D([0], [0],color=colors_dict['G2M'], linewidth=4, linestyle='solid',label='G2/M cells')
    G1_bar=Line2D([0], [0],color=colors_dict['G1'], linewidth=4, linestyle='solid',label='G1 cells')
    S_bar=Line2D([0], [0],color=colors_dict['S'], linewidth=4, linestyle='solid',label='S cells')
    
    exp_peaks=Line2D([0], [0],color='darkred', linewidth=2, linestyle='solid',label='expression peaks')
    vel_zero_bar=Line2D([0], [0],color='red',alpha=0.5, linewidth=4, linestyle='solid',label='velocity 0 point')

    #Add necessary legend elements to the legend
    handles=[spli_leg,unspli_leg,G2M_bar,G1_bar,S_bar,lines_up,lines_down,exp_peaks,vel_zero_bar]
    plt.legend(handles=handles, bbox_to_anchor=(1.10, 1), loc='upper left')
    
    
    #Adjust the layout, save, and show
    plt.title(gene_name)
    plt.tight_layout()
    if plot_name=='':
        plot_name=gene_name+'.png'
    plt.savefig(plot_name,bbox_inches='tight')
    # plt.show()


def plot_velocity_expression_zero_point(vel_dict,vel_CI_dict,bool_dict,vlm_dict,vlm_mean_dict,boundary_dict,gene_name,plot_name='',custom_alpha_zero_point=0.1):
    """
    Function which plots a the layer plot and velocity plot (with the first being above the second).
    The function then adds a range (transparent red) showing where the spliced velocity is considered
    to be zero. It also plots a single line in darkred showing the spliced expression peak on 
    the smoothed data.
    Both of these types of lines extend from one plot to the other as the intent is to show
    the location of these two events in the other plot.
    
    The function saves the plot.

    Parameters
    ----------
    vel_dict : dictionnary
        Dictionnary containing the spliced velocity data.
    vel_CI_dict : dictionnary
        Dictionnary containing the confidence interval data for spliced velocity.
    bool_dict : dictionnary
        Dictionnary containing booleans which indicate where the spliced velocity is uncertain.
    vlm_dict : dictionnary
        Dictionnary containing the spliced and unspliced values for the expression data.
    vlm_mean_dict : dictionnary
        Dictionnary containing the smoothed spliced and unspliced values for the expression data.
    boundary_dict : dictionnary
        Dictionnary containing the cell cycle phase boundaries.
    gene_name : string
        String of the gene name that is to be plotted.
    plot_name : string, optional
        plot path and name. The default is ''.
    custom_alpha_zero_point : float, optional
        The alpha value given to the zero point range. Depending on the x axis, various
        different alphas may be beneficial due to line overlapping. The default is 0.1.

    Returns
    -------
    None.

    """
    #### Determine orientation
    if boundary_dict['G2M']==0:
        orientation='G1'
    else:
        orientation='G2M'
    
    #### Dimension set-up
    plt.figure(None, (7.50,7.00), dpi=600)
    gs = plt.GridSpec(2,1) #2 rows 1 column
    
    #### Layer plot (top location)
    ax1 = plt.subplot(gs[0]) 
    layer_spli_ax=plot_layer_plot(ax1,vlm_dict,vlm_mean_dict,boundary_dict,gene_name,orientation,return_spli_ax=True)
    
    #### velocity plot with additional y label
    ax2 = plt.subplot(gs[1])
    main_dict=subset_the_dicts_merged_CI(gene_name,vel_dict,bool_dict,vel_CI_dict)
    plot_vels_and_CIs(main_dict,ax2,plot_title=gene_name,boundary_dict=boundary_dict,single_rep=False)
    ax2.set_ylabel("velocity",labelpad=-10,fontsize=15)
    
    #### Zero point ranges
    for idx,val in enumerate(main_dict['gap_bool_spli']['start_false']):
        start_range=val
        end_range=main_dict['gap_bool_spli']['end_false'][idx]
        for i in range(start_range,end_range):
            con = ConnectionPatch(xyA=(i,vlm_mean_dict['spliced'][gene_name][i]), xyB=(i,0), coordsA="data", coordsB="data", axesA=layer_spli_ax, axesB=ax2, color="red",alpha=custom_alpha_zero_point)
            ax2.add_artist(con)
    
    #### expression peaks (min and max)
    location_max=np.where(vlm_mean_dict['spliced'][gene_name]==np.max(vlm_mean_dict['spliced'][gene_name]))[0][0]
    con_max = ConnectionPatch(xyA=(location_max,vlm_mean_dict['spliced'][gene_name][location_max]), xyB=(location_max,0), coordsA="data", coordsB="data", axesA=layer_spli_ax, axesB=ax2, color="darkred",alpha=1,lw=1)
    ax2.add_artist(con_max)
    
    location_min=np.where(vlm_mean_dict['spliced'][gene_name]==np.min(vlm_mean_dict['spliced'][gene_name]))[0][0]
    con_min = ConnectionPatch(xyA=(location_min,vlm_mean_dict['spliced'][gene_name][location_min]), xyB=(location_min,0), coordsA="data", coordsB="data", axesA=layer_spli_ax, axesB=ax2, color="darkred",alpha=1,lw=1)
    ax2.add_artist(con_min)
    
    #### Create legend elements and add handles
    from matplotlib.lines import Line2D
    #Build the elements for the legend
    handles,labels=ax2.get_legend_handles_labels()
    spli_leg=Line2D([0], [0],color='#542788', linewidth=1, linestyle='solid',label='spliced')
    unspli_leg=Line2D([0], [0],color='#b35806', linewidth=1, linestyle='solid',label='unspliced')   
    lines_up=Line2D([0], [0],color='black', linewidth=0.5, linestyle='--',label='upper_CI')
    lines_down=Line2D([0], [0],color='black', linewidth=0.5, linestyle='solid',label='lower_CI')
    
    exp_peaks=Line2D([0], [0],color='darkred', linewidth=2, linestyle='solid',label='expression peaks')
    vel_zero_bar=Line2D([0], [0],color='red',alpha=0.5, linewidth=4, linestyle='solid',label='velocity 0 point')
    
    #Create the legend elements for the phase boundaries
    colors_dict = {'G1':np.array([52, 127, 184]),'S':np.array([37,139,72]),'G2M':np.array([223,127,49]),}
    colors_dict = {k:v/256 for k, v in colors_dict.items()}
    
    G2M_bar=Line2D([0], [0],color=colors_dict['G2M'], linewidth=4, linestyle='solid',label='G2/M cells')
    G1_bar=Line2D([0], [0],color=colors_dict['G1'], linewidth=4, linestyle='solid',label='G1 cells')
    S_bar=Line2D([0], [0],color=colors_dict['S'], linewidth=4, linestyle='solid',label='S cells')
    
    #Add necessary legend elements to the legend
    handles.append(spli_leg) 
    handles.append(unspli_leg) 
    handles.append(G2M_bar) 
    handles.append(G1_bar) 
    handles.append(S_bar)    
    handles.append(lines_up) 
    handles.append(lines_down) 
    handles.append(exp_peaks) 
    handles.append(vel_zero_bar) 
    
    #### plot legend and title, correct layout, save plot
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    #Use gene name as plot title
    plt.suptitle(gene_name,ha='center',fontsize=20)
    
    if plot_name=='':
        plot_name=gene_name+'.png'
    plt.savefig(plot_name,bbox_inches='tight')




#%% Wrapper functions for analysis

def wrapper_plot_single_rep_genes(cell_line,replicates,target_rep,gene_list,save_path=''):
    """
    Wrapper function which creates layer/velocity plots as well as velocity/count plots
    for the inputted genes using only the data from the inputted replicate.
    This function enables users to create results using single replicates instead of
    merged replicates
    
    Function written by Yohan Lefol

    Parameters
    ----------
    cell_line : string
        String stating the cell line being used.
    replicates : list
        list of strings for the replicates of the cell line.
    target_rep : string
        string indicating the target replicate for plotting.
    gene_list : list
        list of gene names to be plotted.
    save_path : string, optional
        path of the save location for the plots. Each type of plot will be 
        automatically stored in a appropriately labeled folder. The default is ''.

    Returns
    -------
    None.

    """
    layers=['spliced','unspliced']
    mean_dict,CI_dict,bool_dict,count_dict,boundary_dict=my_utils.get_CI_data (cell_line, layers, target_rep)
    df_dict_exp=my_utils.get_vlm_values(cell_line, layers,target_rep)
    df_dict_exp_mean=my_utils.get_vlm_values(cell_line, layers,target_rep,get_mean=True)
    exp_dict={}
    exp_mean_dict={}
    for layer in layers:
        exp_dict[layer]=df_dict_exp[layer]  
        exp_mean_dict[layer]=df_dict_exp_mean[layer]  
    
    if save_path=='':
        save_path='all_figures/'+cell_line+'/single_replicate_analysis/'+target_rep+'/'
    
    for gene in gene_list:
        plot_layer_smooth_vel(gene, mean_dict, bool_dict, CI_dict, count_dict,exp_dict,exp_mean_dict,boundary_dict,cell_line,single_rep=True,save_path=save_path+'layer_curve')
        plot_curve_count(gene, mean_dict, bool_dict, CI_dict, count_dict,boundary_dict,cell_line,save_path=save_path+'curve_count')

                
#%% statistical analysis functions

def find_phase_association(gene_df,mean_dict,CI_dict,boundary_dict,vlm_dict,layer='spliced',CI='low_CI'):
    """
    Function which identifies the 'start' and 'peak' phase of the layer/CI
    
    The layer indicates the data used and the confidence interval indicates if 
    the start and peak will be searched for within the negative velocities (up_CI)
    or positive velocities (low_CI).
    
    When a low_CI goes above 0, it is considered a positive velocity and thus it symbolises
    the 'start' phase. While the peak in this instance will be the layers highest
    value and the phase of the cell cycle where that value is located.
    It finds the peak for velocity and expression

    Parameters
    ----------
    gene_df : pandas dataframe
        pandas dataframe containing the t statistic results.
    mean_dict : dictionnary
        dictionnary containing the merged replicate values.
    CI_dict : dictionnary
        The upper and lower confidence intervals for the merged replicates.
    boundary_dict : dictionnary
        The cell boundaries for G1, S, and G2M.
    vlm_dict : dictionnary
        contains expression values, used to identify expression phase peak
    layer : string, optional
        Either spliced or unspliced. The default is 'spliced'.
    CI : string, optional
        Either low_CI or up_CI. The default is 'low_CI'.

    Returns
    -------
    phase_association : Dictionnary
        A dictionnary of three lists (start velocity, preak velocity, and peak expression) 
        with the found phases for the gene list provided.

    """
    #Set-up dictionnaries
    phase_association={}
    phase_association['peak_vel']=[]
    phase_association['peak_exp']=[]
    phase_association['start_vel']=[]
    found_idx={}
    
    #Iterate over all genes
    for gene in list(gene_df.index):
        if gene_df[gene_df.index==gene].t[0]==0.0:
            phase_association['peak_vel'].append('NA')
            phase_association['peak_exp'].append('NA')
            phase_association['start_vel'].append('NA')
        else:
            #If low CI we want to search for when the layer has it's max value and when the CI is above 0
            if CI=='low_CI':
                found_idx['peak_vel']=np.where(mean_dict[layer][gene]==max(mean_dict[layer][gene]))[0]
                found_idx['peak_exp']=np.where(vlm_dict[layer][gene]==max(vlm_dict[layer][gene]))[0]
                found_idx['start_vel']=np.where(CI_dict[layer][CI][gene]>0)[0]
            #If up CI we search for the layers minimum value and when the CI is below 0
            elif CI=='up_CI':
                found_idx['peak_vel']=np.where(mean_dict[layer][gene]==min(mean_dict[layer][gene]))[0]
                found_idx['peak_exp']=np.where(vlm_dict[layer][gene]==min(vlm_dict[layer][gene]))[0]
                found_idx['start_vel']=np.where(CI_dict[layer][CI][gene]<0)[0]
            for key in list(found_idx.keys()):
                if len(found_idx[key])==0:
                    found_phase='NA'
                else:
                    val=found_idx[key][0]
                    if val>= boundary_dict['G2M'] and val < boundary_dict['G1']:
                        found_phase='G2M'
                    elif val>=boundary_dict['G1'] and val < boundary_dict['S']:
                        found_phase='G1'
                    elif val>=boundary_dict['S']:
                        found_phase='S'
                #Return phases found for each gene in the same order that the genes
                #were provided
                phase_association[key].append(found_phase)
    return phase_association

def create_t_test_rank_method(gene_df,iterations,replicates,mean_dict,CI_dict,boundary_dict,vlm_dict,layer='spliced',CI='low_CI'):
    """
    Function which takes in a dataframe of genes and computes the t-test statistic for 
    the desired layer and CI. It then finds the phase association for each gene
    based on that same layer and CI. Phase associations show in which phase a gene
    starts being significant (either positive or negative velocity based on specified
    CI) as well as the phase where the layer is at it's peak value (highest or lowest 
    based on specified CI). It finds the peak value for velocity and expression values.
    
    The p_value is computed with a two tailed hypothethis assumption.

    Parameters
    ----------
    gene_df : pandas dataframe
        dataframe with desired genes and layer CI to be submitted to t-test statistic.
    iterations : int
        The number of iterations performed.
    replicates : list
        Replicates used.
    mean_dict : dictionnary
        dictionnary containing the merged replicate values.
    CI_dict : dictionnary
        The upper and lower confidence intervals for the merged replicates.
    boundary_dict : dictionnary
        The cell boundaries for G1, S, and G2M.
    vlm_dict : dictionnary
        contains expression values, used to identify expression phase peak
    layer : string, optional
        Either spliced or unspliced. The default is 'spliced'.
    CI : string, optional
        Either low_CI or up_CI. The default is 'low_CI'.

    Returns
    -------
    t_test_df : pandas dataframe.
        A pandas dataframe with the t-test results, p_value and adjsuted pvalue

    """
    #Imports
    from rpy2.robjects.packages import importr
    from rpy2.robjects.vectors import FloatVector
    from rpy2.robjects.vectors import BoolVector
    stats = importr('stats')
    
    #Param set-up
    iterations_done=iterations*len(replicates)
    deg_of_freedom=iterations_done-1
    layer_CI=layer+'_'+CI
    
    
    # Remove all zeros (non-ranked genes)
    gene_input_list=gene_df[layer_CI][gene_df[layer_CI]!=0]
    
    #Calculate p_value
    my_pvals=list(stats.pt(FloatVector(gene_input_list),df=deg_of_freedom,lower_tail=BoolVector([False])))
    my_pvals = [p * 2 for p in my_pvals] #Multiply by two due to two tailed hypothesis

    #Calculate p_adjsuted values
    num_genes=len(my_pvals)
    p_adjust = list(stats.p_adjust(FloatVector(my_pvals),n=num_genes, method = 'fdr'))

    #Adjust length for non-ranked genes
    non_ranked=len(gene_df[layer_CI])-num_genes
    non_ranked_lst=['NA']*non_ranked

    #Create base dataframe with t value, p_value and padjusted value with associated
    # gene names
    dataframe_dict={}
    dataframe_dict['t']=gene_df[layer_CI]
    dataframe_dict['pvalue']=my_pvals+non_ranked_lst
    dataframe_dict['padjusted']=p_adjust + non_ranked_lst
    t_test_df = pd.DataFrame(dataframe_dict)
    t_test_df.index=gene_df['gene_name']

    # Find phases where layer CI begins and maxes out
    found_phases=find_phase_association(t_test_df,mean_dict,CI_dict,boundary_dict,vlm_dict,layer,CI)
    
    #Add phases to df and return
    t_test_df['phase_peak_vel']=found_phases['peak_vel']
    t_test_df['phase_peak_exp']=found_phases['peak_exp']
    t_test_df['phase_start_vel']=found_phases['start_vel']
    
    
    return(t_test_df)

                
def spearman_and_plot_delays(delay_file,delay_cat_1,delay_cat_2,save_path,save_name):
    """
    Function which calculates the spearman correlation between two delay categories
    within the same cell line. The coorrelation and pvalue are illustrated on
    a seaborn regplot of those two delay categories.

    Parameters
    ----------
    delay_file : pandas dataframe
        A pandas dataframe containing all the delay values for a cell line.
    delay_cat_1 : string
        the type of delay that will be compared with delay_cat_2.
    delay_cat_2 : string
        the type of delay that will be compared with delay_cat_1.
    save_path : string
        path to which the plot will be saved
    save_name : string
        the name of the plot for saving

    Returns
    -------
    None.

    """
    inc_arr=np.asarray(delay_file[delay_cat_1])
    dec_arr=np.asarray(delay_file[delay_cat_2])
    spear_results=stats.spearmanr(inc_arr,dec_arr)
    
    y_max=max(dec_arr)-(max(dec_arr)*0.3)
    x_min=min(inc_arr)-(min(inc_arr)*0.05)
    pval='{:0.3e}'.format(spear_results[1])
    corr_val='%.3f'%spear_results[0]
    
    import seaborn as sns
    sns.set_theme(color_codes=True)
    ax = sns.regplot(x=delay_cat_1, y=delay_cat_2, data=delay_file,truncate=False,scatter_kws={"color": "#4d73b0"}, line_kws={"color": "black"})
    plt.text(x_min, y_max, 'correlation: '+str(corr_val)+'\npvalue: '+str(pval),
             bbox=dict(facecolor='none', edgecolor='black'))
    plt.xlabel(delay_cat_1)
    plt.ylabel(delay_cat_2)
    plt.savefig(save_path+'/'+save_name)
    plt.clf()
    plt.close("all")
    # plt.show()
    gc.collect()
    
    #Resets the 'theme' for the plots as to not interfere with downstream plots
    mpl.rc_file_defaults()


def spearman_delay_categories(delay_dataframe):
    """
    Function which calculates the spearman correlation for all possible iterations
    of the delay categories. The function returns the correlation and pvalue of the
    results in the form of a pandas dataframe.

    Parameters
    ----------
    delay_dataframe : pandas dataframe
        Contains the values and categories used for the spearman correlation.

    Returns
    -------
    df : pandas dataframe
        Contains the correlation and pvalue for all possible iterations of delay
        categories.

    """
    delay_cats=list(delay_dataframe.columns)
    delay_cats.remove('gene_name')
    found_iterations=list(itertools.combinations(delay_cats, 2))
    
    dataframe_list=[]
    for found_iter in found_iterations:
        arr_1=np.asarray(delay_dataframe[found_iter[0]])
        arr_2=np.asarray(delay_dataframe[found_iter[1]])
        spear_results=stats.spearmanr(arr_1,arr_2)
        
        row_name=found_iter[0]+' vs '+found_iter[1]
        res_list=[row_name,spear_results[0],spear_results[1]]
        dataframe_list.append(res_list)
    
    df = pd.DataFrame(dataframe_list, columns = ['Comparison', 'Correlation','Pvalue'])
    return df


def chi_square_cell_lines(cell_line_1,folder_1,cell_line_2,folder_2):
    """
    Function which performs a chi square test between the significant genes
    found in two cell lines of choice.
    
    The function will retrieve the set score results for the two cell lines then 
    remove any genes which are unique for either cell line. A contingency table is built 
    and the chi square results are returned.

    Parameters
    ----------
    cell_line_1 : string
        String definining one of the two cell lines to be compared.
    folder_1 : string
        String definining the folder (replicate) to use for the first cell line
    cell_line_2 : string
        String definining one of the two cell lines to be compared.
    folder_2 : string
        String definining the folder (replicate) to use for the second cell line

    Returns
    -------
    None.

    """
    ranks_line_1=pd.read_csv('data_files/data_results/rank/'+cell_line_1+'/'+folder_1+'_t_test_results.csv')
    ranks_line_2=pd.read_csv('data_files/data_results/rank/'+cell_line_2+'/'+folder_2+'_t_test_results.csv')
    
    # #Find common genes
    # common_genes=[]
    # for gene in ranks_line_1.gene_name.values:
    #     if gene in ranks_line_2.gene_name.values:
    #         common_genes.append(gene)
            
    # #Filter files to contain only common genes across cell lines
    # ranks_line_1=ranks_line_1[ranks_line_1.gene_name.isin(common_genes)]
    # ranks_line_2=ranks_line_2[ranks_line_2.gene_name.isin(common_genes)]
    
    sig_genes_line_1=list(ranks_line_1.gene_name[ranks_line_1.t>0])
    non_sig_genes_line_1=list(ranks_line_1.gene_name[ranks_line_1.t==0])
    
    
    sig_genes_line_2=list(ranks_line_2.gene_name[ranks_line_2.t>0])
    non_sig_genes_line_2=list(ranks_line_2.gene_name[ranks_line_2.t==0])

    yes_1_yes_2=0
    yes_1_no_2=0
    yes_2_no_1=0
    for gene in sig_genes_line_1:
        if gene in sig_genes_line_2:
            yes_1_yes_2+=1
    
    yes_1_no_2=len(sig_genes_line_1)-yes_1_yes_2
    yes_2_no_1=len(sig_genes_line_2)-yes_1_yes_2
    
    no_1_no_2=0
    all_non_sigs=list(set(non_sig_genes_line_1+non_sig_genes_line_2))
    for gene in all_non_sigs:
        if gene not in sig_genes_line_1:
            if gene not in sig_genes_line_2:
                no_1_no_2+=1
    
    
    #Build contingency table
    cont_table=np.array([[yes_1_yes_2,yes_2_no_1],[yes_1_no_2,no_1_no_2]])
    res=stats.chi2_contingency(cont_table)
    
    return res




def chi_square_cell_line_phases(cell_line_1,folder_1,cell_line_2,folder_2,phase_check,comparison_cat):
    """
    A function which performs a chi square test for the phase overlap
    of genes found in two different cell lines. 
    The function first removes any genes unique to either cell line.
    It then takes in the comparison category and the phase to check.
    
    The function will build the contingency table accordingle and return the chi square results.

    Parameters
    ----------
    cell_line_1 : string
        String definining one of the two cell lines to be compared.
    folder_1 : string
        String definining the folder (replicate) to use for the first cell line
    cell_line_2 : string
        String definining one of the two cell lines to be compared.
    folder_2 : string
        String definining the folder (replicate) to use for the second cell line
    phase_check : string
        The phase to be used to check overlap (G1,S,G2M).
    comparison_cat : string
        A string indication which phase association to use. Phase at peak velocity (phase_peak_vel),
        phase at peak expression (phase_peak_exp), or phase at start of active transcription (phase_start_vel)

    Returns
    -------
    None.

    """
    ranks_line_1=pd.read_csv('data_files/data_results/rank/'+cell_line_1+'/'+folder_1+'_t_test_results.csv')
    ranks_line_2=pd.read_csv('data_files/data_results/rank/'+cell_line_2+'/'+folder_2+'_t_test_results.csv')

    # #Find common genes
    # common_genes=[]
    # for gene in ranks_line_1.gene_name.values:
    #     if gene in ranks_line_2.gene_name.values:
    #         common_genes.append(gene)
    
    # #Filter files to contain only common genes across cell lines
    # ranks_line_1=ranks_line_1[ranks_line_1.gene_name.isin(common_genes)]
    # ranks_line_2=ranks_line_2[ranks_line_2.gene_name.isin(common_genes)]
    
    genes_to_check=list(ranks_line_1.gene_name)
    
    yes_1_yes_2=0
    yes_2_no_1=0
    yes_1_no_2=0
    no_1_no_2=0
    for gene in genes_to_check:
        phase_line_1=ranks_line_1[comparison_cat][ranks_line_1.gene_name==gene].values
        if len(phase_line_1)>0:
            phase_line_1=phase_line_1[0]
        else:
            phase_line_1=None
            
        phase_line_2=ranks_line_2[comparison_cat][ranks_line_2.gene_name==gene].values
        if len(phase_line_2)>0:
            phase_line_2=phase_line_2[0]
        else:
            phase_line_2=None
        
    
        if phase_line_1 == phase_check and phase_line_2 == phase_check:
            yes_1_yes_2+=1
        elif phase_line_1 == phase_check and phase_line_2 != phase_check:
            yes_1_no_2+=1
        elif phase_line_1 != phase_check and phase_line_2 == phase_check:
            yes_2_no_1+=1
        else:
            no_1_no_2+=1
    
    #Build contingency table
    cont_table=np.array([[yes_1_yes_2,yes_2_no_1],[yes_1_no_2,no_1_no_2]])
    res=stats.chi2_contingency(cont_table)

    return res


def DEG_comparison_thresh_var(DEG_file_name,thresh_var_dict,ref_key,thresh_name,intersect):
    """
    Function which performs two comparisons with a DEG list.
    It performs one with all provided cell lines, and a second with the provided intersect

    Parameters
    ----------
    DEG_file_name : string
        location and name of the DEG file.
    thresh_var_dict : dictionnary
        Dictionnary containing the log10 variance values for each cell line.
    ref_key : string
        String indicating the cell line to use as a reference in the intersect analysis.
    thresh_name : string
        string showing the threshold used.
    intersect : list
        list of gene names showing the intersect of the cell lines provided.

    Returns
    -------
    None.

    """
    universe=[]
    set_sig_dict={}
    for cell_line in thresh_var_dict.keys():
        universe=universe+list(thresh_var_dict[cell_line].gene_name)
        set_sig_dict[cell_line]=set(list(thresh_var_dict[cell_line].gene_name[thresh_var_dict[cell_line].padjusted<0.01]))
    #Filter based on threshold here
    # cell_line_dict={'HaCat':thresh_subset_HaCat,'293t':thresh_subset_293t,'jurkat':thresh_subset_jurkat}
    
    #Function will also filter for significant genes
    res_dict=create_TCGA_comparison_stat_results(thresh_var_dict,DEG_file_name,universe)
    #Convert to dataframe and save
    res_df=pd.DataFrame.from_dict(res_dict)
    res_df.to_csv('analysis_results/TCGA_cell_line_comp_thresh_log10_'+thresh_name+'.csv')
    
    #Perform intersect analysis
    ref_intersect=thresh_var_dict[ref_key][thresh_var_dict[ref_key].gene_name.isin(intersect)]
    
    intersect_cell_dict={'intersect':ref_intersect}
    res_dict=create_TCGA_comparison_stat_results(intersect_cell_dict,DEG_file_name,universe)
    #Convert to dataframe and save
    res_df=pd.DataFrame.from_dict(res_dict)
    res_df.to_csv('analysis_results/TCGA_cell_line_comp_intersect_log10_'+thresh_name+'.csv')

def create_TCGA_comparison_stat_results(cell_line_gene_dict,DEG_file_name,universe):
    """
    Function intended to be used with a DEGc file from a TCGA analysis. However this
    function can be used for any comparisons. The function will compare each cell line
    with a given file name (which must have an ID column). The function will perform
    a chi-square test to measure the overlap between each cell line (split per cell 
    cycle phase) and the given gene list. Genes for the cell lines will be split
    based on the cell cycle phase for which they have their highest expression point.
    A universe must also be provided for the function. The universe is (normally) 
    comprised of all unique genes found in each cell line.
    
    The function will return a dictionnary with the observed, expected, and 
    associated pvalue.

    Parameters
    ----------
    cell_line_gene_dict : Dictionnary
        Dictionnary containing the three cell lines results. The file is expected to 
        be the t_test_results.csv file per cell line. These are expected to be in 
        pandas dataframe format
    DEG_file_name : string
        path to the file containing the gene list to compare with.
    universe : pandas dataframe
        Dataframe containing the universe to be used in the chi-square analysis.

    Returns
    -------
    A dictionnary containing the observed, expected, and pvalue for each comparison.
    Comparisons are each cell cycle phase at each cell line

    """
    gene_lst_1=pd.read_csv(DEG_file_name)
    gene_lst_1=list(gene_lst_1.ID)
    gene_lst_1=set(gene_lst_1)
    
    universe=set(universe)
    universe=list(universe)
    
    comp_res_dict={'Cell_line':[],'Phase':[],'Observed':[],'Expected':[],'pvalue':[]}
    for cell_line in list(cell_line_gene_dict.keys()):
        for phase in ['G1','S','G2M']:
            #Create subsets
            gene_df=cell_line_gene_dict[cell_line]
            gene_df=gene_df[gene_df.padjusted<0.01]
            gene_lst_2=list(gene_df.gene_name[gene_df.phase_peak_exp==phase])
    
            #Set up variables
            yes_1_yes_2=0
            yes_1_no_2=0
            yes_2_no_1=0
            no_1_no_2=0
            
            #Prepare for contingency table
            for gene in universe:
                if gene in gene_lst_1 and gene in gene_lst_2:
                    yes_1_yes_2+=1
                elif gene in gene_lst_1 and gene not in gene_lst_2:
                    yes_1_no_2+=1
                elif gene not in gene_lst_1 and gene in gene_lst_2:
                    yes_2_no_1+=1
                elif gene not in gene_lst_1 and gene not in gene_lst_2:
                    no_1_no_2+=1
                
            #Build contingency table and run stat
            cont_table=np.array([[yes_1_yes_2,yes_2_no_1],[yes_1_no_2,no_1_no_2]])
            print('##########')
            print(cell_line)
            print(phase)
            print(cont_table)
            comp_res_dict['Cell_line'].append(cell_line)
            comp_res_dict['Phase'].append(phase)
            if cont_table[0][0]==0 and cont_table[0][1]==0:
                #Cannot do comparison
                comp_res_dict['Observed'].append('No genes')
                comp_res_dict['Expected'].append('No genes')
                comp_res_dict['pvalue'].append('NA')
            else:
                res=stats.chi2_contingency(cont_table)
                
                #Build list of results - observed expected pvalue
                expected_yes_yes=res[3][0][0]
                observed_yes_yes=yes_1_yes_2
                pvalue=res[1]
                #Add results to dictionnary
                comp_res_dict['Observed'].append(observed_yes_yes)
                comp_res_dict['Expected'].append(expected_yes_yes)
                comp_res_dict['pvalue'].append(pvalue)
            
    return(comp_res_dict)


