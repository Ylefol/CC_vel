#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 15:01:41 2022

@author: yohanl
"""
import os
import numpy as np


def create_dict_from_iters(gene_dict,path,replicate):
    """
    Function which takes in iterations of velocity or vlm calculations and creates
    a dictionnary with that data. The resulting dictionnary starts by associating 
    a primary key, which is the replicate which has been iterated over. It then loads
    the n number of replicates for each gene.
    The function also takes in the same dictionnary it returns as it may load several
    replicates.

    Parameters
    ----------
    gene_dict : dictionnary
        A dictionnary, either empty or already containing replicate data loaded 
        by this function.
    path : string
        String indicating the directory path to the iterated files that need merging.
    replicate : string
        String defining the replicate. Will be used as the key to the data.

    Returns
    -------
    gene_dict : dictionnary
        A dictionnary containing the loaded iterations of the replicate provided
        to this function.

    """
    list_of_files=os.listdir(path)
    
    #Iterate over all files to get common genes and number of cells (indexes)
    #This handles things on a per replicate basis. Genes and cells between replicates
    #Are balanced later
    col_list=[]
    row_list=[]
    for file in list_of_files:
        temp_df=np.load(path+"/"+file, allow_pickle=True)
        temp_df=temp_df.reset_index()
        temp_df.drop('index', inplace=True, axis=1)
        # print(temp_df)
        if col_list==[]:
            col_list=list(temp_df.columns)
        else:
            col_list=list(set(col_list) & set(list(temp_df.columns)))
        if row_list==[]:
            row_list=list(temp_df.index)
        else:
            row_list=list(set(row_list) & set(list(temp_df.index)))
    
    
    gene_dict[replicate]={}
    
    #Set up the numpy 2D arrays for result storage
    for gene in col_list:
        gene_dict[replicate][gene]=np.zeros(shape=(len(row_list),len(list_of_files)))
    
    #Iterate over files and store results in respective 2D arrays
    for idx,file in enumerate(list_of_files):
        my_df=np.load(path+"/"+file, allow_pickle=True)
        my_df=my_df.reset_index()
        my_df.drop('index', inplace=True, axis=1)
        for gene in gene_dict[replicate].keys():
            if gene in my_df.columns and gene !='Unnamed: 0':
                # print(my_df[gene])
                gene_dict[replicate][gene][:,idx]=my_df[gene].values
    
    return gene_dict



def balance_gene_dictionnary(gene_dict,target_gene):
    """
    Function which takes in a dictionnary of replicates and balances them. 
    Balancing implies that both dictionnaries have the same genes (sub keys)
    and the same number of cells for each gene.
    In regards to balancing cells, the smalles replicate is identified, and the other 
    replicates have cells removed at even intervals to match the number of cells
    of the smallest replicate.
    

    Parameters
    ----------
    gene_dict : dictionnary
        A dictionnary, either empty or already containing replicate data loaded 
        by this function.
    target_gene : string
        Any gene present in all replicates, this is used to identify the number
        of cells in each replicate.

    Returns
    -------
    gene_dict : dictionnary
        The balanced dictionnary of replicates.
    smallest_reps : list
        A list of string(s) indicating which replicate was the smallest.

    """
    #Cell number will be the same for all genes of a replicate, so we pick a single gene
    #to compare the replicates
    smallest_reps=[]
    for rep in gene_dict.keys():
        if smallest_reps==[]:
            smallest_reps.append(rep)
        elif len(gene_dict[smallest_reps[0]][target_gene])==len(gene_dict[rep][target_gene]):
            smallest_reps.append(rep)
        elif len(gene_dict[smallest_reps[0]][target_gene])>len(gene_dict[rep][target_gene]):
            smallest_reps=[rep]
    
    #Adjust cell quantity in replicates that are not the smallest
    #Adjustments removes evenly spaced cells
    cell_num_target=len(gene_dict[smallest_reps[0]][target_gene])
    for rep in gene_dict.keys():
        if rep in smallest_reps:
            continue
        for gene in gene_dict[rep].keys():
            idx=np.round(np.linspace(0, len(gene_dict[rep][gene]) -1, cell_num_target)).astype(int)
            gene_dict[rep][gene]=gene_dict[rep][gene][idx]
            
    return gene_dict,smallest_reps



def merge_replicates(path,replicates,layer,do_vlm=False):
    """
    A wrapper function which takes in iteration data, calls the functions to create 
    a dictionnary from the iteration data for each replicate, balance this dictionnary
    as to ensure that all replicates have the same genes and the same number of cells.
    It then collapses the replicates into a single set of data per gene.
    
    Parameters
    ----------
    path : string
        A string indicating the directory path to the iteration data.
    replicates : list
        A list of strings for the replicates to be analyzed.
    layer : string
        Either 'unspliced' or 'spliced' to indicate which layer is being worked on.
    do_vlm : Boolean, optional
        Indicates if the data to be merged is velocity data or vlm (expression)
        data. The default is False.

    Returns
    -------
    final_gene_dict : dictionnary
        A dictionnary with the balanced and collapsed gene data.
    small_reps : list
        A list containing the smallest replicate(s).

    """
    #Set up result dictionnaries
    merge_rep_gene_dict={}
    for replicate in replicates:
        if do_vlm==False:
            full_path=path+'/'+replicate+'/Iterations/'+layer
        else:
            full_path=path+'/'+replicate+'/vlm_vals_iters/'+layer
        
        #Creates a dictionnary of genes for the replicate being iterated over
        merge_rep_gene_dict=create_dict_from_iters(gene_dict=merge_rep_gene_dict,path=full_path,replicate=replicate)
    
    #If only one replicate, the downstream steps are not important
    if len(replicates)==1:
        return merge_rep_gene_dict[replicates[0]],replicates

    #Identifty genes common to all replicates
    gene_list=[]
    for rep in merge_rep_gene_dict.keys():
        if gene_list==[]:
            gene_list=list(merge_rep_gene_dict[rep].keys())
        else:
            gene_list=list(set(gene_list) & set(list(merge_rep_gene_dict[rep].keys())))
    
    

    
    #Exclude any genes not found in every replicate
    for rep in merge_rep_gene_dict.keys():
        unwanted =  set(merge_rep_gene_dict[rep]) - set(gene_list)
        for unwanted_key in unwanted: del merge_rep_gene_dict[rep][unwanted_key]
    
    
    
    
    #Identify smallest dataset, adjust cell number of other datsets to smallest
    merge_rep_gene_dict,small_reps=balance_gene_dictionnary(gene_dict=merge_rep_gene_dict,target_gene=gene_list[0])


    #Create a new dictionnary which concatenantes the different iterations within
    #The replicates. To save on RAM, previous dictionnary is deleted
    final_gene_dict={}
    for idx,rep in enumerate(replicates):
        #Reached end
        if idx==len(replicates)-1:
            break
        #Concatenate current replicate with following replicate in the list
        for gene in gene_list:
            final_gene_dict[gene]=np.concatenate((merge_rep_gene_dict[rep][gene],merge_rep_gene_dict[replicates[idx+1]][gene]),axis=1)
    
    return final_gene_dict,small_reps

def calculate_dta_mean_and_confidence_intervals(gene_dict,replicates,do_CI,num_iter,z_val):
    """
    Function which takes in a dictionnary of gene data created from the merge_replicates()
    function. This dictionnary has gene as keys and each gene contains x number of data points
    per cell representing the number of iterations and replicates used to create the data.
    
    For each gene, the mean value of all the iterations is calculated as well as the confidence
    interval for the iterations performed.
    
    Each gene has n number of cells, with the value of each cell having been calculated x number
    of times. Each gene will, after this function, have a confidence interval for each cell as well
    as a mean expression for each cell.

    Parameters
    ----------
    gene_dict : dictionnary
        A dictionnary with the balanced and collapsed gene data.
    replicates: list
        A list of strings indicating the replicates used.
    do_CI : boolean
        Indicates if the confidence interval is to be calculated, if not, only
        the mean expression is calculated.
    num_iter : int
        The number of iterations performed for each replicate.
    z_val : float
        The z-value for the calculation of the confidence interval. Assigning the 
        z-value enables the customization of the degree of confidence. For example,
        a z-value of 1.645 will be a confidence of 90%.

    Returns
    -------
    gene_dict : dictionnary
        A dictionnary with the mean expression of each cell for each gene.
    up_CI_dict : dictionnary
        A dictionnary with the upper confidence interval of each cell for each gene.
    low_CI_dict : dictionnary
        A dictionnary with the lower confidence interval of each cell for each gene.
    std_dict : dictionnary
        A dictionnary containing the standard deviation for the iterations of each cell

    """
    #Calculate the variability between the iterations for each cell per gene
    # my_var_dict={}
    up_CI_dict={}
    low_CI_dict={}
    
    std_dict={}
    # for gene in gene_dict.keys():
    #     my_var_dict[gene]=np.var(gene_dict[gene],axis=1)
    
    #The number of iterations is the number of iterations * the number of replicates
    num_iter=num_iter*len(replicates)
    
    #Perform calculations for each gene
    for gene in gene_dict.keys():
        if do_CI:
            std_dict[gene]=gene_dict[gene].std(axis=1)
            CI = (std_dict[gene]/np.sqrt(num_iter)) * z_val
            gene_dict[gene]=gene_dict[gene].mean(axis=1)
            up_CI_dict[gene]=gene_dict[gene]+CI
            low_CI_dict[gene]=gene_dict[gene]-CI
        else:
            gene_dict[gene]=gene_dict[gene].mean(axis=1)

    if do_CI:
        return gene_dict, up_CI_dict, low_CI_dict, std_dict
    else:
        return gene_dict



def create_bool_dictionnary(gene_dict,up_CIs,low_CIs):
    """
    A dictionnary which determines when data is validated by it's confidence intervals.
    For each datapoint (cell) the function checks the confidence interval. If the data
    point is a positive value, the function checks if the lower confidence interval
    is above the 0 point, if it is, a value of True is associated to that cell.
    If the value is negative, the function checks if the upper confidence interval 
    is below 0.

    Parameters
    ----------
    gene_dict : dictionnary
        A dictionnary of genes containing one expression value per cell.
    up_CIs : dictionnary
        Dictionnary containing the upper confidence interval for each gene and each cell
        found in the gene_dict.
    low_CIs : dictionnary
        Dictionnary containing the lower confidence interval for each gene and each cell
        found in the gene_dict.

    Returns
    -------
    bool_dict : dictionnary
        Dictionnary containing a boolean value for each gene at each cell based on if the
        confidence intervals validate the data point.

    """
    
    gene_keys=list(low_CIs.keys()) #Take gene list from which ever CI
    
    CI_dict={}
    for gene in gene_keys:
        CI_dict[gene]={}
        CI_dict[gene]['lower_CI']=low_CIs[gene]
        CI_dict[gene]['upper_CI']=up_CIs[gene]
    bool_dict={}
    for gene in gene_dict.keys():
        bool_dict[gene]=[]
        for idx,mean_val in enumerate(gene_dict[gene]):
            if CI_dict[gene]['lower_CI'][idx]>0:
                bool_dict[gene].append(True)
            elif CI_dict[gene]['upper_CI'][idx]<0:
                bool_dict[gene].append(True)
            else:
                bool_dict[gene].append(False)
    return bool_dict