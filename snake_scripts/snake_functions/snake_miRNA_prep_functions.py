#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 12:43:55 2021

@author: yohanl
"""

import pandas as pd
import numpy as np
import math
import re

def rpm_normalization(df_path):
    """
    Perform rpm normalization and subsequent average calculation on dataframe.

    Parameters
    ----------
    df_path : string
        path to the dataframe that will be rpm normalized.

    Returns
    -------
    my_means : pandas.core.series.Series
        The average rpm of each miRNA in the dataframe.

    """

    my_matrix=pd.read_csv(df_path,index_col=0)
    my_means=my_matrix
    for col in my_matrix.columns:
        scaling_factor=sum(my_matrix[col])
        for idx,val in enumerate(my_matrix[col]):
            my_means[col][idx]=val*1000000/scaling_factor
    

    my_means=my_means.mean(axis=1)
    return my_means

def categorize_findings(TS_path,rpm_df,min_miRNA_thresh=None,max_miRNA_thresh=None):
    """
    Categorizes the list of miRNAs based on the results of the TargetScan file
    Sorted based on if it was found, the ones not found are split into dedicated
    categories for further sorting.
    The miRNA list can be filtered according to an rpm

    Parameters
    ----------
    TS_path : string
        file path to the TargetScan prediction file
    rpm_df : pandas.core.series.Series
        The average rpm of each miRNA in the dataframe.
    min_miRNA_thresh : int, optional
        minimum value for which miRNA rpms are filtered. Only miRNA with a value equal
        or superior to the threshold are kept. The default is None.
    max_miRNA_thresh : int, optional
        maximum value for which miRNA rpms are filtered. Only miRNA with a value equal
        or below to the threshold are kept. The default is None.

    Returns
    -------
    main_dict : dictionnary
        A dictionnary containing all miRNAs of the list in their respective category.

    """
    main_dict={}
    main_dict['found']=[]
    main_dict['not_found_all']=[]
    main_dict['not_found_accounted_for']=[]
    main_dict['not_found_not_p']=[]
    main_dict['not_found_single_p']=[]
    main_dict['not_found_both_p']=[]
    
    TS_file=pd.read_csv(TS_path,sep='\t')
    TS_file=TS_file[TS_file['Gene Tax ID']==9606]
    
    #Filter for RPM desired
    miRNA_list=[]
    for idx,val in rpm_df.iteritems():
        if max_miRNA_thresh==None:
            if val>=min_miRNA_thresh:
                miRNA_list.append(idx)
        else:
            if val>=min_miRNA_thresh and val<=max_miRNA_thresh:
                miRNA_list.append(idx)
    
    
    for i in miRNA_list:
        if i in TS_file.miRNA.unique():
            main_dict['found'].append(i)
        else:
            main_dict['not_found_all'].append(i)
    found_remove_p=[] 
    for i in main_dict['found']:
        if i.endswith('p'):
            found_remove_p.append(i[:-3]) 
    
    not_found_remove_p=[]
    for i in main_dict['not_found_all']:
        if isinstance(i,float)==True:
            if math.isnan(i)==True: #Sometimes a nan sneaks through
                continue
        if i.endswith('p'):
            if i[:-3] not in found_remove_p:
                not_found_remove_p.append(i[:-3])
                main_dict['not_found_single_p'].append(i)
        else:
            main_dict['not_found_not_p'].append(i)
        
    from collections import Counter
    for k,v in Counter(not_found_remove_p).items():
        if v>1:
            main_dict['not_found_both_p'].append(k+'-3p')
            main_dict['not_found_both_p'].append(k+'-5p')
            main_dict['not_found_single_p'].remove(k+'-3p')
            main_dict['not_found_single_p'].remove(k+'-5p')
    
    
    for i in main_dict['not_found_single_p']:
        if i[-3:]=='-3p':
            if i[:-3]+'-5p' not in TS_file.miRNA.unique():
                continue
            else:
                main_dict['not_found_single_p'].remove(i)
        else:
            if i[:-3]+'-3p' not in TS_file.miRNA.unique():
                continue
            else:
                main_dict['not_found_single_p'].remove(i)
    
    
    for i in main_dict['not_found_all']:
        if i not in main_dict['not_found_not_p'] and i not in main_dict['not_found_single_p'] and i not in main_dict['not_found_both_p']:
            main_dict['not_found_accounted_for'].append(i)
    
    return main_dict


def check_miRNA_family(miRNA_dict,miR_family_path, TS_path):
    """
    Verifies if the not found miRNAs have a subscript format.
    If this is the case, the will be sorted back into the accounted for category.

    Parameters
    ----------
    miRNA_dict : dictionnary
        the list of sorted miRNAs from the categorize_findings function.
    miR_family_path : string
        The path to the miRNA family document.
    TS_path : string
        The path to the TargetScan prediction file

    Returns
    -------
    miRNA_dict : dictionnary
        a newly sorted version of the miRNAs.

    """
    miR_family=pd.read_csv(miR_family_path)
    TS_file=pd.read_csv(TS_path,sep='\t')
    TS_file=TS_file[TS_file['Gene Tax ID']==9606]
    for key in ['not_found_not_p', 'not_found_single_p', 'not_found_both_p']:
        for miR in miRNA_dict[key]:
            if type(miR)==float:
                continue
            miR_split=miR.split('.')[0]
            miR_split=miR_split.split('hsa-')[1]
            found_indices=np.where(miR_family['Human microRNA family'].str.contains(miR_split)==True)
            if len(found_indices[0])>0:
                for idx in found_indices[0]:
                    if type(miR_family.miRNAs[idx])==float:
                        continue
                    for val in miR_family.miRNAs[idx].replace('\xa0',' ').split():
                        if miR in val:
                            if re.search(r'\.\d',val) is not None:
                                if val in list(TS_file.miRNA):
                                    miRNA_dict['found'].append(val)
                                    miRNA_dict[key].remove(miR)
                                    miRNA_dict['not_found_all'].remove(miR)
                                    if key=='not_found_both_p':
                                        if miR.endswith('-3p')==True:
                                            no_p_miR=miR.split('-3p')[0]
                                            miRNA_dict['not_found_both_p'].remove(no_p_miR+'-5p')
                                            miRNA_dict['not_found_accounted_for'].append(no_p_miR+'-5p')
                                        else:
                                            no_p_miR=miR.split('-5p')[0]
                                            miRNA_dict['not_found_both_p'].remove(no_p_miR+'-3p')
                                            miRNA_dict['not_found_accounted_for'].append(no_p_miR+'-3p')                            
    return miRNA_dict





def check_non_conserved(miRNA_dict, non_conserved_path):
    """
    Checks if missing miRNAs are found in the non_conserved target scan file

    Parameters
    ----------
    miRNA_dict : Dictionnary
        A sorted dictionnary of miRNAs.
    non_conserved_path : string
        path to the non-conserved target scan file.

    Returns
    -------
    my_df : pandas.core.frame.DataFrame
        A dataframe with the finalized categorization.

    """
    TS_file_non_conserved=pd.read_csv(non_conserved_path,sep='\t')
    
    TS_file_non_conserved=TS_file_non_conserved[TS_file_non_conserved['Gene Tax ID']==9606]
    
    
    new_dict={}
    new_dict['found']=miRNA_dict['found']
    new_dict['not_found_all']=miRNA_dict['not_found_all']
    new_dict['not_found_accounted_for']=miRNA_dict['not_found_accounted_for']
    new_dict['not_found_not_accounted_for']=[]
    for key in ['not_found_not_p', 'not_found_single_p', 'not_found_both_p']:
        for miR in miRNA_dict[key]:
            found=False
            if miR not in TS_file_non_conserved.miRNA.unique():
                if miR+'.1' not in TS_file_non_conserved.miRNA.unique() or miR+'.2' not in TS_file_non_conserved.miRNA.unique():
    
                    new_dict['not_found_not_accounted_for'].append(miR)
                    found=True
            if found==False:
                new_dict['not_found_accounted_for'].append(miR)
                
        
    my_df=pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in new_dict.items() ]))
    return my_df
