#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 09:03:23 2021

@author: yohanl
"""
#import snake_utils as my_func
#from snake_scripts.snake_functions import snake_utils as my_utils
from snake_functions import snake_utils as my_utils
import os
import scanpy as sc
import numpy as np
import pandas as pd
import logging as logg
import math
import scipy.stats as stats

import matplotlib.pyplot as plt

#%% Folder management

def set_up_folders_reassignment_script(cell_line_var,samp_var):
    """
    Simple function which creates the necessary folders to store the results of the
    phase reassignment script
    

    Parameters
    ----------
    cell_line_var : string
        String indicating the cell line being used.
    samp_var : string
        string indicating the replicate being used.

    Returns
    -------
    main_path : string
        path to the main figures folder for the cell line and replicate.

    """
    
    my_utils.create_folder('all_figures/'+cell_line_var+'/'+samp_var+'/figures')
    main_path='all_figures/'+cell_line_var+'/'+samp_var
    
    my_utils.create_folder(main_path+'/figures/norm_dist_plots')
    
    my_utils.create_folder(main_path+'/figures/violin_plots/Pre')
    my_utils.create_folder(main_path+'/figures/violin_plots/Post')
    
    my_utils.create_folder(main_path+'/figures/scatter_plots/Pre')
    my_utils.create_folder(main_path+'/figures/scatter_plots/Post')
    
    my_utils.create_folder(main_path+'/figures/pca_bar_line_plots/initial')
    my_utils.create_folder(main_path+'/figures/pca_bar_line_plots/regressed')
    my_utils.create_folder(main_path+'/figures/pca_bar_line_plots/score_filtered')
    my_utils.create_folder(main_path+'/figures/pca_bar_line_plots/coordinate_filtered')
    # create_folder(main_path+'/figures/pca_bar_line_plots/cleaned_groups')
    my_utils.create_folder(main_path+'/figures/pca_bar_line_plots/filtered_reassigned')
    my_utils.create_folder(main_path+'/figures/pca_bar_line_plots/reassigned')
    
    return main_path


#%% Pre-processing

def filter_loom_barcode(adata,filter_list,new_file_name=None):
    """
    Filters cells based on provided list of barcodes
    Overwrites the AnnData object provided

    Parameters
    ----------
    adata : AnnData object
        The AnnData object to be filtered.
    filter_list : list
        A list of barcodes to filter out from the loom file.
    new_file_name : String, optional
        The name of the new loom file, if left to default, no file will be written. The default is None.

    Returns
    -------
    adata : AnnData object
        The filtered AnnData object.

    """
    
    velo_bc = []
    for i in adata.obs.index: 
        velo_bc.append(i.split(':')[1].rstrip()[:-1])
    
    #Get a list of indices to filter out
    ind_dict = dict((k,i) for i,k in enumerate(velo_bc))
    inter = set(ind_dict).intersection(filter_list)
    bad_indices = [ ind_dict[x] for x in inter ]
    
    #Creates and fills a list of indexes that are to be kept
    good_indices = []
    for i in ind_dict.values():
        if i in bad_indices:
            pass
        else:
            good_indices.append(i)
    
    
    #Creates the list to use in creating a anndata type
    keep_cells=[adata.obs.index[i] for i in good_indices]
    
    #Creates the new anndata by filtering the old anndata with the good_indices
    adata = adata[keep_cells, :]
    
    
    if new_file_name !=None:
        adata.write_loom(new_file_name+".loom", write_obsm_varm=False)
    
    return adata



def get_chrM_list(path):
    """
    Fetches an already generated list of mitochondrial genes

    Parameters
    ----------
    path : string
        path to the file containing mitochondrial gene names.

    Returns
    -------
    MT_list : list
        list of mitochondrial genes.

    """
    with open(path,'r') as f:
        MT_list=list(f.readlines())  
    MT_list = [x.strip() for x in MT_list] 
    return MT_list


def filter_based_on_spliced_unspliced_ratio(adata,layer_to_use,min_percent=None,max_percent=None):
    """
    Filters cells based on the percentage of reads of a layer in comparison with the main matrix.
    Only one percent can be inputted at a time.
    The AnnData object provided will be overwritten by the resulting filter
    
    Function written by Yohan Lefol
    
    Parameters
    ----------
    adata : AnnData
        The AnnData or loom object to be filtered.
    layer_to_use : string
        A string indicating the layer on which the filter will be performed.
    min_percent : int, optional
        The minimum percentage of layer reads per cell. The default is None.
    max_percent : int, optional
        The maximum percentage of layer reads per cell. The default is None.

    Returns
    -------
    None.

    """
    
    #Checks that only one of two thresholds has been inputed
    if min_percent is None and max_percent is None:
        print("A minimum or maximum percent threshold must be inputed")
        return()
    if min_percent is not None and max_percent is not None:
        print("Only one type of percent threshold can be inputed")
        return()
    
    #Generates total_reads
    X=adata.X
    total_reads=np.sum(X,axis=1) #Sum of all reads in each cell_ID
    total_reads=total_reads.astype(np.int64) #Converts float to int
    total_reads=total_reads.A1 #Flattens matrix into array

    #Checks that the layer entered exists
    if layer_to_use not in adata.layers.keys():
        print("The layer given was not found")
        return()
    
    #Generates layer reads
    X_layer=adata.layers[layer_to_use]
    total_layer_reads=np.sum(X_layer,axis=1) #Sum of all reads in each cell_ID
    total_layer_reads=total_layer_reads.astype(np.int64) #Converts float to int
    total_layer_reads=total_layer_reads.A1 #Flattens matrix into array
    
    #Creates the ratio of layer reads
    #Instantiates an empty list to store layer percentages per cell
    div_list=[]
    #Goes through each array and creates the percentages as necessary
    for idx,val in enumerate(total_reads):
        if total_layer_reads[idx] == 0 | val==0:
            div_list.append(0)
        else:
            div_list.append(total_layer_reads[idx]/val*100)
            
    #Converts the list of percentages to an array
    layer_percents=np.asarray(div_list)

    #Creates the mask array according to given threshold
    if min_percent is not None:
        layer_subset= layer_percents >= min_percent
    if max_percent is not None:
        layer_subset= layer_percents <= max_percent
    
    #Subsets the adata object
    adata._inplace_subset_obs(layer_subset)
    
from PIL import Image
def get_unspli_and_mito_thresholds(cell_line_var,samp_var):
    """
    A function which will show images and ask the user to input
    the maximum and minimum unspliced thresholds to use in the filtering
    of unspliced reads. It will also ask the users input for the mitochondrial
    threhsold to use.
    
    Function written by Yohan Lefol

    Parameters
    ----------
    cell_line_var : string
        a string indicating the cell line used.
    samp_var : string
        a string indicating the replicate used.

    Returns
    -------
    max_unspli : int
        maximum unspliced threhsold inputted by the user.
    min_unspli : int
        minimum unspliced threhsold inputted by the user.
    mito_thresh : int
        mtochodinrial threhsold inputted by the user.

    """
    img_1=Image.open('all_figures/'+cell_line_var+'/'+samp_var+'/figures/violin_plots/Pre/percent_MT.png')
    img_1.show()
    img_2=Image.open('all_figures/'+cell_line_var+'/'+samp_var+'/figures/violin_plots/Pre/unspliced.png')
    img_2.show()
    print("\n#######################################################################\n")
    print("What is the upper unspliced threshold in percentage: ")
    max_unspli=int(input())
    print("What is the minimum unspliced threshold in percentage: ")
    min_unspli=int(input())
    print("What is the mitochondrial reads threshold in percentage: ")
    mito_thresh=int(input())
    print("\n#######################################################################\n")
    img_1.close()
    img_2.close()
    
    return max_unspli,min_unspli,mito_thresh


def filter_adata_using_min_intron_count(adata,min_intron_count):  
    """
    Filters an AnnData object based on the number of unspliced reads in the
    unspliced layer for specific genes. Overwrites the provided AnnData file
    with it's filtered version
    
    Function written by Yohan Lefol

    Parameters
    ----------
    adata : AnnData
        The AnnData or loom file to be filtered.
    min_intron_count : int
        The minimum intron (unspliced) count per gene.

    Returns
    -------
    None.

    """
        
    X=adata.layers["unspliced"]
    
    #Counts everything on the zero axis
    nb_per_gene=np.sum(X>0,axis=0)
    
    #Flattens a matrix into an array
    nb_per_gene=nb_per_gene.A1
    #creates an array of True or False
    gene_subset=nb_per_gene>=min_intron_count
    #Subsets the adata file using the generated gene_subsets array
    adata._inplace_subset_var(gene_subset)
    
def filter_MT_percent(adata,MT_list,MT_percent_threshold):
    """
    Filters cells based on the percent of mitochondrial reads. The AnnData object
    will be overwritten by the resulting filter.
    
    Function written by Yohan Lefol
    
    Parameters
    ----------
    adata : AnnData
        The AnnData or loom object to be filtered.
    MT_list : list
        A list of mitochondrial genes, can be generated through the create_chrM_list() function.
    MT_percent_threshold : int
        An integer representing the percent threshold allowed.

    Returns
    -------
    None.

    """
    
    X=adata.X
    total_reads=np.sum(X,axis=1) #Sum of all reads in each cell_ID
    total_reads=total_reads.astype(np.int64) #Converts float to int
    total_reads=total_reads.A1 #Flattens matrix into array
    
    #Check quantity of MT genes in dataset
    #If as the solution is different based on if it is 1, >1 or 0
    MT_found=0
    the_one_found=''
    for gene in adata.var.index:
        if gene in MT_list:
            MT_found=MT_found+1
            the_one_found=gene

    #If only one MT gene was found in the dataset
    if MT_found==1: 
        number_mito_per_cell=adata.obs_vector(the_one_found)
        number_mito_per_cell=number_mito_per_cell.astype(np.int64)
        
    elif MT_found>1:
        #Create an adata subset with only mitochondrial genes
        mito_gene_indicator= np.in1d(adata.var_names,MT_list)
        adata_mito=adata[:,mito_gene_indicator]
        
        #Create an array with only mitochondrial counts per cell
        X_mito=adata_mito.X
        number_mito_per_cell = np.sum(X_mito,axis=1) #sum of all reads for cells
        total_reads=total_reads.astype(np.int64) #Converts float to int
        number_mito_per_cell = number_mito_per_cell.A1 #Flattens matrix into array
    else:
        print("No mitochondrial genes found in data set")
        return()
    
    #Instantiates an empty list to store MT_percentages per cell
    div_list=[]
    #Goes through each array and creates the percentages as necessary
    for idx,val in enumerate(total_reads):
        if number_mito_per_cell[idx] == 0 | val==0:
            div_list.append(0)
        else:
            div_list.append(number_mito_per_cell[idx]/val*100)
            
    #Converts the list of percentages to an array
    mito_percent=np.asarray(div_list)     
        
    #Creates a True/False mask to use in an adata subset
    cell_subset=mito_percent<=MT_percent_threshold        

    #Subsets the adata file using the generated cell_subset array
    adata._inplace_subset_obs(cell_subset)


def check_cols_and_rows(adata):
    """
    Simple function to check that there are no genes with 0 reads
    
    Function written by Yohan Lefol

    Parameters
    ----------
    adata : AnnData
        The AnnData object.

    Returns
    -------
    None.

    """
    
    if np.any(adata.X.sum(axis=0)==0)==True:
        start=len(adata.var_names)
        sc.pp.filter_genes(adata,min_counts=1)
        number_removed=abs(len(adata.var_names)-start)
        print("removed ",number_removed," genes with 0 reads")
    
    
def prepare_CC_lists(CC_path):
    """
    Function that reads a file and retrieves the genes associated to each phase
    
    Function written by Geir Armund Svan Hasle

    Parameters
    ----------
    CC_path : string
        path to file containing genes with associated cell cycle phases.

    Returns
    -------
    G1_list : list
        list of G1 genes.
    S_list : list
        list of S genes.
    G2M_list : list
        list of G2M genes.

    """
    df = pd.read_csv(CC_path,delimiter=',')
    S_list=[]
    G1_list=[]
    G2M_list=[]
    
    for idx,val in enumerate(df['gene']):
        phase=df['phase'][idx]
        S_list.append(val) if phase=='S' else None
        G1_list.append(val) if phase=='G1' else None
        G2M_list.append(val) if phase=='G2/M' else None

    return G1_list,S_list,G2M_list


def select_for_CC_gene_markers(adata, G1_list, S_list, G2M_list):
    """
    Used to subset AnnData object to only include cell cycle regulating genes
    
    Function written by Yohan Lefol

    Parameters
    ----------
    adata : AnnData
        The AnnData object.
    G1_list : list
        list of G1 genes.
    S_list : list
        list of S genes.
    G2M_list : list
        list of G2M genes.

    Returns
    -------
    None.

    """
    
    bool_df=pd.DataFrame(index=adata.var_names,columns=["CC_markers"])
    for idx,val in enumerate(bool_df.index):
        if val in S_list:
            bool_df["CC_markers"][idx]=True
        elif val in G1_list:
            bool_df["CC_markers"][idx]=True
        elif val in G2M_list:
            bool_df["CC_markers"][idx]=True
        else:
            bool_df["CC_markers"][idx]=False
    
    bool_df=bool_df.astype(bool)
    adata.var['highly_variable'] = bool_df["CC_markers"].values
    #return adata
        
def selection_method(adata,highly_variable=True,CC_path=None):
    """
    Creates a subset of an AnnData object based on a selection method.
    Selecting for highly variable genes using Scanpy's method, or selecting for
    cell cycle regulating genes. Subsets stored as highly_variable for simplicity
    
    Function written by Yohan Lefol

    Parameters
    ----------
    adata : AnnData Object
        The AnnData/loom file.
    highly_variable : Boolean, optional
        if true, the selection method will be scanpy's highly variable gene selection
        otherwise it will select for cell cycle regulating genes. The default is True.
    CC_path : string
        path to file containing genes with associated cell cycle phases.
        

    Returns
    -------
    adata : AnnData object
        The overwritten AnnData object.

    """
    if highly_variable==True:
        sc.pp.highly_variable_genes(adata, min_mean=0.01235,max_mean=3,min_disp=0.5)
    else:
        g1_list,s_list,g2m_list=prepare_CC_lists(CC_path)
        select_for_CC_gene_markers(adata,g1_list,s_list,g2m_list)
    adata = adata[:, adata.var.highly_variable]
    return adata

#%% Cell cycle scoring


def score_genes_in_layer(
        adata,
        gene_list,
        layer_choice='spliced',
        ctrl_size=50,
        gene_pool=None,
        n_bins=25,
        score_name='score',
        random_state=0,
        copy=False,
        use_raw=False,
        supress_gene_warning=False):  # we use the scikit-learn convention of calling the seed "random_state"
    """Score a set of genes [Satija15]_.

    The score is the average expression of a set of genes subtracted with the
    average expression of a reference set of genes. The reference set is
    randomly sampled from the `gene_pool` for each binned expression value.

    This reproduces the approach in Seurat [Satija15]_ and has been implemented
    for Scanpy by Davide Cittaro.
    
    Modifications by Y.Lefol: Added the option of specifying the layer to work on

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        The annotated data matrix.
    gene_list : iterable
        The list of gene names used for score calculation.
    layer_choice : `string`
        String containing the layer to be used as a matrix.
    ctrl_size : `int`, optional (default: 50)
        Number of reference genes to be sampled. If `len(gene_list)` is not too
        low, you can set `ctrl_size=len(gene_list)`.
    gene_pool : `list` or `None`, optional (default: `None`)
        Genes for sampling the reference set. Default is all genes.
    n_bins : `int`, optional (default: 25)
        Number of expression level bins for sampling.
    score_name : `str`, optional (default: `'score'`)
        Name of the field to be added in `.obs`.
    random_state : `int`, optional (default: 0)
        The random seed for sampling.
    copy : `bool`, optional (default: `False`)
        Copy `adata` or modify it inplace.
    use_raw : `bool`, optional (default: `False`)
        Use `raw` attribute of `adata` if present.
    supress_gene_warning : `bool'
        Supress gene warning if True
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with an additional field
    `score_name`.

    Examples
    --------
    See this `notebook <https://github.com/theislab/scanpy_usage/tree/master/180209_cell_cycle>`__.
    """
    
    from scipy.sparse import issparse

    # start = logg.info(f'computing score {score_name!r}')
    adata = adata.copy() if copy else adata

    if random_state:
        np.random.seed(random_state)

    gene_list_in_var = []
    var_names = adata.raw.var_names if use_raw else adata.var_names
    for gene in gene_list:
        if gene in var_names:
            gene_list_in_var.append(gene)
        else:
            if supress_gene_warning==False:
                logg.warning(f'gene: {gene} is not in adata.var_names and will be ignored')
    gene_list = set(gene_list_in_var[:])

    if not gene_pool:
        gene_pool = list(var_names)
    else:
        gene_pool = [x for x in gene_pool if x in var_names]

    # Trying here to match the Seurat approach in scoring cells.
    # Basically we need to compare genes against random genes in a matched
    # interval of expression.

    _adata = adata.raw if use_raw else adata
    if issparse(_adata.layers[layer_choice]):
        obs_avg = pd.Series(
            np.nanmean(
                _adata[:, gene_pool].layers[layer_choice].toarray(), axis=0), index=gene_pool)  # average expression of genes
    else:
        obs_avg = pd.Series(
            np.nanmean(_adata[:, gene_pool].layers[layer_choice], axis=0), index=gene_pool)  # average expression of genes

    obs_avg = obs_avg[np.isfinite(obs_avg)] # Sometimes (and I don't know how) missing data may be there, with nansfor

    n_items = int(np.round(len(obs_avg) / (n_bins - 1)))
    obs_cut = obs_avg.rank(method='min') // n_items
    control_genes = set()

    # now pick `ctrl_size` genes from every cut
    for cut in np.unique(obs_cut.loc[list(gene_list)]):
        r_genes = np.array(obs_cut[obs_cut == cut].index)
        np.random.shuffle(r_genes)
        control_genes.update(set(r_genes[:ctrl_size]))  # uses full r_genes if ctrl_size > len(r_genes)

    # To index, we need a list - indexing implies an order.
    control_genes = list(control_genes - gene_list)
    gene_list = list(gene_list)


    X_list = _adata[:, gene_list].layers[layer_choice]
    if issparse(X_list): X_list = X_list.toarray()
    X_control = _adata[:, control_genes].layers[layer_choice]
    if issparse(X_control): X_control = X_control.toarray()
    X_control = np.nanmean(X_control, axis=1)

    if len(gene_list) == 0:
        # We shouldn't even get here, but just in case
        logg.hint(
            f'could not add \n'
            f'    {score_name!r}, score of gene set (adata.obs)'
        )
        return adata if copy else None
    elif len(gene_list) == 1:
        score = _adata[:, gene_list].layers[layer_choice] - X_control
    else:
        score = np.nanmean(X_list, axis=1) - X_control

    adata.obs[score_name] = pd.Series(np.array(score).ravel(), index=adata.obs_names)

    # logg.info(
    #     '    finished',
    #     time=start,
    #     deep=(
    #         'added\n'
    #         f'    {score_name!r}, score of gene set (adata.obs)'
    #     ),
    # )
    return adata if copy else None


def my_score_genes_cell_cycle_improved(
        adata,
        layer_choice,
        CC_path,
        copy=False,
        **kwargs):
    """Score cell cycle genes.
    Given two lists of genes associated to S phase and G2M phase, calculates
    scores and assigns a cell cycle phase (G1,S or G2M). See 
    :func:`~scanpy.api.score_genes` for more information.

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        The annotated data matrix.
    layer_choice : `string`
        String containing the layer to be used as a matrix.
    CC_path : `string'
        String for the path to the file containing the cell cycle genes and their phases
    copy : `bool`, optional (default:`False`)
        DESCRIPTION. The default is False.
    **kwargs : optional keyword arguments
        Are passed to :func:`~scanpy.api.score_genes`. `ctrl_size` is not
        possible, as it's set as `min(len(s_genes), len(g2m_genes))`.

    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    **G1_score** : `adata.obs`, dtype `object`
        The score for G1 phase for each cell.
    **S_score** : `adata.obs`, dtype `object`
        The score for S phase for each cell.
    **G2M_score** : `adata.obs`, dtype `object`
        The score for G2M phase for each cell.
    **phase** : `adata.obs`, dtype `object`
        The cell cycle phase (`S`,`G2M` or `G1`) for each cell/
    
    See also
    -------
    score_genes
    Examples
    -------
    See this `notebook <https://github.com/theislab/scanpy_usage/tree/master/180209_cell_cycle>`__.
    """    #logg.info('calculating cell cycle phase')    
    df = pd.read_csv(CC_path,delimiter=',')
    s_genes=[]
    g1_genes=[]
    g2m_genes=[]
    
    for idx,val in enumerate(df['gene']):
        phase=df['phase'][idx]
        s_genes.append(val) if phase=='S' else None
        g1_genes.append(val) if phase=='G1' else None
        g2m_genes.append(val) if phase=='G2/M' else None
    
    
    
    adata = adata.copy() if copy else adata
    ctrl_size = min(len(s_genes), len(g2m_genes), len(g1_genes))
    s_n_bins = round ((len(g1_genes)+len(g2m_genes))/ctrl_size)
    g1_n_bins = round ((len(s_genes)+len(g2m_genes))/ctrl_size)
    g2m_n_bins = round ((len(g1_genes)+len(s_genes))/ctrl_size)
    
    score_dict={}
    for i in range(50):
        #Avoid printing the same warnings 50 times
        if i != 0:
            #add s-score
            score_genes_in_layer(adata, gene_list=s_genes, layer_choice=layer_choice, score_name='S_score', ctrl_size=ctrl_size, n_bins=s_n_bins, supress_gene_warning=True, **kwargs)
            #add g2m-score
            score_genes_in_layer(adata, gene_list=g2m_genes, layer_choice=layer_choice, score_name='G2M_score', ctrl_size=ctrl_size, n_bins=g2m_n_bins, supress_gene_warning=True, **kwargs)
            #add g1-score
            score_genes_in_layer(adata, gene_list=g1_genes, layer_choice=layer_choice, score_name='G1_score', ctrl_size=ctrl_size, n_bins=g1_n_bins, supress_gene_warning=True, **kwargs)
        else:#Shows warnings
            #add s-score
            score_genes_in_layer(adata, gene_list=s_genes, layer_choice=layer_choice, score_name='S_score', ctrl_size=ctrl_size, n_bins=s_n_bins, **kwargs)
            #add g2m-score
            score_genes_in_layer(adata, gene_list=g2m_genes, layer_choice=layer_choice, score_name='G2M_score', ctrl_size=ctrl_size, n_bins=g2m_n_bins, **kwargs)
            #add g1-score
            score_genes_in_layer(adata, gene_list=g1_genes, layer_choice=layer_choice, score_name='G1_score', ctrl_size=ctrl_size, n_bins=g1_n_bins, **kwargs)
            
            
        if not 'G1_score' in adata.obs.columns:
            print("WARNING: No G1-genes found in data set. Computing G1-score as -sum (S_score,G2M_score)")
            adata.obs['G1_score'] = -adata.obs[['S_score','G2M_score']].sum(1)
            
        scores = adata.obs[['S_score','G2M_score','G1_score']]
        score_dict[i]=scores
    
    #Reset warnings
    new_score_dict={}
    for s_type in ['S_score','G2M_score','G1_score']:
        new_score_dict[s_type]=np.zeros(shape=(len(adata.obs_names),len(score_dict)))
    
    for idx, df in score_dict.items():
            for key in new_score_dict.keys():
                new_score_dict[key][:,idx]=df[key].values
    
    for key in new_score_dict.keys():
        new_score_dict[key]=new_score_dict[key].mean(axis=1)
    
    scores=pd.DataFrame.from_dict(new_score_dict)
    scores.index=score_dict[0].index
    
    
    adata.obs['S_score'] = scores.S_score
    adata.obs['G2M_score'] = scores.G2M_score
    adata.obs['G1_score'] = scores.G1_score
    
    
    
    #default phase is 'not_assigned'
    phase = pd.Series('not_assigned', index=scores.index)
    
    #The = is to remove all `not assigned' as they cause issues downstream. if scores
    #are identical, then wether it is in one phase or the other does not matter much
    
    #if G2M is higher than S and G1, it's G2M
    phase[(scores.G2M_score >= scores.S_score) & (scores.G2M_score >= scores.G1_score)] = 'G2M'
    
    #if S is higher than G2M and G1, it's S
    phase[(scores.S_score >= scores.G2M_score) & (scores.S_score >= scores.G1_score)] = 'S'
    
    #if G1 is higher than G2M and S, it's G1
    phase[(scores.G1_score >= scores.G2M_score) & (scores.G1_score >= scores.S_score)] = 'G1'
    
    #Checks that each phase is represented by at least one cell
    #If that is not the case, the cell expressing the highest score of the mising
    #phase will be reassigned to that phase.
    for check_phase in  ['G1','S','G2M']:
        if check_phase not in phase.unique():
            score_var=check_phase+'_score'
            highest_phase_score=scores[score_var].idxmax()
            phase.at[highest_phase_score]=check_phase
    adata.obs['phase'] = phase
    #logg.hint(' \'phase\',cell cycle phase (adata.obs)') )
    return adata if copy else None


def score_ordering_mod(expression_df,phase):
    """
    Orders the scores for a specific phase for the 'compare_marker_genes_per_phase_mod' function
    
    Function initially written by Geir Armund Svan Hasle, adapted for
    lower RAM usageby Yohan Lefol
    

    Parameters
    ----------
    expression_df : pandas dataframe
        The expressions of the genes to be ordered by score.
    phase : string
        specifies the phase.

    Returns
    -------
    None.

    """
    p=phase
    score_dict = {}
    ind_dict = {'G1': 0, 'S': 1, 'G2/M': 2, 'G2': 2}
    total_score = 0
    # for p in expression_df['known_phase'].unique():
    expr_mat = expression_df[expression_df['known_phase'] == p]
    expr_mat= expr_mat[['G1_expr','S_expr','G2M_expr']]
    expr_mat = expr_mat.values
    score_list = []
    for row in expr_mat:
        score_list.append(int(np.where(row == np.max(row))[0][0] == ind_dict[p]))   
    phase_score = sum(score_list)
    score_dict[p] = phase_score
    total_score += phase_score
    
    print(p," phase: ","{}/{} genes classified correctly".format(phase_score,len(expression_df[expression_df['known_phase'] == p])))
        
    del score_dict#To save on RAM

def compare_marker_genes_per_phase_mod(data,cc_path, phase_choice,do_plots=True,plot_path="./figures"):
    """
    A function which grades a AnnData file based on the expression of each gene for each cell
    and wether or not those genes have been associated to the expected phase
    The function gives the ability to plot, however it should be noted that this
    function can be quite RAM intensive and should be used accordingly
    
    Function initially written by Geir Armund Scan Hasle, adapted for lower RAM
    usage by Yohan Lefol

    Parameters
    ----------
    data : AnnData object
        The AnnData object containing the gene/cell and phase data.
    cc_path : string
        The file path to the cell cycle candidate list
    phase_choice : string
        G1, S, or G2M, used to select which phase to grade.
    do_plots : boolean, optional
        If plots are to be made. The default is True.
    plot_path : boolean, optional
        Where plots should be saved. The default is "./figures".

    Returns
    -------
    None.

    """
    
    
    g1_start=data.uns['phase_boundaries']['g1_start']
    s_start=data.uns['phase_boundaries']['s_start']
    g2m_start=data.uns['phase_boundaries']['g2m_start']
    
    known_df = pd.read_csv(cc_path,delimiter=',').dropna()
    if phase_choice=='G2M':
        p='G2/M'
    else:
        p=phase_choice
    expression_df = pd.DataFrame()
    p_df = known_df[known_df['phase'] == p]   #G2M == G2/M
    for i, r in p_df.iterrows():
        if not r['gene'] in list(data.var.index):
            continue
        gene = r['gene']  
        if do_plots==True:  #Adds total expression
            tot_expr = data[:,gene].X.mean()
            expression_df = expression_df.append({
                'known_phase': p,
                'gene_symbol': gene,
                'G1_expr': data[g1_start:s_start-1,:][:,gene].X.mean(),
                'S_expr': data[s_start:g2m_start-1,:][:,gene].X.mean(),
                'G2M_expr': data[g2m_start:,:][:,gene].X.mean(),
                'Total_expr': tot_expr,
            }, 
                ignore_index=True)
        else:   #Does not add total expression
            expression_df = expression_df.append({
                'known_phase': p,
                'gene_symbol': gene,
                'G1_expr': data[g1_start:s_start-1,:][:,gene].X.mean(),
                'S_expr': data[s_start:g2m_start-1,:][:,gene].X.mean(),
                'G2M_expr': data[g2m_start:,:][:,gene].X.mean(),
            }, 
                ignore_index=True)
    if expression_df.empty==True:
        print("No genes found for phase ",p)
        return
    expression_df.index = expression_df['gene_symbol']
    expression_df=expression_df.drop('gene_symbol',axis=1)
    #print(expression_df)
    if do_plots==True:
        if not os.path.exists(plot_path):
            os.makedirs(plot_path, exist_ok=True)
        for p in expression_df['known_phase'].unique():
            phase_df = expression_df[expression_df['known_phase'] == p].copy()
            phase_df=phase_df.astype({'G1_expr':float,'G2M_expr':float,'S_expr':float,'Total_expr':float})
            phase_df['G1_expr']=phase_df['G1_expr'].abs()
            phase_df['G2M_expr']=phase_df['G2M_expr'].abs()
            phase_df['S_expr']=phase_df['S_expr'].abs()
            phase_df['Total_expr']=phase_df['Total_expr'].abs()
            if phase_df.empty:
                print("No {} genes detected".format(p))
                continue
            ax = phase_df.plot(kind='bar', title='Mean expression of {} marker genes by modelled phase'.format(p), figsize=(10,10))
            ax.set_xlabel('Gene symbols')
            ax.set_ylabel('Reads Per Million')
            plt.savefig(os.path.join(plot_path,"mean_expression_{}.pdf".format(p.replace('/','-'))))
    
    score_ordering_mod(expression_df,p)
    del expression_df #To save on RAM
    # return expression_df

#%% Cell cycle score filtering and reassignment

def score_filter_reassign(adata,cell_percent):
    """
    Calcualtes the necessary number of cells, then takes the the top x amount of cells
    for each phase score and associates that cell to the phase of that score.
    After having done the reassignment, it overwrites the AnnData object with
    the updated AnnData object.
    
    Function written by Yohan Lefol

    Parameters
    ----------
    adata : AnnData
        The AnnData object.
    cell_percent : float
        A percentage value in decimals.

    Returns
    -------
    None.

    """
    cell_num=int(len(adata.obs)*cell_percent)
    index_list=[]
    phase_series=pd.Series('not_assigned', index=adata.obs.index)
    for p in ['G1','S','G2M']:        
        if p == 'G1':
            score_series=adata.obs.G1_score.sort_values(ascending=False)
        elif p == 'S':
            score_series=adata.obs.S_score.sort_values(ascending=False)
        else:
            score_series=adata.obs.G2M_score.sort_values(ascending=False)
            
        for idx,val in enumerate(score_series):
            if idx<=cell_num:
                index_list.append(score_series.index[idx])
                cell_index=np.where(adata.obs.index==score_series.index[idx])[0][0]
                
                phase_series[cell_index]=p
                # adata.obs.phase[cell_index]=p  
    
    adata.obs['phase'] = phase_series
    bool_list=[]
    for index in adata.obs.index:
        bool_list.append(True) if index in index_list else bool_list.append(False)
            
    adata._inplace_subset_obs(bool_list)



def coordinate_filter_reassign(adata,cell_percent):
    """
    Sorts all the cells based on their distance from the 0,0 coordinate. The function
    assumes that the coordinates used are PC1 and PC2. Once sorted, it removes the
    top x amount of cells closest to the 0,0 point. The AnnData object is then
    overwritten with the updated AnnData object.
    
    Function written by Yohan Lefol

    Parameters
    ----------
    adata : AnnData
        The AnnData object.
    cell_percent : float
        A percentage value in decimals.

    Returns
    -------
    None.

    """
    cell_pca_dict={}
    cell_pca_dict_x={}
    cell_pca_dict_y={}
    for idx,var in enumerate(adata.obs_names):
        cell_pca_dict[var]=list(abs(adata.obsm['X_pca'][idx])[0:2])
        cell_pca_dict_x[var]=abs(adata.obsm['X_pca'][idx])[0]
        cell_pca_dict_y[var]=abs(adata.obsm['X_pca'][idx])[1]
    
    sorted_x = sorted(cell_pca_dict_x.items(), key=lambda item: item[1])
    my_inc=0
    ranked_x={}
    for idx,val in enumerate(sorted_x):
        ranked_x[val[0]]=my_inc
        my_inc+=1
        
    sorted_y = sorted(cell_pca_dict_y.items(), key=lambda item: item[1])
    my_inc=0
    ranked_y={}
    for idx,val in enumerate(sorted_y):
        ranked_y[val[0]]=my_inc
        my_inc+=1
    
    cell_ranks={}
    for x in ranked_x.items():
        for y in ranked_y.items():
            if x[0]==y[0]:
                cell_ranks[x[0]]=(x[1]+y[1])
                break  
    cell_ranks = sorted(cell_ranks.items(), key=lambda item: item[1])

    num_to_filter=round(cell_percent*len(cell_ranks))
    cell_filter_list=[i[0] for i in cell_ranks[0:num_to_filter]]
    
    bool_list=[]
    for index in adata.obs.index:
        bool_list.append(False) if index in cell_filter_list else bool_list.append(True)
            
    adata._inplace_subset_obs(bool_list)



def compute_angles(points):
    """
    A function that calculates the angles of the cells based on the PCA coordinates
    Cells are first assigned from -pi to pi. Cells assigned to -pi are translated
    by 2pi so that each cell has an angle correspongin to one period if the cell cycle
    were to be shown as a circle.
    
    Function written by Geir Armund Svan Hasle

    Parameters
    ----------
    points : PCA data points
        Often given using adata.obsm['X_pca'][:,:2].

    Returns
    -------
    angles : numpy array
        An array containing the angles of each cell.

    """
    angles = np.arctan2(points[:,0], points[:,1])
    angles[angles < 0] += 2*np.pi
    return angles


def find_angle_boundaries(adata):
    """
    Finds the angle boundaries based on the found order/phase boundaries
    
    Function written by Yohan Lefol
    
    Parameters
    ----------
    adata : AnnData
        AnnData object.

    Returns
    -------
    g1_ang_start : float
        The angle boundary for G1.
    s_ang_start : float
        The angle boundary for S.
    g2m_ang_start : float
        The angle boundary for G2M.

    """
    adata = adata[adata.obs['order'].argsort(),:]
    g1_ang_start=adata.obs.angles[adata.uns['phase_boundaries']['g1_start']]
    s_ang_start=adata.obs.angles[adata.uns['phase_boundaries']['s_start']]
    g2m_ang_start=adata.obs.angles[adata.uns['phase_boundaries']['g2m_start']]
    #OLD VERSION
    # for idx,val in enumerate(adata.obs.order):
    #     if val == adata.uns['phase_boundaries']['g1_start']:
    #         g1_ang_start=adata.obs.angles[idx]
    #     elif val == adata.uns['phase_boundaries']['s_start']:
    #         s_ang_start=adata.obs.angles[idx]
    #     elif val == adata.uns['phase_boundaries']['g2m_start']:
    #         g2m_ang_start=adata.obs.angles[idx]
    
    return g1_ang_start, s_ang_start, g2m_ang_start


def phase_angle_assignment(adata,g1_limit,s_limit,g2m_limit):
    """
    Reassigns the phases of each cell based on their angles and the inputted
    phase angle boundaries.
    
    Function written by Yohan Lefol

    Parameters
    ----------
    adata : AnnData Object
        The AnnData object containing the cells and associated phases.
    g1_limit : float
        The g1 angle boundary.
    s_limit : float
        the s angle boundary.
    g2m_limit : float
        the G2M angle boundary.

    Returns
    -------
    None.

    """
    
    list_of_keys=[]
    list_of_values=[]
    
    angle_dict={}
    angle_dict['G1']=g1_limit
    angle_dict['G2M']=g2m_limit
    angle_dict['S']=s_limit
    angle_dict={k: v for k, v in sorted(angle_dict.items(), key=lambda item: item[1])}
    for i in angle_dict.keys():
        list_of_keys.append(i)
    for i in angle_dict.values():
        list_of_values.append(i)
    
    phase_series=pd.Series('not_assigned', index=adata.obs.index)
    for idx, val in enumerate(adata.obs.angles):
        if val >= list_of_values[0] and val<list_of_values[1]:
            phase_series[adata.obs.index[idx]]=list_of_keys[0]
        elif val >=list_of_values[1] and val< list_of_values[2]:
            phase_series[adata.obs.index[idx]]=list_of_keys[1]
        else:
            phase_series[adata.obs.index[idx]]=list_of_keys[2]
            
    adata.obs['phase'] = phase_series
    
    
def shift_data(data, n, direction = 'positive', reverse = False):
    """
    shifts the order of the AnnData object by a selected amount in the selected
    direction
    
    Function written by Geir Armund Svan Hasle

    Parameters
    ----------
    data : AnnData object
        The data that will be shifted.
    n : int
        The distance by which the data will be shifted.
    direction : string, optional
        positive or negative; determines the direction of the shift. The default is 'positive'.
    reverse : boolean, optional
        reverses the data if True. The default is False.

    Raises
    ------
    ValueError
           If the direction inputted is invalid

    Returns
    -------
    data : AnnData
        Return the shifted data.

    """
    
    if not direction in ['positive', 'negative']:
        raise ValueError('direction must be: positive,negative')
    #Need to find g2m g1 junction
    if direction == 'negative':
        data.obs['order'] -= n
        data.obs['order'][data.obs['order'] < 0] += len(data.obs)
    else:
        data.obs['order'] += n
        data.obs['order'][data.obs['order'] > len(data.obs)] -= len(data.obs)
    sort_order = data.obs['order'].argsort()[::-1] if reverse else data.obs['order'].argsort() 
    data = data[sort_order,:].copy()
    data.obs['order'] = np.arange(len(data.obs))
    return data

#%% Find boundaries - normal distribution method

def normal_dist_boundary_wrapper(my_adata,save_path):
    """
    A wrapper function which calculates the normal distribution of the S and G2M
    phase along with the kenrel density of the G1 phase. The kde of the G1 is plotted
    as well as the merged S,G2M normal distribution with the kde of G1.
    
    The function then determines the orientation of the data using the means of S, G2M,
    and G1.
    
    The crossover points are then found for S-G2M, G1 crosses are then extracted from
    this information.
    
    Restults are loaded into a dictionnary and returned.

    Parameters
    ----------
    my_adata : AnnData object
        The AnnData object to be filtered.
    save_path : string
        The location in which the plots will be saved.

    Returns
    -------
    dict_crossover : dictionnary
        A dictionnary containing three keys/values for the crossover indexes.
    orientation : string
        Either G1 or G2M to indicate the orientation of the dataset.

    """
    #Calculated normal distribution of S and G2M, also calculated G1 kernel density
    norm_dist_dict,x,mean_dict=calc_norm_dist(my_adata,save_path)
    
    #Plots the normal distribution
    save_name=save_path+'norm_dist_plot.pdf'
    plot_norm_dist(norm_dist_dict,x,save_name)
    
    #Creates two phase lists, one with all phases, another with only S and G2M
    list_of_phases,list_s_g2m=create_list_of_phases(norm_dist_dict)
    
    #Find the mean angle of G1
    mean_dict=find_G1_mean(my_adata,list_of_phases,mean_dict)
    
    #Determine the orientation using the mean angle of the phases
    orientation=find_orientation_from_mean_angles(mean_dict)
    
    if orientation=='G1':
        P1='S'
        P2='G2M'
    else:
        P1='G2M'
        P2='S'
        
    #Identify crossover points for S and G2M. One will be identified as the 'roll_cross'
    #Which represents the correct location of the S G2M cross over, the second point
    #called 'target_cross' is used to identify the location of where G1 grosses will
    #be calculated
    S_G2M_crosses=find_S_G2M_crossovers(list_s_g2m,P1,P2)
    
    #Identify both G1 crossover points
    found_idx_P1=find_G1_crossover(list_of_phases,orientation,S_G2M_crosses,look_for=P1)
    found_idx_P2=find_G1_crossover(list_of_phases,orientation,S_G2M_crosses,look_for=P2)
    
    #Load results in a crossover dict which is later used for reassignment.
    dict_crossover=create_crossover_dict(my_adata,found_idx_P1,found_idx_P2,S_G2M_crosses,orientation)
    
    
    return dict_crossover,orientation


def calc_norm_dist(my_adata,save_path):
    """
    Function which calculates the normal distribution of S and G2M phase along with 
    kernel density of G1. The kernel density of G1 is plotted alone and with the 
    normal distribution of S and G2M.
    
    For the calculation of the normal distribution, the calculation is attempted
    with the angles as found in the anndata object, and with the angles shifted 
    by 180 degrees. The version with the lowest variability is used for the calculation
    of the normal distribution.

    Parameters
    ----------
    my_adata : AnnData object
        The AnnData object to be filtered.
    save_path : string
        The location in which the plots will be saved.

    Returns
    -------
    norm_dist_dict : dictionnary
        Contains the normal distribution values for S and G2M and the kernal density
        values for G1.
    x : pandas.series
        All the angles in the dataset.
    mean_dist_dict : dictionnary
        Contains the mean angles of S and G2M.

    """
    #Calculate the mean and standard deviation
    #Mean is stored in 0 index and std in index 1
    norm_dist_dict={}
    mean_dist_dict={}
    for phase in my_adata.obs['phase'].unique():
        
        x=my_adata.obs['angles'].copy()
        
        #Calculate and plot the kernel density for the G1 phase
        if phase=='G1':
            g1_idx=list(np.where(my_adata.obs['phase']=='G1')[0])      
            g1_data=x[g1_idx]
            fig = plt.figure()
            g1_data=g1_data.plot.kde(ind=x)
            fig.savefig(save_path+'/G1_kernel_density_plot.png')
            
            g1_data=g1_data.get_lines()[0].get_xydata()
            g1_data=g1_data[:,1]
            norm_dist_dict[phase]=list(g1_data)
        #Phase is either S or G2M
        else:
            #Creates a 180 degree shifted version of x (the angles)
            rolled_x=x.copy()
            for cell in rolled_x.index:
                if rolled_x[cell]-math.pi<0:
                    rolled_x[cell]=2*math.pi-abs(rolled_x[cell]-math.pi)
                else:
                    rolled_x[cell]=rolled_x[cell]-math.pi
            
            norm_dist_dict[phase]=[]
            
            #Normal calculation
            #Calculate mean and stdev
            ang_subset=x[np.where(my_adata.obs['phase']==phase)[0]]
            variance = ang_subset.std()
            
            #Calculate the values using the rolled x
            ang_subset_roll=rolled_x[np.where(my_adata.obs['phase']==phase)[0]]
            variance_roll = ang_subset_roll.std()
    
            # Replaces the variance
            if variance_roll < variance:
                variance=variance_roll
                mu = ang_subset_roll.mean()
                #Since we are using the shifted values, we add pi back to the mean
                #In order to preserve the correct location on the graph
                if mu+math.pi>2*math.pi:
                    mu=(mu+math.pi)-2*math.pi
                else:
                    mu=mu+math.pi
            else:
                mu = ang_subset.mean()
                
            #Calculate sigma
            sigma = math.sqrt(variance)
            
            #Load the mean into the dictionnary
            mean_dist_dict[phase]=mu
            
            #Calculate the normal distribution and add it to the dictionnary
            for ang in x:
                norm_dist_dict[phase].append(max(stats.norm.pdf(ang, mu- (2*math.pi), sigma), stats.norm.pdf(ang, mu, sigma), stats.norm.pdf(ang, mu+ (2*math.pi), sigma)))
        
    return norm_dist_dict,x,mean_dist_dict



def create_list_of_phases(norm_dist_dict):
    """
    Function which uses the normal distribution / kernel density results
    to create a list of phases. The list of phases represents the phase with
    the highest value at each point/angle on the x axis.
    
    The function also generates a list of phases using only S and G2M. This list is
    used later to establish the S-G2M crossover points, which then serve as a means
    to better identify the correct location of G1.

    Parameters
    ----------
    norm_dist_dict : dictionnary
        A dictionnary containing the normal distribution values of S and G2M along
        with the kernel density values of G1.

    Returns
    -------
    list_of_phases : list
        A list of phases (G1,S,or G2M) indicating which phase is dominant at each angles.
    list_s_g2m : list
        A list of phases which excludes G1 from the possibilies.

    """
    G1_vals=norm_dist_dict['G1']
    G2M_vals=norm_dist_dict['G2M']
    S_vals=norm_dist_dict['S']
    
    list_of_phases=[]
    list_s_g2m=[]
    
    for idx in range(len(G1_vals)):
        #Prep a phase list to find the S-G2M crossover without accounting for G1
        if S_vals[idx]>G2M_vals[idx]:
            list_s_g2m.append('S')
        else:
            list_s_g2m.append('G2M')
        
        if G1_vals[idx]>G2M_vals[idx] and G1_vals[idx]>S_vals[idx]:
            phase='G1'
        elif S_vals[idx]>G2M_vals[idx] and S_vals[idx]>G1_vals[idx]:
            phase='S'
        elif G2M_vals[idx]>G1_vals[idx] and G2M_vals[idx]>S_vals[idx]:
            phase='G2M'
        list_of_phases.append(phase)
    
    return list_of_phases,list_s_g2m


def find_G1_mean(my_adata,list_of_phases,mean_dict):
    """
    Function which finds the 'mean' of G1 using the list of phases. 
    
    Ideally G1 will be isolated to one area of the list of phases, however since G1
    often spans the entire dataset in regards to angles, it may appear as dominant
    in small areas. The common behaviour is therefore one area where G1 is dominant for
    several points/angles in a row, and smaller G1 dominant areas.
    
    This function identifies the large area where G1 is dominant, and uses those
    angles to calculate it's mean angle.

    Parameters
    ----------
    my_adata : AnnData object
        The AnnData object to be filtered.
    list_of_phases : list
        A list of phases (G1,S,or G2M) indicating which phase is dominant at each angles.
    mean_dict : dictionnary
        A dictionnary containing the mean angles of S and G2M. G1 is added during this function

    Returns
    -------
    mean_dict : dictionnary
        A dictionnary containing the mean angles of G1, S, and G2M.

    """
    
    #Rolls the dataset to ensure that the G1 areas don't overlap with the 0 point
    phase_roll_for_ori=np.where(np.asarray(list_of_phases)!='G1')[0][0]
    ori_phase_list=np.roll(list_of_phases,-phase_roll_for_ori)
    
    #Initiate usefull parameters
    G1_length_dict={}
    my_inc=0
    start_idx=None
    
    #Find location and length of G1 dominant areas
    for idx,phase in enumerate(ori_phase_list):
        if phase=='G1':
            if start_idx==None:
                start_idx=idx
            my_inc+=1
            if idx==len(ori_phase_list)-1: #Reached the end on a G1
                G1_length_dict[start_idx]=my_inc
        elif phase!='G1' and start_idx!=None:
            G1_length_dict[start_idx]=my_inc
            start_idx=None
            my_inc=0
            
    #Sort the dictionnary to make the longest G1 area appear at index 0, then extract start location
    sorted_G1_length=sorted(G1_length_dict.items(),key=lambda x: x[1],reverse=True)
    index_G1=sorted_G1_length[0][0]
    
    #Adjust location based on roll of the dataset
    index_G1=index_G1+phase_roll_for_ori
    if index_G1>len(list_of_phases):
        index_G1=len(list_of_phases)-index_G1
    
    #Extract end point of G1
    index_G1_end=index_G1+sorted_G1_length[0][1]+1
    #Get subset
    g1_angle_subset=my_adata.obs['angles'][index_G1:index_G1_end]
    #Calculate mean using the subset
    g1_angle_mean=g1_angle_subset.mean()
    
    #Add new mean to the dictionnary
    mean_dict['G1']=g1_angle_mean
    
    return mean_dict


def find_orientation_from_mean_angles(mean_dict):
    """
    Function which determines the orientation of the data using the mean angle
    of the phases.
    
    The values are ordered in ascending values, the order of the phases is then checked
    and the appropriate orientation is given.
    
    Parameters
    ----------
    mean_dict : dictionnary
        dictionnary containing the mean angle for G1, S, and G2M.

    Returns
    -------
    orientation string
        Either G1 or G2M to indicate the order of the phases in the dataset.

    """
    ordered_dict={k: v for k, v in sorted(mean_dict.items(), key=lambda item: item[1])}
    phase_order=list(ordered_dict.keys())
    
    if (phase_order[0]=='G1' and phase_order[1]=='S') or (phase_order[0]=='S' and phase_order[1]=='G2M') or (phase_order[0]=='G2M' and phase_order[1]=='G1'):
        orientation='G1'
    elif (phase_order[0]=='G1' and phase_order[1]=='G2M') or (phase_order[0]=='G2M' and phase_order[1]=='S') or (phase_order[0]=='S' and phase_order[1]=='G1'):
        orientation='G2M'
    else:#Should never enter this area
        print('cannot find orientation with norm dict')
        print(ordered_dict)
        return False
    
    return(orientation)

def find_S_G2M_crossovers(list_s_g2m,P1,P2):
    """
    Function which finds the two S-G2M crossovers in the dataset and determines
    which one is the 'real' crossover in the dataset, and which one marks the expected
    location of G1. Respectively called 'roll_cross' and 'target_cross'.

    Parameters
    ----------
    list_s_g2m : list
        A list of either S or G2M which defines the locations in which they are dominant.
    P1 : string
        phase 1 - should be S if orientation is G1, should be G2M if orientation is G2M.
    P2 : string
        phase 2 - should be G2M if orientation is G1, should be S if orientation is G2M.

    Returns
    -------
    S_G2M_crosses: dictionnary
        A dictionnary containing the location of the two crossovers. The keys of 
        the dictionnary define the crossover type.

    """
    #Find a value to roll the data with ensuring that a cross over is not located
    #At the 0 point
    my_inc=0
    for idx,val in enumerate(list_s_g2m):
        if idx==len(list_s_g2m)-1:
            check_idx=0
        else:
            check_idx=idx+1
        if val==list_s_g2m[check_idx]:
            my_inc+=1
            if my_inc==3:
                roll_val=idx
                break
        else:
            my_inc=0
    
    #Roll the data
    list_s_g2m=list(np.roll(list_s_g2m,-roll_val))
    
    #Find the crosses
    S_G2M_crosses={}
    for idx,phase in enumerate(list_s_g2m):
        if idx==len(list_s_g2m)-1:
            check_idx=0
        else:
            check_idx=idx+1
        if phase==P1 and list_s_g2m[check_idx]==P2:
            S_G2M_crosses['roll_cross']=idx
        elif phase==P2 and list_s_g2m[check_idx]==P1:
            S_G2M_crosses['target_cross']=idx
        
    
    #Adjust found values due to the roll
    for key in S_G2M_crosses.keys():
        S_G2M_crosses[key]=S_G2M_crosses[key]+roll_val
        if S_G2M_crosses[key]>len(list_s_g2m):
            S_G2M_crosses[key]=S_G2M_crosses[key]-len(list_s_g2m)
    
    return(S_G2M_crosses)


def find_G1_crossover(list_of_phases,orientation,S_G2M_crosses,look_for):
    """
    Function which finds the two G1 crossovers between S and G2M
    
    The two crossovers use the S_G2M 'target_cross' as the starting point.
    
    The crossovers are found via two ways. Starting from the 'target_cross', the
    crossover with the phase 'before' G1 is identified by taking a slice between the start
    point (0) and the 'target_cross', then reversing the list and finding the index where it
    changes phases. The phase 'after' is found by taking a slive between the 'target_cross'
    and the end of the dataset, while stopping the search once the phase is no longer G1.

    Parameters
    ----------
    list_of_phases : list
        A list of phases (G1,S,or G2M) indicating which phase is dominant at each angles.
    orientation : String
        Either G1 or G2M to indicate the sequence of phases in the dataset.
    S_G2M_crosses : dictionnary
        Contains the two crossovers of S and G2M as well as what they represent.
    look_for : string
        Either S or G2M, represents the crossover with G1 that will be searched for.

    Returns
    -------
    found_idx : int
        The index where the crossover between G1 and 'look_for' occurs.

    """
    
    roll_val=S_G2M_crosses['roll_cross']
    
    #Roll the list to the correct S_G2M crossover, ensures that G1 crosses will
    #not overlap the 0 point
    list_of_phases=np.roll(list_of_phases,-roll_val)
    #Adjust location of target crossover based on the roll
    target_cross=S_G2M_crosses['target_cross']
    target_cross=target_cross-roll_val
    if target_cross<0:
        target_cross=len(list_of_phases)+target_cross
        
    #Determines if reversal is necessary
    if orientation=='G1' and look_for=='S':
        reversal=False
    elif orientation=='G1' and look_for=='G2M':
        reversal=True
    elif orientation=='G2M' and look_for=='S':
        reversal=True
    elif orientation=='G2M' and look_for=='G2M':
        reversal=False
    
    #Establishes the indexes for the slices used on list_of_phases
    if reversal==True:
        start_idx=0
        end_idx=target_cross
    else:
        start_idx=target_cross
        end_idx=None
    
    #Creates the slice to find the desired crossover
    search_range=list(list_of_phases[start_idx:end_idx])
    if reversal==True:
        search_range.reverse()
    
    #Iterates over the slice to find the crossover
    for idx,p in enumerate(search_range):
        if idx==len(search_range)-1:
            check_idx=0
        else:
            check_idx=idx+1
        if p=='G1' and search_range[check_idx]==look_for:
            found_idx=idx
            break
        
    #Correct for the slice made
    found_idx=found_idx+start_idx
    #Correct for the reversal
    if reversal==True:
        found_idx=(len(search_range)-1)-idx
        #This step is done since the pipeline expects the index crossover to be 
        #The last index of the phase before crossing over. If reversal occured,
        #Then the current index would be the first G1 and not the last of the phase
        #that comes before it.
        found_idx=found_idx-1
        
    #Correct for roll
    found_idx=found_idx+roll_val
    if found_idx>(len(list_of_phases)-1):
        found_idx=found_idx-(len(list_of_phases)-1)

    return found_idx


    
    
def create_crossover_dict(my_adata,found_idx_P1,found_idx_P2,S_G2M_crosses,orientation):
    """
    Simple function which loads the results in a dictionnary which suits the format
    of the code downstream of this sequence of functions

    Parameters
    ----------
    my_adata : AnnData object
        The AnnData object to be filtered.
    found_idx_P1 : int
        The index for phase 1-G1 crossover.
    found_idx_P2 : int
        The index for phase 2 -G1 crossover.
    S_G2M_crosses : dictionnary
        Contains the two S_G2M crossovers.
    orientation : string
        Either G1 or G2M to indicate the sequence of phases in the data.

    Returns
    -------
    dict_crossover : dictionnary
        A dictionnary containing the crossover results.

    """
    dict_crossover={}
    if orientation=='G1':
        dict_crossover['G1_G2M']=found_idx_P2
        dict_crossover['S_G1']=found_idx_P1
    else:
        dict_crossover['G1_G2M']=found_idx_P1
        dict_crossover['S_G1']=found_idx_P2
    dict_crossover['S_G2M']=S_G2M_crosses['roll_cross']
    
    #Adjusts for the zero index factor of python.
    for key,val in dict_crossover.items():
        if val==0:
            dict_crossover[key]=len(my_adata.obs)-1
        else:
            dict_crossover[key]=val-1
    
    return dict_crossover

#%% Plotting

def scanpy_pp_plots(
    adata,
    MT_list,
    std_plots,
    MT_plots,
    unspliced_plots,
    path,
    sub_folder):
    """
    Creation of some usefull pre-processing plots using scanpy functions

    Parameters
    ----------
    adata : AnnData object
        The AnnData object which will serve as the data for the plots.
    MT_list : list
        list containing mitochondrial genes.
    std_plots : Boolean
        Standard violin plots for n_genes, n_counts and a scatter plot  using both.
    MT_plots : Boolean
        violin plot for percent MT and scatter using percent MT and n_counts.
    unspliced_plots : Boolean
        violin plot for unspliceed and scatter using unspliced and n_counts.
    path : String
        path in which the pdf will be saved
    sub_folder : String
        name of the subfolder
    

    Returns
    -------
    None.

    """    
    #Create standard plots
    adata.obs['n_counts'] = adata.X.sum(axis=1).A1
    
    original_path=os.getcwd()
    os.chdir(path)
    # fig, axs = plt.subplots(7)
    if std_plots is True:
        sc.pl.violin(adata,['n_genes'], jitter=0.4,show=False,save="_plots/"+sub_folder+"/n_genes.pdf")
        sc.pl.violin(adata,['n_counts'], jitter=0.4,show=False,save="_plots/"+sub_folder+"/n_counts.pdf")
        sc.pl.scatter(adata,x='n_counts',y='n_genes',show=False,save="_plots/"+sub_folder+"/n_counts_n_genes.pdf")
    
    if MT_plots is True: #Create percent_MT plots
    
        MT_bool_list=[] #Bool list of MT_list in adata
        for val in adata.var_names:
            if val in MT_list:
                MT_bool_list.append(True)
            else:
                MT_bool_list.append(False)        
        adata.obs["percent_MT"] = np.sum(adata[:,MT_bool_list].X,axis=1).A1/np.sum(adata.X,axis=1).A1
        
        sc.pl.violin(adata,['percent_MT'], jitter=0.4,show=False,save="_plots/"+sub_folder+"/percent_MT.pdf")
        sc.pl.violin(adata,['percent_MT'], jitter=0.4,show=False,save="_plots/"+sub_folder+"/percent_MT.png")
        sc.pl.scatter(adata,x='n_counts',y='percent_MT',show=False,save="_plots/"+sub_folder+"/n_counts_percent_MT.pdf")
        
    if unspliced_plots is True:
        adata.obs["unspliced"] = np.sum(adata.layers['unspliced'],axis=1).A1/np.sum(adata.X,axis=1).A1
        sc.pl.violin(adata,['unspliced'], jitter=0.4,show=False,save="_plots/"+sub_folder+"/unspliced.pdf")
        sc.pl.violin(adata,['unspliced'], jitter=0.4,show=False,save="_plots/"+sub_folder+"/unspliced.png")
        sc.pl.scatter(adata,x='n_counts',y='unspliced',show=False,save="_plots/"+sub_folder+"/n_counts_unspliced.pdf")

    
    os.chdir(original_path)
    
    
    
def perform_scanpy_pca(adata,compute,exclude_gene_counts,exclude_CC,save_path,sub_folder):
    """
    Wrapper function for the scanpy PCA function. Used to create a diversity
    of PCAs.
    Mainly created for code legibility
    
    Function written by Geir Armund Svan Hasle

    Parameters
    ----------
    adata : AnnData
        The Anndata object.
    compute : boolean
        Wether PCA computation is to be done.
    exclude_gene_counts : boolean
        If n_genes and n_counts is to be plotted.
    exclude_CC : boolean
        If cell_cycle components (phase scores and phases) should be plotted.
    save_path : String
        path to which the PCAs will be saved
    sub_folder : String
        name of the sub folder

    Returns
    -------
    None.

    """
    original_path=os.getcwd()
    os.chdir(save_path)   
    
    if compute==True:
        sc.tl.pca(adata,svd_solver='arpack')
    if exclude_gene_counts==False:
        sc.pl.pca(adata,color=['n_genes'],show=False, save="_bar_line_plots/"+sub_folder+"/n_genes")
        sc.pl.pca(adata,color=['n_counts'],show=False, save="_bar_line_plots/"+sub_folder+"/n_counts")
    if exclude_CC==False:
        sc.pl.pca(adata,color=['G1_score'],show=False, save="_bar_line_plots/"+sub_folder+"/G1_score")
        sc.pl.pca(adata,color=['S_score'],show=False, save="_bar_line_plots/"+sub_folder+"/S_score")
        sc.pl.pca(adata,color=['G2M_score'],show=False, save="_bar_line_plots/"+sub_folder+"/G2M_score")
        sc.pl.pca(adata,color=['phase'],show=False, save="_bar_line_plots/"+sub_folder+"/phase")
        
    os.chdir(original_path)   
    
    
    
def plot_norm_dist(plot_dict,x_axis,save_file):
    """
    Small plotting function to be used in conjunction with calculate_norm_dist()
    This function will plot the calculated normal distribution of the three phases.
    
    Function written by Yohan lefol
    
    Parameters
    ----------
    plot_dict : dictionnary
        A dictionnary containing the calculated normal distances for the phases.
        Calculations are done by calculate_norm_dist()
    x_axis : numpy array
        The x axis used in the plot.
    save_file : string
        The location and name (with extension) of the saved plot.

    Returns
    -------
    None.

    """
    colors_dict = {'G1':np.array([52, 127, 184]),'S':np.array([37,139,72]),'G2M':np.array([223,127,49]),}
    colors_dict = {k:v/256 for k, v in colors_dict.items()}
    plt.figure(None, (10,5.5), dpi=80)
    for phase in plot_dict:
        plt.plot(x_axis,plot_dict[phase],color=colors_dict[phase],label=phase)
    plt.legend()
    plt.savefig(save_file)
    # plt.show()
 
