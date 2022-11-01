#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 12:23:58 2020

@author: yohan
"""

import numpy as np
import velocyto as vcy
from snake_functions import snake_velocyto_functions as my_func
import sys

#Load file
loom_path=sys.argv[1]

replicate=(list(loom_path)[-6])
cell_line_var=(loom_path.split('_')[-2])
# cell_line_var=cell_line_var.split('/')[1]

for loop_number in range(int(sys.argv[2])):
    np.random.seed() #Random seed at each iteration
    vlm = vcy.VelocytoLoom(loom_path)

    # Read column attributes form the loom file and specify colors, set vlm.colorandum
    colors_dict = {'G1':np.array([52, 127, 184]),
      'S':np.array([37,139,72]),
      'G2M':np.array([223,127,49]),}
    colors_dict = {k:v/256 for k, v in colors_dict.items()}
    vlm.set_clusters(vlm.ca["phase"], cluster_colors_dict=colors_dict)
    
    #Downsampling cells - equalize levels of each type/phase
    tosample = int(np.ceil(np.mean(np.unique(vlm.ca["phase"], return_counts=1)[1])))
    downsample_ixs = []
    for ct in np.unique(vlm.ca["phase"]):
        ixs = np.where(vlm.ca["phase"] == ct)[0]
        downsample_ixs.append(np.random.choice(ixs, min(tosample, len(ixs)), replace=False))
    downsample_ixs = np.concatenate(downsample_ixs)
    vlm.filter_cells(bool_array=np.in1d(np.arange(vlm.S.shape[1]), downsample_ixs))  #Downsample filter
    
    #Assign the CellID and Gene variables as it is what is searched for by velocyto functions
    #Bit of code was needed preceding an update from Velocyto
    # vlm.ca["CellID"]=vlm.ca["obs_names"]
    # vlm.ra['Gene']=vlm.ra["var_names"]
    
    boundary_dict=my_func.create_boundary_dict(vlm)
    
    
    #filters specific layers based on specified quantites in dp (beginning of code)
    vlm.score_detection_levels(min_expr_counts=30, min_cells_express=10, min_expr_counts_U=30, min_cells_express_U=10)
    vlm.filter_genes(by_detection_levels=True, by_cluster_expression=False)
    
    
    #Normalize unspliced and spliced layer
    combined_size = (vlm.S.sum(0) / np.percentile(vlm.S.sum(0), 95)) + (vlm.initial_cell_size / np.percentile(vlm.initial_cell_size, 95))
    combined_Usize = (vlm.U.sum(0) / np.percentile(vlm.U.sum(0), 95)) + (vlm.initial_Ucell_size / np.percentile(vlm.initial_Ucell_size, 95))
    vlm._normalize_S(relative_size=0.25*combined_size*np.median(vlm.S.sum(0)),
                      target_size=np.median(vlm.S.sum(0)))
    vlm._normalize_U(relative_size=0.5*combined_Usize*np.median(vlm.U.sum(0)),
                      target_size=np.median(vlm.U.sum(0)))
    
    vlm.perform_PCA()
    
    
    k = int(sys.argv[3])
    
    vlm.knn_imputation(n_pca_dims=20,k=k, balanced=True,
                        b_sight=np.minimum(k*8, vlm.S.shape[1]-1),
                        b_maxl=np.minimum(k*4, vlm.S.shape[1]-1))
    vlm.normalize_median()
    vlm.fit_gammas(maxmin_perc=[2,95], limit_gamma=True)
    
    vlm.normalize(which="imputed", size=False, log=True)
    
    
    vlm.Pcs = np.array(vlm.pcs[:,:2], order="C")
    
    vlm.predict_U()
    vlm.calculate_velocity()
    vlm.calculate_shift()
    vlm.extrapolate_cell_at_t(delta_t=1)
    
    #Use correlation to estimate transition probabilities for every cells to its embedding neighborhood
    #hidim --> high dimensional space
    #embed --> Name of the attribute containing the embedding, here it is PCs, but it could be  ts
    #Transform --> Transformation that is applied on the hidim. Can be sqrt or log
    #psc --> pseudocount added in variance normalizing transform
    vlm.estimate_transition_prob(hidim="Sx_sz", embed="Pcs", transform="log", psc=1,
                                  n_neighbors=150, knn_random=True, sampled_fraction=1)
    
    
    #Use the transition probability to project the velocity direction on the embedding
    #sigma_corr --> the kernel scalling
    #expression scaling --> rescaled arrows to penalize arrows that explain very small amoung of expression differences
    vlm.calculate_embedding_shift(sigma_corr = 0.05, expression_scaling=False)
    #Expression scaling to False makes arrows larger and 'curvier' as no penalties implemented
    
    
    #Calculate the velocity using a point on a regular grid and a gaussian kernel
    #steps --> number of steps in the grid for each axis
    vlm.calculate_grid_arrows(smooth=0.5, steps=(25, 25), n_neighbors=150)
    
    window_val=int(0.20*len(vlm.ca['CellID']))
    
    
    spli_dict,unspli_dict=my_func.create_smooth_vels(vlm,window_size=window_val,return_dict=True)
    
    # my_func.create_CI_CSVs(spli_dict, cell_line=cell_line_var, replicate=replicate, layer='spliced',file_name=str(loop_number+1))
    # my_func.create_CI_CSVs(unspli_dict, cell_line=cell_line_var, replicate=replicate, layer='unspliced',file_name=str(loop_number+1))
    
    my_func.save_smooth_vels(vlm, spli_dict, cell_line=cell_line_var, replicate=replicate, layer='spliced',file_name=str(loop_number+1))
    my_func.save_smooth_vels(vlm, unspli_dict, cell_line=cell_line_var, replicate=replicate, layer='unspliced',file_name=str(loop_number+1))
    
    
    my_func.save_vlm_values(vlm,cell_line=cell_line_var, replicate=replicate, layer='spliced',file_name=str(loop_number+1))
    my_func.save_vlm_values(vlm,cell_line=cell_line_var, replicate=replicate, layer='unspliced',file_name=str(loop_number+1))
    
    

