#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 07:07:28 2025

@author: yohanl
"""

import scanpy as sc

adata=sc.read_h5ad('with_transitions.h5ad')



adata.obs['cell_cycle_theta']


sc.pl.pca(adata,color='cell_cycle_theta')
adata = adata[adata.obs.sort_values("cell_cycle_theta").index]

# Verify sorting
print(adata.obs["cell_cycle_theta"].head())  # Check if sorting is correct

import numpy as np

#G1/S transition
G1_S_transition=adata.obs.index[adata.obs['CyclinE_score']==np.max(adata.obs['CyclinE_score'])][0]

#S/G2 transition
condition = adata[G1_S_transition:].obs["G2M_score"] > adata[G1_S_transition:].obs["S_score"]
S_G2_transition = condition.idxmax() if condition.any() else None

#Does work so just label 0 as first G1
adata.obs["phase"] = ""
adata.obs.loc[:G1_S_transition, "phase"] = "G1"
adata.obs.loc[G1_S_transition:S_G2_transition, "phase"] = "S" 
adata.obs.loc[S_G2_transition:, "phase"] = "G2M" 