#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:01:29 2024

@author: yohanl
"""
import scvelo as scv
adata= scv.read('data_files/phase_reassigned/CC_HaCat_B.loom')

import matplotlib as plt
plt.rcParams['figure.dpi'] = 1200

#Perform downsampling
import numpy as np
tosample = int(np.ceil(np.mean(np.unique(adata.obs["phase"], return_counts=1)[1])))

downsample_ixs = []
for ct in np.unique(adata.obs["phase"]):
    ixs = np.where(adata.obs["phase"] == ct)[0]
    downsample_ixs.append(np.random.choice(ixs, min(tosample, len(ixs)), replace=False))
downsample_ixs = np.concatenate(downsample_ixs)


bool_array=np.in1d(np.arange(adata.layers['spliced'].shape[0]), downsample_ixs)

#Gotta filter it here in my own way 
adata_subset = adata[bool_array, :]


scv.pp.filter_and_normalize(adata_subset, n_top_genes=len(adata.var))
scv.pp.moments(adata_subset, n_pcs=2, n_neighbors=550)

scv.tl.recover_dynamics(adata_subset)

scv.tl.velocity(adata_subset, mode='dynamical')
scv.tl.velocity_graph(adata_subset)

scv.pl.velocity_embedding_stream(adata_subset, basis='pca',color='phase')


scv.tl.latent_time(adata_subset)
scv.pl.scatter(adata_subset, color='latent_time', color_map='gnuplot', size=80)


# top_genes = adata_subset.var['fit_likelihood'].sort_values(ascending=False).index
# scv.pl.scatter(adata_subset, basis=top_genes[:15], ncols=5, frameon=False,color='phase')
