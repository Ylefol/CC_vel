#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 13:10:47 2025

@author: yohanl
"""

import scanpy as sc
import numpy as np
import os

os.chdir('/media/yohanlefol/Expansion/CC_vel/')

adata = sc.read_h5ad('all_samples_filtered.h5ad')

adata.obs['CellID']=adata.obs_names
adata.obs['Sample_ID_old']=adata.obs['Sample_ID'].copy()
adata.obs['Sample_ID']=adata.obs_names


adata.obs = adata.obs.drop(columns=['starsolo_barcodes','cbMatch','cbPerfect','cbMMunique', 
                                    'cbMMmultiple', 'genomeU', 'genomeM', 'featureU', 'featureM', 
                                    'exonic', 'intronic', 'exonicAS', 'intronicAS', 'mito', 
                                    'countedU', 'countedM', 'nUMIunique', 'nGenesUnique', 
                                    'nUMImulti', 'nGenesMulti', 'sublib', 
                                    'stype', 'bc1_well', 'bc2_well', 'bc3_well', 'parsebio_bc', 
                                    'library_id', 'library_idx', 'Demux_Cell_Concentration_Countess_Iii_Cells_Per_Ul', 
                                    'Demux_Target_Number_Barcoded_Cells', 'droplet_type', 'nuclear_fraction'])




adata.var_names=list(adata.var['gene_symbols'])
adata.var["Gene"] = adata.var_names

sc.pp.normalize_total(adata,target_sum=1e4)
sc.pp.log1p(adata)

sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# sc.pl.umap(adata=adata,color=['Sample_ID_old','Demux_Cell_Type','Sample_Group'],ncols=1)

#Splits needed
# 1 Demux_Cell_Type -- HaCat cells OR A549
# 2 Sample_Group -- CRISPRi AGO2 OR CRISPRi Dicer OR Control
# 3 Sample_ID_old -- 1 OR 2 (A549) 3 OR 4 (HaCat)
# 4 Expression of DICER1 OR AGO2
import re
for cell_type in ['HaCat cells','A549 cells']:
    if cell_type=='HaCat cells':
        replicates=['3','4']
        cell_name='HaCat'
    elif cell_type=='A549 cells':
        replicates=['1','2']
        cell_name='A549'
    for samp_group in ['CRISPRi AGO2', 'CRISPRi Dicer', 'Control']:
        for rep in replicates:
            if rep=='1' or rep=='3':
                rep_name='A'
            elif rep=='2' or rep=='4':
                rep_name='B'
            sub_adata=adata[adata.obs['Demux_Cell_Type']==cell_type]#celltype
            sub_adata=sub_adata[sub_adata.obs['Sample_Group']==samp_group]#group
            sub_adata=sub_adata[sub_adata.obs['Sample_ID_old'].str.startswith(rep)]#replicate

            if samp_group=='Control':#No split on gene expression
                save_name=cell_name+'-'+re.sub(" ", "-", samp_group)+'_'+rep_name
                for col in sub_adata.obs.columns:
                    if sub_adata.obs[col].dtype.name == 'category':
                        sub_adata.obs[col] = sub_adata.obs[col].astype(str)
                for col in sub_adata.var.columns:
                    if sub_adata.var[col].dtype.name == 'category':
                        sub_adata.var[col] = sub_adata.var[col].astype(str)
                sub_adata.write_loom('data_files/initial_data/loom_files/'+save_name+'.loom', write_obsm_varm=True)
                continue #Skip gene segment

            sub_adata_base = sub_adata  # preserve base before gene/exp filtering
            if samp_group == 'CRISPRi AGO2':
                target_gene='AGO2'
            elif samp_group == 'CRISPRi Dicer':
                target_gene='DICER1'

            for exp in ['KO','EXP']:
                if exp=='KO':
                    gene_adata = sub_adata_base[sub_adata_base[:, target_gene].X ==0] #KO
                elif exp=='EXP':
                    gene_adata = sub_adata_base[sub_adata_base[:, target_gene].X >0] #Expressed

                save_name=cell_name+'-'+re.sub(" ", "-", samp_group)+'-'+exp+'_'+rep_name

                for col in gene_adata.obs.columns:
                    if gene_adata.obs[col].dtype.name == 'category':
                        gene_adata.obs[col] = gene_adata.obs[col].astype(str)
                for col in gene_adata.var.columns:
                    if gene_adata.var[col].dtype.name == 'category':
                        gene_adata.var[col] = gene_adata.var[col].astype(str)

                gene_adata.write_loom('data_files/initial_data/loom_files/'+save_name+'.loom', write_obsm_varm=True)
        
    
    


 