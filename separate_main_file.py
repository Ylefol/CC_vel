#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 13:10:47 2025

@author: yohanl
"""

import scanpy as sc

adata = sc.read_h5ad('all_samples_filtered.h5ad')

adata.obs['CellID']=adata.obs_names
adata.obs['Sample_ID_old']=adata.obs['Sample_ID'].copy()
adata.obs['Sample_ID']=adata.obs_names


adata.obs = adata.obs.drop(columns=['starsolo_barcodes','cbMatch','cbPerfect','cbMMunique', 
                                    'cbMMmultiple', 'genomeU', 'genomeM', 'featureU', 'featureM', 
                                    'exonic', 'intronic', 'exonicAS', 'intronicAS', 'mito', 
                                    'countedU', 'countedM', 'nUMIunique', 'nGenesUnique', 
                                    'nUMImulti', 'nGenesMulti', 'sample_id', 'sublib', 
                                    'stype', 'bc1_well', 'bc2_well', 'bc3_well', 'parsebio_bc', 
                                    'library_id', 'library_idx', 'Demux_Cell_Concentration_Countess_Iii_Cells_Per_Ul', 
                                    'Demux_Target_Number_Barcoded_Cells', 'droplet_type', 'nuclear_fraction'])




adata.var_names=list(adata.var['gene_symbols'])
adata.var["Gene"] = adata.var_names


# import velocyto as vcy
# template=sc.read_loom('CC_vel/data_files/phase_reassigned/AG_HaCat_A.loom')
# vlm = vcy.VelocytoLoom('CC_vel/data_files/phase_reassigned/AG_HaCat_A.loom')
# vlm_bis=vcy.VelocytoLoom('CC_vel/data_files/phase_reassigned/AG_HaCat-Control_A.loom')

# for i in adata.obs.columns:
#     print(adata.obs[i])


sub_adata=adata[(adata.obs['Demux_Cell_Type']=='HaCat cells')&(adata.obs['Sample_Group']=='CRISPRi AGO2')]
sub_adata.write_loom('CC_vel/data_files/initial_data/loom_files/HaCat-CRISPRi-AGO2_A.loom', write_obsm_varm=True)

sub_adata=adata[(adata.obs['Demux_Cell_Type']=='HaCat cells')&(adata.obs['Sample_Group']=='CRISPRi Dicer')]
sub_adata.write_loom('CC_vel/data_files/initial_data/loom_files/HaCat-CRISPRi-Dicer_A.loom', write_obsm_varm=True)

sub_adata=adata[(adata.obs['Demux_Cell_Type']=='HaCat cells')&(adata.obs['Sample_Group']=='Control')]
sub_adata.write_loom('CC_vel/data_files/initial_data/loom_files/HaCat-Control_A.loom', write_obsm_varm=True)


sub_adata=adata[(adata.obs['Demux_Cell_Type']=='A549 cells')&(adata.obs['Sample_Group']=='CRISPRi AGO2')]
sub_adata.write_loom('CC_vel/data_files/initial_data/loom_files/A549-CRISPRi-AGO2_A.loom', write_obsm_varm=True)

sub_adata=adata[(adata.obs['Demux_Cell_Type']=='A549 cells')&(adata.obs['Sample_Group']=='CRISPRi Dicer')]
sub_adata.write_loom('CC_vel/data_files/initial_data/loom_files/A549-CRISPRi-Dicer_A.loom', write_obsm_varm=True)

sub_adata=adata[(adata.obs['Demux_Cell_Type']=='A549 cells')&(adata.obs['Sample_Group']=='Control')]
sub_adata.write_loom('CC_vel/data_files/initial_data/loom_files/A549-Control_A.loom', write_obsm_varm=True)







sub_adata.write_loom('CC_vel/data_files/initial_data/loom_files/test_A.loom', write_obsm_varm=True)



temp=sc.read_loom('CC_vel/data_files/initial_data/loom_files/test_A.loom',X_name="")
