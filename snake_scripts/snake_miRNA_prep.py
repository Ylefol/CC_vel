#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 14:10:59 2020

@author: yohan
"""

from snake_functions import snake_miRNA_prep_functions as my_func
import sys

#Load list, rpm normalize
matrix_path=sys.argv[1]

rpm_df=my_func.rpm_normalization(matrix_path)

#Create TS_file and subset as needed
TS_path='data_files/miRNA_files/TS_files/Predicted_Targets_Context_Scores.default_predictions.txt'

thresholds=sys.argv[2].split('/')[-1]
thresholds=thresholds.split('.')[0]
min_threshold_value=int(thresholds.split('_')[2])
max_threshold_value=thresholds.split('_')[3]

if max_threshold_value=='None':
    max_threshold_value=None
else:
    max_threshold_value=int(max_threshold_value)

#First categorization (found, not found, accounted for etc...)
miRNA_dict=my_func.categorize_findings(TS_path, rpm_df, min_miRNA_thresh=min_threshold_value, max_miRNA_thresh=max_threshold_value)

#Load up miRNA family path
miR_family='data_files/miRNA_files/TS_files/miRNA_TS_family.csv'

#Checks unaccounted for miRNA using miRNA family file, this uncovers subset versions
miRNA_dict=my_func.check_miRNA_family(miRNA_dict,miR_family,TS_path)

#Sets non_conserved file path
non_conserved_path='data_files/miRNA_files/TS_files/Nonconserved_Site_Context_Scores.txt'

#Checks if the remaining unaccounted for miRNA are non_conserved and return final df
my_df=my_func.check_non_conserved(miRNA_dict,non_conserved_path)


#Save final df
my_df.to_csv(sys.argv[2],index=False)