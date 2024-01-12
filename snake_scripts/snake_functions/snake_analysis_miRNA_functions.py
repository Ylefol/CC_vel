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

import matplotlib.pyplot as plt


#%% Processing and sorting

def find_three_UTR_lengths(gtf_path,gene_list,output_path=None,get_df=True):
    """
    Function which utilizes the pyranges package and a gtf file to extract the 
    3'UTR length for the provided gene list.
    
    The gtf file may contain several transcripts per gene, and therefore several
    UTRs for a single gene. The function calcualtes each UTR length for each transcript 
    and preserve the largest UTR as that genes 3'UTR length
    
    Function written by Yohan Lefol

    Parameters
    ----------
    gtf_path : string
        The path to the gtf file to be used.
    gene_list : list of strings
        list of gene names for which the 3'UTR length will be calculated.
    output_path : string, optional
        A save path. The default is None.
    get_df : boolean, optional
        If the dataframe should be returned or not. The default is True.

    Returns
    -------
    my_df : pandas dataframe
        A dataframe containing each gene name along with the calcualted 3'UTR
        as well as some extra gtf related information such as teh transcript for
        which the 3'UTR length was calculated.

    """
    #Read the gtf file using pyranges
    gr=pr.read_gtf(gtf_path)
    
    #Manueally establish columns of iportance
    col_list=['gene_name','transcript_name','transcript_id','Source','Feature','Chromosome','Strand','Start','End','Length']
    my_dict={}
    for col in col_list:
        my_dict[col]=[]

    #Keep only essential elements within the gtf file (this is critical for computation speed)
    gr=gr[gr.gene_name.isin(gene_list)]
    gr=gr[col_list]
    gr=gr[gr.Feature.isin(['UTR','transcript'])]

    #Iterate over each gene
    for gene in gene_list:
        gtf_data=gr[gr.gene_name==gene]
        if len(gtf_data)>0:
            #Iterate over each transcript
            for tr_name in gtf_data.transcript_name.unique():
                if isinstance(tr_name,float)==False:
                    subset_transcript=gtf_data[gtf_data.transcript_name==tr_name]
                    subset_UTR=subset_transcript[subset_transcript.Feature=='UTR']
                    if len(subset_UTR)>0:
                        
                        #Several UTRs can be found, if that is the case, we find the correct UTR by using strand along with 
                        #Checking for the according start or end position (using the strand)
                        if subset_transcript.Strand.values[0]=='+':
                            transcript_val=subset_transcript.End[subset_transcript.Feature=='transcript']
                            final_sub=subset_UTR[subset_UTR.End==transcript_val.values[0]]
                        else:
                            transcript_val=subset_transcript.Start[subset_transcript.Feature=='transcript']
                            final_sub=subset_UTR[subset_UTR.Start==transcript_val.values[0]]
                        
                        if len(final_sub)>0:
                            for key in my_dict.keys():
                                if key == 'Feature':
                                    my_dict[key].append("3'UTR")
                                elif key == 'Length':
                                    my_dict[key].append(abs(final_sub.Start.values[0]-final_sub.End.values[0]))
                                else:
                                    my_dict[key].append(final_sub.values()[0:1][0][key].values[0])
    
    #Convert to pandas dataframe
    my_df=pd.DataFrame(my_dict,columns=list(my_dict.keys()))
    
    #Create final dataframe for either saving or returning
    big_boy_list=[]
    for gene in my_df.gene_name.unique():
        subset=my_df[my_df.gene_name==gene]
        big_boy_list.append(subset[subset.Length==np.max(subset.Length)].transcript_id.values[0])
        
    my_df=my_df[my_df.transcript_id.isin(big_boy_list)]
    
    if output_path != None:
        my_df.to_csv(output_path,index=False)
    if get_df==True:
        return my_df
    

def create_read_UTR_results(cell_line,target_rep,gtf_path,gene_list):
    """
    A small function which checks if UTR lengths already exists, if that is the case
    it reads the file instead of running the creating function again
    
    Function written by Yohan Lefol

    Parameters
    ----------
    cell_line : string
        The name of the cell line being used.
    target_rep : string
        The single or merged replicates that are being used
    gtf_path : string
        The path to the gtf file.
    gene_list : list
        list of strings (genes) indicating the genes for which the 3'UTR needs
        to be calculated.

    Returns
    -------
    UTR_df : pandas dataframe
        The calculated 3'UTR lengths along with some extra gtf related information.

    """
    UTR_file_name=target_rep+'_UTR_length.csv'
    if os.path.isdir('data_files/data_results/UTR_length/'+cell_line)==False:
        my_utils.create_folder('data_files/data_results/UTR_length/'+cell_line)
        
    if UTR_file_name in os.listdir('data_files/data_results/UTR_length/'+cell_line):
        UTR_df=pd.read_csv('data_files/data_results/UTR_length/'+cell_line+'/'+UTR_file_name)
    else:
        find_three_UTR_lengths(gtf_path,gene_list,output_path='data_files/data_results/UTR_length/'+UTR_file_name)
        UTR_df=pd.read_csv('data_files/data_results/UTR_length/'+UTR_file_name)
        
    
    return UTR_df


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


def target_scan_analysis(TS_path,gene_dict,cell_line,miR_thresh,miRNA_list=None):
    """
    Function that uses a targetscan prediction file and creates three dictionnaries
    based on the number of target sites observed, the weight context score of each gene
    and the sum of weighted context for each gene.
    The function automatically filters for miRNAs which are specific to humans, 
    the function can also filter based on a set list of miRNAs.
    
    Function written by Yohan Lefol

    Parameters
    ----------
    TS_path : string
        The file patht to the TargetScan prediction file.
    gene_dict : dictionnary
        A dictionnary containing categoris and gene names.
    cell_line : string
        A string indicating the cell_line used
    miR_thresh : string
        String indicating the threshold used for miRNAs
    miRNA_list : list, optional
        A list of miRNAs that will specify which miRNA target sites will be
        searched for in the TargetScan file. The default is None.

    Returns
    -------
    output_dict : dictionnary
        A dictionnary with the number of target sites per gene.
    output_weight_dict : dictionnary
        A dictionnary with all the weighted context score for each gene.
    sum_weight_dict : dictionnary
        A dictionnary with the sum of the weighted context score for each gene.
    df_gene_miR : pandas dataframe
        Dataframe containing the gene list with associated miRNAs
    text_res : string
        String with the text results of the analysis

    """
    
    TargetScan_file=pd.read_csv(TS_path,sep='\t')
    
    TargetScan_file=TargetScan_file[TargetScan_file['Gene Tax ID']==9606]
    
    if miRNA_list is not None:
        TargetScan_file=TargetScan_file[TargetScan_file['miRNA'].isin(miRNA_list)]

    TargetScan_file=TargetScan_file[TargetScan_file['Gene Symbol'].isin(gene_dict[list(gene_dict.keys())[0]])]
    df_gene_miR=TargetScan_file[['miRNA','Gene Symbol']]
    df_gene_miR=df_gene_miR.set_index(np.arange(0,len(df_gene_miR.index)))

    output_dict={}
    output_weight_dict={}
    sum_weight_dict={}
    for key in gene_dict.keys():
        gene_list=gene_dict[key]
        output_dict[str(key)]={}
        output_weight_dict[str(key)+'_weighted']={}
        sum_weight_dict[key]=[]
        list_no_miRNA=[]
        total_num=0
        total_good_num=0
        for gene in gene_list:
            num_found=len(np.where(TargetScan_file['Gene Symbol']==gene)[0])
            if num_found==0:
                list_no_miRNA.append(gene)
            else:
                total_num=total_num+num_found
                
            if num_found in output_dict[str(key)]:#Key exists
                output_dict[str(key)][num_found].append(gene)
            else:#create key and add gene
                output_dict[str(key)][num_found]=[gene]
            
            subset=TargetScan_file[TargetScan_file['Gene Symbol']==gene]
            sum_weight=sum(subset['weighted context++ score'])
            sum_weight_dict[key].append(sum_weight)
            
            good_targets=len(np.where(subset['weighted context++ score']<=-0.3)[0])
            
            if good_targets in output_weight_dict[str(key)+'_weighted']:#Key existsts
                output_weight_dict[str(key)+'_weighted'][good_targets].append(gene)
            else:#create key and add gene
                output_weight_dict[str(key)+'_weighted'][good_targets]=[gene]
            
            total_good_num=total_good_num+good_targets

    # print("For this dict: "+str(key))
    elem_1='Information for '+cell_line+' '+miR_thresh+'\n'
    elem_2='Total number of genes: '+str(len(gene_list))+'\n'
    elem_3='Number of genes that have targets: '+str(abs(len(gene_list)-len(list_no_miRNA)))+'\n'
    elem_4='Number of genes that have no targets: '+str(len(list_no_miRNA))+'\n'
    elem_5='Total number of targets found for the group: '+str(total_num)+'\n'
    elem_6='Total number of targets with a weighted context <-0.3: '+str(total_good_num)+'\n'
    
    text_res=elem_1+elem_2+elem_3+elem_4+elem_5+elem_6

    return output_dict,output_weight_dict,sum_weight_dict,df_gene_miR,text_res

#%% Plotting functions
    
def create_miRweight_boxplot(vari_df, weight_dict, target_key, cell_line,plot_name,miR_thresh,delay_cutoff=None,save_path=''):
    """
    Function which creates a boxplot showing the difference in variance
    between three groups of miRNA weights (0,>=-0.3, and <-0.3).
    The variance is log transformed and the pvalues between these log transformations
    are indicated on lines connecting the different groups.
    
    The number of genes in each gorup is shown in parenthesis under
    their respective groups.

    Parameters
    ----------
    vari_df : pandas dataframe
        dataframe containing the name of the genes along with their variance.
    weight_dict : dictionnary
        dictionnary containing the gene names and the associated miRNA weight.
    target_key : string
        key that indicates which subset of information needs to be plotted.
    cell_line : string
        name of the cell line being used, serves to determine save location of the plot.
    plot_name : string
        title of the plot.
    miR_thresh : string
        The threshold used for miRNA filters, variable is used for naming purposes
    delay_cutoff : int, optional
        Implement a cutoff on the delays/variance being used. The default is None.
    save_path : string
        The path to which the plot will be saved

    Returns
    -------
    None.

    """
    new_dict={}
    new_dict['genes']=vari_df['gene_names']
    new_dict['miR_weights']=weight_dict[target_key]
    new_dict['variance']=list(vari_df[target_key])
    extracted_df=pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in new_dict.items() ]))
    extracted_df = extracted_df.sort_values(by=['miR_weights'], ascending=False)

    
    #Apply the cutoff if necessary
    if delay_cutoff:
        extracted_df=extracted_df[abs(extracted_df.variance)<=delay_cutoff]
    #Find number of genes in each groups
    G1_num=len(np.where(np.asarray(extracted_df['miR_weights'])==0)[0])
    G2_num=len(np.where(np.asarray(extracted_df['miR_weights'])>=-0.3)[0])-len(np.where(np.asarray(extracted_df['miR_weights'])==0)[0])
    G3_num=len(np.where(np.asarray(extracted_df['miR_weights'])<-0.3)[0])

    # Convert results to log10
    sub_list_1=np.log10(list(extracted_df.variance[extracted_df.miR_weights==0]))
    sub_list_2=np.log10(list(extracted_df.variance[(extracted_df.miR_weights<0) & (extracted_df.miR_weights>=-0.3)]))
    sub_list_3=np.log10(list(extracted_df.variance[extracted_df.miR_weights<-0.3]))

    # Calculate and format the pvalue from the ttest between each category
    pval_g1_g2="{0:.1e}".format(stats.ttest_ind(sub_list_1,sub_list_2)[1])
    pval_g2_g3="{0:.1e}".format(stats.ttest_ind(sub_list_2,sub_list_3)[1])
    pval_g1_g3="{0:.1e}".format(stats.ttest_ind(sub_list_1,sub_list_3)[1])

    # Create list of results
    data_list=[sub_list_1,sub_list_2,sub_list_3]
    #Plot the boxplot
    fig = plt.figure(figsize =(7, 4),dpi=900) 
    ax  = fig.add_subplot(111)
    ax.set_aspect(0.17)
    bp = ax.boxplot(data_list)
    # bp["medians"][0][0][1]=3
    plt.xticks([1, 2, 3], ['0\n('+str(G1_num)+')', '>= -0.3\n('+str(G2_num)+')', '< -0.3\n('+str(G3_num)+')'],size=15)
    plt.yticks(size=15)
    # plt.ylim(0,0.5)
    
    #Extract the 'caps' and 'fliers' of the boxplots
    extract_lines = {key : [v.get_data() for v in value] for key, value in bp.items()}

        
    my_caps=extract_lines['caps']
    idx_lst=[1,3,5]
    my_caps= [my_caps[i] for i in idx_lst]
    cap_dict={}
    #Format the extracted caps for easier use
    for idx,groups in enumerate(['group_1','group_2','group_3']):
        cap_dict[groups]=my_caps[idx]
        
    #Calculate the height needed for lines to compare the different results
    G1_G2=max(cap_dict['group_1'][1][0],cap_dict['group_2'][1][0])*1.10
    G2_G3=max(cap_dict['group_2'][1][0],cap_dict['group_3'][1][0])*1.10
    if len(extract_lines['fliers'][1][1]) > 0:
        max_mid_flier=max(extract_lines['fliers'][1][1]) #Obtain maximum height of flier for middle point
        G1_G3=max(G1_G2,G2_G3,max_mid_flier)*1.12
    else:
        G1_G3=max(G1_G2,G2_G3)*1.12
    # Plot line between the groups
    plt.hlines(G1_G2,cap_dict['group_1'][0][1],cap_dict['group_2'][0][0],colors='black')
    #Calculate the mid point on the line above, then plot the pvalue results on the line
    mid_x=cap_dict['group_1'][0][0]+((cap_dict['group_2'][0][0]-cap_dict['group_1'][0][0])/2)
    plt.text(mid_x,G1_G2*1.01,s=pval_g1_g2,va='bottom',ha='center')
    
    plt.hlines(G2_G3,cap_dict['group_2'][0][1],cap_dict['group_3'][0][0],colors='black')
    mid_x=cap_dict['group_2'][0][0]+((cap_dict['group_3'][0][0]-cap_dict['group_2'][0][0])/2)
    plt.text(mid_x,G2_G3*1.01,s=pval_g2_g3,va='bottom',ha='center')
    
    plt.hlines(G1_G3*1.1,cap_dict['group_1'][0][1],cap_dict['group_3'][0][0],colors='white')#Plot white bar to increase height of window so the text fits
    plt.hlines(G1_G3,cap_dict['group_1'][0][1],cap_dict['group_3'][0][0],colors='black')
    mid_x=cap_dict['group_1'][0][0]+((cap_dict['group_3'][0][0]-cap_dict['group_1'][0][0])/2)
    plt.text(mid_x,G1_G3*1.01,s=pval_g1_g3,va='bottom',ha='center')
    
    # #Replaces the 'medians' with MADs
    # #This essentially serves as a means to use a line of the same dimensions
    # #as the median but using the MAD statistic
    # for MAD_idx,MAD in enumerate(MAD_list):
    #     for idx,val in enumerate(extract_lines['medians'][MAD_idx][1]):
    #         extract_lines['medians'][MAD_idx][1][idx]=MAD
    # #Removal of the median line
    # for median in bp['medians']: 
    #     median.set(linewidth = 0) 
        
    # for line_idx,line in enumerate(extract_lines['medians']):
    #     plt.plot(extract_lines['medians'][line_idx][0],extract_lines['medians'][line_idx][1],color='red')
    # Set the labels
    if save_path=='':
        plot_path='all_figures/'+cell_line+'/merged_replicates/miRNA_boxplots/'+miR_thresh
    else:
        plot_path=save_path+'/'+miR_thresh
    if not os.path.exists(plot_path):
        os.makedirs(plot_path, exist_ok=True)
    ax.set_ylabel('log10(variance ratio)',size=20)
    ax.set_xlabel('miRNA weight category',size=20)
    ax.set_title(('Boxplot for '+target_key+' cells'),size=25)
    # plot_path="all_figures",cell_line
    # plt.savefig(os.path.join(plot_path,name),bbox_inches='tight')
    plt.savefig((plot_path+'/'+plot_name+'.png'),bbox_inches='tight')
    plt.clf()
    plt.close("all")
    # plt.show()
    gc.collect()

#%% Wrapper functions for analysis

def wrapper_miRNA_boxplot_analysis(cell_line,replicates,layers,target_folder,variance_param,miRNA_thresh_list,save_path='',gene_selection='all',single_rep=False):
    """
    Wrapper which enables the plotting of miRNA boxplots for either single replicates or
    merged replicates. The wrapper function takes in a list of miRNA thresholds
    which it will execute in one after the other, for each subsequent execution
    it will remove the genes which had 'weights' from the previous threshold.
    
    The order in which thresholds are given is therefore important, where the likely
    utility of the function is to give the larger thresholds first and make your way 
    down to the smaller threhsolds.
    
    Each threshold must have the associated file in 'data_files/miRNA_files/categorized '
    
    Function written by Yohan Lefol

    Parameters
    ----------
    cell_line : string
        Indicates the cell linebing used.
    replicates : list
        list of strings indicating the replicates belonging to the cell line.
    layers : list
        list of strings for the layers used (likely ['spliced','unspliced'].
    target_folder : string
        Either a specific replciate or folder group (ex:A_B) showing where the data
        will be fetched from.
    variance_param : string
        Can be 'All', 'Phases', or 'Both'. This indicates which groupings will
        be plotted for the miRNA boxplots, where 'All' is all cells, 'phases' will
        split cells into their found cell cycle phases, and 'Both' will do both
        of the described methods
    miRNA_thresh_list : list
        List of thresholds (ex: 100_1000) Thresholds are executed sequentially based
        on ordering in the provided list.
    save_path : string, optional
        Path where the results will be saved. The default is ''.
    gene_selection : list, optional
        A list of genes to use in the analysis. The default is 'all'
    single_rep : Boolean, optional
        Boolean stating if the given 'target_folder' is a single replicate or not.
        The default is False.

    Returns
    -------
    None.

    """
    #Define save_path if not done by user
    if save_path=='' and single_rep==True:
        save_path='all_figures/'+cell_line+'/single_replicate_analysis/'+target_folder+'/miRNA_boxplot'
    elif save_path=='' and single_rep==False:
        save_path='all_figures/'+cell_line+'/merged_replicate_analysis/'+target_folder+'/miRNA_boxplot'
    
    #Target scanfile
    TS_path='data_files/miRNA_files/TS_files/Predicted_Targets_Context_Scores.default_predictions.txt'
    
    #Get main data
    mean_dict,CI_dict,bool_dict,count_dict,boundary_dict=my_utils.get_CI_data(cell_line,layers,target_folder,gene_selection)
    gene_dict,variance_dict=prep_variances_for_TS(mean_dict,boundary_dict,keys_to_use=variance_param)
    
    #Convert to proper format
    variance_df = pd.DataFrame.from_dict(variance_dict)
    variance_df['gene_names']=variance_df.index
    
    #Ensures that the order is the same as in gene_dict - this is very important
    first_gene_dict_key=list(gene_dict.keys())[0]
    true_sort = [s for s in gene_dict[first_gene_dict_key] if s in variance_df.gene_names.unique()]
    variance_df = variance_df.set_index('gene_names').loc[true_sort].reset_index()
    
    
    gene_exclusion_list=[]
    for miRNA_thresh in miRNA_thresh_list:
        #Perform miRNA analysis
        path_miRNA='data_files/miRNA_files/categorized/'+cell_line+'_miRNA_'+str(miRNA_thresh)+'.csv'
        miRNA_pd=pd.read_csv(path_miRNA)
        miRNA_list=list(miRNA_pd.found)
        TS_dict,TS_weight_dict,sum_weight_dict,miRNA_df,text_res=target_scan_analysis(TS_path,gene_dict=gene_dict,cell_line=cell_line,miR_thresh=miRNA_thresh,miRNA_list=miRNA_list)
        
        #Remove genes in necessary
        if len(gene_exclusion_list) != 0:
            gene_removal_values=variance_df.gene_names.isin(gene_exclusion_list).value_counts()
            for my_bool, cnts in gene_removal_values.iteritems():
                if my_bool==True:
                    text_res=text_res+"Removing "+str(cnts)+" genes using the provided exclusion list"
            good_idx=list(np.where(~variance_df["gene_names"].isin(gene_exclusion_list)==True)[0])
            variance_df=variance_df.iloc[good_idx]
            
            #Subset_sum_weight_dict
            for key in sum_weight_dict.keys():
                sum_weight_dict[key] = [sum_weight_dict[key][i] for i in good_idx]
            
        
        #Plot the boxplots
        for key in sum_weight_dict.keys():
            my_name=key+'_'+str(miRNA_thresh)
            create_miRweight_boxplot(vari_df=variance_df, weight_dict=sum_weight_dict, target_key=key, cell_line=cell_line,plot_name=my_name,miR_thresh=miRNA_thresh,delay_cutoff=None,save_path=save_path)
    
        #Save text res
        with open(save_path+'/'+my_name+'_genes_used_summary.txt', 'w') as f:
            f.write(text_res)

    
        #Update gene exclusion list
        result_df=variance_df.copy()
        result_df['weight']=sum_weight_dict[list(sum_weight_dict.keys())[0]]
        miR_col=[]
        for gene in result_df['gene_names']:
            miRNA_idx=list(np.where(miRNA_df['Gene Symbol']==gene)[0])
            miRNA_df_subset=miRNA_df.loc[miRNA_idx]
            miR_slash_list = '/'.join(miRNA_df_subset["miRNA"])
            miR_col.append(miR_slash_list)
        result_df['miRNAs']=miR_col
                
        save_name=save_path+'/'+my_name+'_data_results.csv'
        result_df.to_csv(save_name,index=False)
        
        idx_genes_with_weights=list(np.where(result_df.weight!=0)[0])
        new_gene_exclusion=result_df.iloc[idx_genes_with_weights]
        new_gene_exclusion=list(new_gene_exclusion.gene_names)
        
        if len(gene_exclusion_list) == 0:
            gene_exclusion_list=new_gene_exclusion
        else:
            for gene in new_gene_exclusion:
                gene_exclusion_list.append(gene)
                
                
#%% statistical analysis functions

def spearman_comp_delay_with_UTR(UTR_file,sig_delays,delay_cat='inc_to_+1'):
    """
    Function which calculates the spearman correlation between the 3'UTR length and
    a delay category.
    
    The function first sorts both sets of values by gene name, then performs the 
    correlation for the provided delay category

    Parameters
    ----------
    UTR_file : pandas dataframe
        The UTR file containing the 3'UTR lengths.
    sig_delays : pandas dataframe
        The delays for the significant genes.
    delay_cat : string, optional
        string indicating the delay category that will be correlated with the 
        3'UTR lengths. The default is 'inc_to_+1'.


    Returns
    -------
    None.

    """
    list_of_genes=list(UTR_file.gene_name[UTR_file.gene_name.isin(sig_delays.gene_name)])
    
    UTR_sub=UTR_file[UTR_file.gene_name.isin(list_of_genes)]
    delay_sub=sig_delays[sig_delays.gene_name.isin(list_of_genes)]
    delay_sub=delay_sub.sort_values(by='gene_name')
    UTR_sub=UTR_sub.sort_values(by='gene_name')
    
    x_arr=np.asarray(UTR_sub.Length)
    y_arr=np.asarray(delay_sub[delay_cat])
    return stats.spearmanr(x_arr,y_arr)
    
def compare_UTR_based_on_delay(UTR_file,sig_delays,delay_cat,delay_thresh):
    """
    Function which calculates the Mann Whitney U statistic between two sets of
    3'UTR lengths. The two sets are created based on a threshold where one set
    will be all the 3'UTR lengths of genes which are greater or equal to the 
    delay threshold of the provided delay category. While the other set will be
    the other genes (below the delay threshold for the delay category)

    Parameters
    ----------
    UTR_file : pandas dataframe
        dataframe containing the 3' UTR lengths.
    sig_delays : pandas dataframe
        dataframe containing the delay values for each category.
    delay_cat : string
        the delay category to be used.
    delay_thresh : int
        the delay threshold to be used.

    Returns
    -------
    None.

    """
    significant_genes=sig_delays.gene_name
    sig_UTRs=UTR_file[UTR_file.gene_name.isin(significant_genes)]

    delay_idx_pos=np.where(sig_delays[delay_cat]>=delay_thresh)
    delay_genes_pos=[list(sig_delays.gene_name)[i] for i in list(delay_idx_pos[0])]
    UTR_arr_pos=np.asarray(sig_UTRs.Length[sig_UTRs.gene_name.isin(delay_genes_pos)])
    
    delay_idx_neg=np.where(sig_delays[delay_cat]<delay_thresh)
    delay_genes_neg=[list(sig_delays.gene_name)[i] for i in list(delay_idx_neg[0])]
    UTR_arr_neg=np.asarray(sig_UTRs.Length[sig_UTRs.gene_name.isin(delay_genes_neg)])
    
    return stats.mannwhitneyu(UTR_arr_pos,UTR_arr_neg)