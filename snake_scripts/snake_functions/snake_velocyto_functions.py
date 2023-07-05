#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 09:36:01 2021

@author: yohanl
"""

# import snake_utils as my_utils
#The 'snake_analysis_functions' can be called either through snakemake or 
#through an analysis script. This changes the directore in which the utils
#will be located. This block account for either option
try:
    from snake_functions import snake_utils as my_utils
except:
    from snake_scripts.snake_functions import snake_utils as my_utils


import velocyto as vcy
import numpy as np
import pandas as pd
import math
import os

import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt

import rpy2.robjects as robj
from rpy2.robjects.packages import importr

#%% data processing/calculations

def find_nearest(array,value):
    """
    Function which finds the value closest to a given value

    Parameters
    ----------
    array : numpy ndarray
        The array which will be searched.
    value : int
        The value which will be used to find the nearest value within the array.

    Returns
    -------
    array[idx] : int/float
        The value found to be nearest to the inputted value.

    """
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

def create_boundary_dict(vlm):
    """
    Generates a boundary dictionnary containning the order point which marks the
    boundary between phases. The order associated to a phase is the first order
    of that specific phase.
    
    The function also accounts for orientation reversal
    
    Function written by Yohan Lefol

    Parameters
    ----------
    vlm : velocyto.analysis.VelocytoLoom
        The loom file as read by velocyto.

    Returns
    -------
    boundary_dict : dictionnary
        A dictionnary containing the order boundaries of each phase, it is later
        used in plotting functions.

    """
    all_angles=vlm.ca["angles"].copy()
    all_angles.sort()
    boundary_dict={}
    for ang in np.unique(vlm.ca["angle_boundaries"]):
        boundary_angle=find_nearest(all_angles,ang)
        boundary_order=np.where(vlm.ca["angles"]==boundary_angle)[0][0]
        ang_phase=vlm.ca["phase"][np.where(vlm.ca['angle_boundaries']==ang)[0][0]]
        boundary_dict[ang_phase]=[boundary_order,vlm.colorandum[np.where(vlm.ca['phase']==ang_phase)[0][0]]]
    cell_num_dict={}
    for k in ["G1","S","G2M"]:
        cell_num_dict[k]=len(np.where(vlm.ca['phase']==k)[0])
                        
    # orientation=np.unique(vlm.ca['orientation'])[0]
    # print(boundary_dict)
    #new_boundaries
    # if orientation=='G1':
    #     X=len(vlm.ca["CellID"])-boundary_dict['G1'][0]
    #     boundary_dict['G1'][0]=0
    #     boundary_dict['S'][0]=cell_num_dict["G1"]
    #     boundary_dict['G2M'][0]=cell_num_dict["G1"]+cell_num_dict["S"]
    # else:
    #     X=len(vlm.ca["CellID"])-boundary_dict['G2M'][0]
    #     boundary_dict['G1'][0]=cell_num_dict["G2M"]+cell_num_dict["S"]
    #     boundary_dict['G2M'][0]=0
    #     boundary_dict['S'][0]=cell_num_dict["G2M"]
    
    # arr_1=np.arange(start=0,stop=X)
    # arr_2=np.arange(start=X,stop=len(vlm.ca["CellID"]))
    
    # new_order=np.concatenate([arr_2,arr_1])
    vlm.ca["new_order"]=vlm.ca["order"]
    
    return boundary_dict


def smooth_layers(vlm,bin_size, window_size, spliced_array, unspliced_array, orientation):
    """
    The function first calculates the mean points for each data using the bin size,
    it then elongates the data to fit the number of order points, and finally
    runs the data throught the moving_average function in order to smooth it.
    
    Function written by Yohan Lefol

    Parameters
    ----------
    vlm : velocyto.analysis.VelocytoLoom
        The loom file as read by velocyto.
    bin_size : int
        The bin size used to calculate the mean points.
    window_size : int
        The window size used for the moving average smoothing.
    spliced_array : numpy nd.array
        The array containing the spliced data.
    unspliced_array : numpy nd.array
        The array containing the unspliced data.
    orientation : string
        Either G1 or G2M to indicate the orientation.

    Returns
    -------
    spli_mean_array : numpy nd.array
        The mean and smoothed spliced data.
    unspli_mean_array : numpy nd.array
        The mean and smoothed unspliced data .

    """
    num_bin=int(len(vlm.ca["new_order"])/bin_size)
    spli_mean_list=[]
    unspli_mean_list=[]
    for order in range(num_bin):
        if order==0:    #First iteration
            spli_mean_list.append(np.mean(spliced_array[0:bin_size]))
            unspli_mean_list.append(np.mean(unspliced_array[0:bin_size]))
        else:  
            spli_mean_list.append(np.mean(spliced_array[order*bin_size:(order+1)*bin_size]))              
            unspli_mean_list.append(np.mean(unspliced_array[order*bin_size:(order+1)*bin_size]))
    
    
    last_index_check=num_bin*bin_size
    if len(vlm.ca["new_order"])%bin_size==0:
        last_index_check=int(last_index_check-(bin_size/2))

    last_val_spli=np.mean(spliced_array[last_index_check:len(vlm.ca["new_order"])])
    last_val_unspli=np.mean(unspliced_array[last_index_check:len(vlm.ca["new_order"])])
    
    spli_mean_list.append(last_val_spli)
    unspli_mean_list.append(last_val_unspli)
    
    spli_mean_list.insert(0,last_val_spli)
    unspli_mean_list.insert(0,last_val_unspli)

    # These two for loops extend the means as desired.
    for idx,val in enumerate(spli_mean_list):
        if idx==0:#First iterattion:
            spli_mean_array=np.linspace(start=val,stop=spli_mean_list[idx+1],num=bin_size)
        else:
            if idx!=len(spli_mean_list)-1:
                if idx == len(spli_mean_list)-2:#Last iteration
                    spli_mean_array=np.concatenate([spli_mean_array,np.linspace(start=val,stop=spli_mean_list[idx+1],num=len(vlm.ca["new_order"])-(num_bin*bin_size))])
                else:
                    spli_mean_array=np.concatenate([spli_mean_array,np.linspace(start=val,stop=spli_mean_list[idx+1],num=bin_size)])
    for idx,val in enumerate(unspli_mean_list):
        if idx==0:#First iterattion:
            unspli_mean_array=np.linspace(start=val,stop=unspli_mean_list[idx+1],num=bin_size)
        else:
            if idx != len(unspli_mean_list)-1:
                if idx == len(unspli_mean_list)-2:
                    unspli_mean_array=np.concatenate([unspli_mean_array,np.linspace(start=val,stop=unspli_mean_list[idx+1],num=len(vlm.ca["new_order"])-(num_bin*bin_size))])
                else:
                    unspli_mean_array=np.concatenate([unspli_mean_array,np.linspace(start=val,stop=unspli_mean_list[idx+1],num=bin_size)])
    
    spli_mean_array=my_utils.moving_average(spli_mean_array,window_size=window_size,orientation=orientation)
    unspli_mean_array=my_utils.moving_average(unspli_mean_array,window_size=window_size,orientation=orientation)

    return spli_mean_array,unspli_mean_array

def smooth_calcs(vlm, bin_size, window_size, spli_arr, unspli_arr, choice='mean'):
    """
    Function that calculates the derivative of spliced and unspliced, both velocity
    arrays are then smoothed using the smooth_layers function.
    
    Function written by Yohan Lefol

    Parameters
    ----------
    vlm : velocyto.analysis.VelocytoLoom
        The loom file as read by velocyto.
    bin_size : int
        The size of the bin used for mean lines in smooth_layers function.
    window_size : int
        The size of the window used for the moving average function.
    spli_arr : numpy nd.array
        The numpy array containing the spliced data.
    unspli_arr : numpy nd.array
        The numpy array containing the unspliced data.
    choice : string, optional
        Used to either return the mean lines or velocity lines. The default is 'mean'.
        
    Returns
    -------
    spli_array : numpy nd.array
        The array containing the modified spliced data.
    unspli_array : numpy nd.array
        The array containing the modifyed data.
    spli_mean_array : numpy nd.array
        The array containing the mean spliced data.
    unspli_mean_array : numpy nd.array
        The array containing the mean unspliced data.
    """
    
    orientation=np.unique(vlm.ca['orientation'])[0]
    spli_mean_array,unspli_mean_array=smooth_layers(vlm,bin_size=bin_size,window_size=window_size,spliced_array=spli_arr,unspliced_array=unspli_arr,orientation=orientation)
    
    # spli_mean_array=spli_arr
    # unspli_mean_array=unspli_arr    
    cell_axis=list(range(0,len(spli_mean_array)))
    if choice=='mean':
        return spli_mean_array,unspli_mean_array
    else:   #Choice should be vel
        # deriv_spli=np.diff(spli_mean_array)/np.diff(vlm.ca["new_order"])
        # deriv_unspli=np.diff(unspli_mean_array)/np.diff(vlm.ca["new_order"])
        deriv_spli=np.diff(spli_mean_array)/np.diff(cell_axis)
        deriv_unspli=np.diff(unspli_mean_array)/np.diff(cell_axis)
        
        #Numpy.diff removes a value, so we extrapolate the in between value between the last number of the array and the first
        #This regains the expected number of values.
        new_value_spli=np.linspace(deriv_spli[-1],deriv_spli[0],num=3)[1]
        new_value_unspli=np.linspace(deriv_unspli[-1],deriv_unspli[0],num=3)[1]
        
        deriv_spli=np.append(deriv_spli,new_value_spli)
        deriv_unspli=np.append(deriv_unspli,new_value_unspli)
        
        # deriv_spli,deriv_unspli=smooth_layers(vlm,bin_size=bin_size,window_size=window_size,spliced_array=deriv_spli,unspliced_array=deriv_unspli,orientation=orientation)
        # spli_mean_array,unspli_mean_array=smooth_layers(vlm,bin_size=bin_size,window_size=window_size,spliced_array=spli_mean_array,unspliced_array=unspli_mean_array,orientation=orientation)

    return deriv_spli,deriv_unspli,spli_mean_array,unspli_mean_array



def create_smooth_vels(vlm,window_size,return_dict=False):
    """
    Creates the smoothed velocities for both spliced and unspliced using the vlm object
    The smoothed velocities can be returned in dictionnary format or dataframe format
    
    The Sx variable of the vlm object contains the Knn smoothed spliced expression

    Function written by Yohan Lefol

    Parameters
    ----------
    vlm : velocyto.analysis.VelocytoLoom
        The loom file as read by velocyto.
    window_size : int
        The size of the window to use during the moving average smoothing step.
    return_dict : boolean, optional
        Indicates if the smoothed velocities should be returned in dictionnary or dataframe format.
        The default is False.

    Returns
    -------
    spli_return : dictionnary or pandas dataframe
        The values for the smoothed spliced velocity.
    unspli_return : dictionnary or pandas dataframe
        The values for the smoothed unspliced velocity.
    spli_mean_return : dictionnary or pandas dataframe
        The values for the mean spliced values used to calculate velocity.
    unspli_mean_return : dictionnary or pandas dataframe
        The values for the mean unspliced values used to calculate velocity.

    """
    spli_dict={}
    unspli_dict={}
    spli_mean_dict={}
    unspli_mean_dict={}
    
    for idx,val in enumerate(vlm.ra['Gene']):
        # deriv_spli,deriv_unspli=smooth_calcs(vlm,bin_size=100,window_size=window_size,spli_arr=vlm.Sx_sz[idx,:],unspli_arr=vlm.Ux_sz[idx,:],choice='vel')
        deriv_spli,deriv_unspli,spli_mean,unspli_mean=smooth_calcs(vlm,bin_size=100,window_size=window_size,spli_arr=vlm.Sx[idx,:],unspli_arr=vlm.Ux[idx,:],choice='vel')
        
        spli_dict[val]=deriv_spli
        unspli_dict[val]=deriv_unspli
        
        spli_mean_dict[val]=spli_mean
        unspli_mean_dict[val]=unspli_mean
    
    if return_dict==False:
        spli_return = pd.DataFrame.from_dict(spli_dict)
        unspli_return = pd.DataFrame.from_dict(unspli_dict)
        
        spli_mean_return = pd.DataFrame.from_dict(spli_mean_dict)
        unspli_mean_return = pd.DataFrame.from_dict(unspli_mean_dict)
    else:
        spli_return = spli_dict
        unspli_return = unspli_dict
        
        spli_mean_return = spli_mean_dict
        unspli_mean_return = unspli_mean_dict
        
    return spli_return,unspli_return,spli_mean_return,unspli_mean_return


#%% Plotting utils

def array_to_rmatrix(X):
    """
    
    Function taken from hgForebrainGlutamatergic velocyto notebook
    """
    nr, nc = X.shape
    xvec = robj.FloatVector(X.transpose().reshape((X.size)))
    xr = robj.r.matrix(xvec, nrow=nr, ncol=nc)
    return xr

def principal_curve(X, pca=True):
    """
    Function taken from hgForebrainGlutamatergic velocyto notebook
    
    Parameters
    ----------
    input : numpy.array
    returns:
    Result::Object
        Methods:
        projections - the matrix of the projection
        ixsort - the order ot the points (as in argsort)
        arclength - the lenght of the arc from the beginning to the point
    """
    # convert array to R matrix
    xr = array_to_rmatrix(X)
    
    if pca:
        #perform pca
        t = robj.r.prcomp(xr)
        #determine dimensionality reduction
        usedcomp = max( sum( np.array(t[t.names.index('sdev')]) > 1.1) , 4)
        usedcomp = min([usedcomp, sum( np.array(t[t.names.index('sdev')]) > 0.25), X.shape[0]])
        Xpc = np.array(t[t.names.index('x')])[:,:usedcomp]
        # convert array to R matrix
        xr = array_to_rmatrix(Xpc)

    #import the correct namespace
    princurve = importr("princurve",on_conflict="warn")
    
    #call the function
    fit1 = princurve.principal_curve(xr)

    
    #extract the outputs
    class Results:
        pass
    results = Results()
    results.projections = np.array( fit1[0] )
    results.ixsort = np.array( fit1[1] ) - 1 # R is 1 indexed
    results.arclength = np.array( fit1[2] )
    results.dist = np.array( fit1[3] )
    
    if pca:
        results.PCs = np.array(xr) #only the used components
        
    return results

def despline():
    """
    Function originates from DentateGyrus notebook for Velocyto

    Returns
    -------
    None.

    """
    ax1 = plt.gca()
    # Hide the right and top spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')


#%% Plotting


def plot_fractions_mod(vlm, title= None, save2file: str=None) -> None:
    """
    A modified version of velocyto's plot fraction function allowing a user to
    specify the plot's title.

    Parameters
    ----------
    vlm : velocyto.analysis.VelocytoLoom
        The loom file as read by velocyto.
    title : String, optional
        String value to be given as the title of the plot. The default is None.
    save2file : String, optional
        The path in which to save the file. The default is None.

    Returns
    -------
    None.


    """
    plt.figure(figsize=(3.2, 5))
    try:
        chips, chip_ix = np.unique(vlm.ca["SampleID"], return_inverse=1)
    except KeyError:
        chips, chip_ix = np.unique([i.split(":")[0] for i in vlm.ca["CellID"]], return_inverse=1)
    n = len(chips)
    for i in np.unique(chip_ix):
        tot_mol_cell_submatrixes = [X[:, chip_ix == i].sum(0) for X in [vlm.S, vlm.A, vlm.U]]
        total = np.sum(tot_mol_cell_submatrixes, 0)
        _mean = [np.mean(j / total) for j in tot_mol_cell_submatrixes]
        _std = [np.std(j / total) for j in tot_mol_cell_submatrixes]
        plt.ylabel("Fraction")
        my_bar=plt.bar(np.linspace(-0.2, 0.2, n)[i] + np.arange(3), _mean, 0.5 / (n * 1.05), label=chips[i])
        plt.errorbar(np.linspace(-0.2, 0.2, n)[i] + np.arange(3), _mean, _std, c="k", fmt="none", lw=1, capsize=2)

        # Hide the right and top spines
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().xaxis.set_ticks_position('bottom')
        plt.gca().spines['left'].set_bounds(0, 0.8)
        plt.legend()
        
    plt.xticks(np.arange(3), ["spliced", "ambiguous", "unspliced"])
    plt.tight_layout()
    
    for rect in my_bar:
        height=rect.get_height()
        height=round(height,2)
        plt.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(15, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')
    if title:
        plt.title(title)
    else:
        plt.title("Plot  fractions")
    if save2file:
        plt.savefig(save2file, bbox_inches="tight")
        


def plot_velocity_field(vlm, save2file=None):
    """
    Function which plots the velocity field along with a principal curve
    which shows the overall directionnality of the data.
    
    Function taken from the hgForebrainGlutamatergic notebook
    Function adapted by Yohan Lefol

    Parameters
    ----------
    vlm : velocyto.analysis.VelocytoLoom
        The loom file as read by velocyto.
    save2file : String
        The path to which the figure is saved
        
    Returns
    -------
    None.

    """
    #Calls the principal curve function, used to produce a principal curve on the velocity field
    pc_obj =principal_curve(vlm.pcs[:,:4], False)
    pc_obj.arclength = np.max(pc_obj.arclength) - pc_obj.arclength
    
    #Plots the velocity field
    plt.figure(None,(9,9))
    #Plots the main field
    vlm.plot_grid_arrows(scatter_kwargs_dict={"alpha":0.7, "lw":0.7, "edgecolor":"0.4", "s":70, "rasterized":True},
                          min_mass=2.9, angles='xy', scale_units='xy',
                          headaxislength=2.75, headlength=5, headwidth=4.8, quiver_scale=0.35, scale_type="absolute")
    #Plot the arrows
    plt.plot(pc_obj.projections[pc_obj.ixsort,0], pc_obj.projections[pc_obj.ixsort,1], c="w", lw=6, zorder=1000000)
    plt.plot(pc_obj.projections[pc_obj.ixsort,0], pc_obj.projections[pc_obj.ixsort,1], c="k", lw=3, zorder=2000000)
    # plt.gca().invert_xaxis()
    plt.axis("off")
    plt.axis("equal");
    legend_elements = [mpl.lines.Line2D([0], [0], marker='o', color="w", label='G1', markerfacecolor=vlm.colorandum[np.where(vlm.ca['phase']=='G1')[0][0]], markersize=10),
                       mpl.lines.Line2D([0], [0], marker='o', color="w", label='S', markerfacecolor=vlm.colorandum[np.where(vlm.ca['phase']=='S')[0][0]], markersize=10),
                       mpl.lines.Line2D([0], [0], marker='o', color="w", label='G2M', markerfacecolor=vlm.colorandum[np.where(vlm.ca['phase']=='G2M')[0][0]], markersize=10)]    
    
    plt.legend(handles=legend_elements,loc='best')
    despline()
    # plt.legend(loc='best')
    if save2file:
        plt.savefig(save2file, bbox_inches="tight")



def plot_markov(vlm, path=None):
    """
    Plots the standard PCA plot, and two markov plots, one showing the predicted
    points/cells for being the starting point,
    and another which shows the predicted end points/cells
    
    Function taken from the DentateGyrus velocyto notebook
    Function adapted by Yohan Lefol
    
    Parameters
    ----------
    vlm : velocyto.analysis.VelocytoLoom
        The loom file as read by velocyto.
    path : String
        The path to which the figures will be saved
    
    Returns
    -------
    None.
    
    """
    steps = 100, 100
    grs = []
    for dim_i in range(vlm.embedding.shape[1]):
        m, M = np.min(vlm.embedding[:, dim_i]), np.max(vlm.embedding[:, dim_i])
        m = m - 0.025 * np.abs(M - m)
        M = M + 0.025 * np.abs(M - m)
        gr = np.linspace(m, M, steps[dim_i])
        grs.append(gr)
    
    meshes_tuple = np.meshgrid(*grs)
    gridpoints_coordinates = np.vstack([i.flat for i in meshes_tuple]).T
    
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors()
    nn.fit(vlm.embedding)
    dist, ixs = nn.kneighbors(gridpoints_coordinates, 1)
    
    diag_step_dist = np.sqrt((meshes_tuple[0][0,0] - meshes_tuple[0][0,1])**2 + (meshes_tuple[1][0,0] - meshes_tuple[1][1,0])**2)
    min_dist = diag_step_dist / 2
    ixs = ixs[dist < min_dist]
    gridpoints_coordinates = gridpoints_coordinates[dist.flat[:]<min_dist,:]
    dist = dist[dist < min_dist]
    
    ixs = np.unique(ixs)
    
    plt.figure(None,(8,8))
    vcy.scatter_viz(vlm.embedding[ixs, 0], vlm.embedding[ixs, 1], c=vlm.colorandum[ixs], alpha=1, s=30, lw=0.4, edgecolor="0.4")
    if path:
        plt.savefig(path+"/markov_PCA", bbox_inches="tight")

    ###Start of markov end
    vlm.prepare_markov(sigma_D=diag_step_dist, sigma_W=diag_step_dist/2., direction='forward', cells_ixs=ixs)
    vlm.run_markov(starting_p=np.ones(len(ixs)), n_steps=2500)
    diffused_n = vlm.diffused - np.percentile(vlm.diffused, 3)
    diffused_n /= np.percentile(diffused_n, 97)
    diffused_n = np.clip(diffused_n, 0, 1)
    plt.figure(None,(7,7))
    plt.title(label="End points")
    vcy.scatter_viz(vlm.embedding[ixs, 0], vlm.embedding[ixs, 1], c=diffused_n, alpha=0.5, s=50, lw=0., edgecolor=None, cmap="viridis_r", rasterized=True)
    plt.axis("off")
    if path:
        plt.savefig(path+"/markov_end_points", bbox_inches="tight")
    
    ##Start of Markov beginning
    vlm.prepare_markov(sigma_D=diag_step_dist, sigma_W=diag_step_dist/2., direction='backwards', cells_ixs=ixs)
    vlm.run_markov(starting_p=np.ones(len(ixs)), n_steps=2500)
    diffused_n = vlm.diffused - np.percentile(vlm.diffused, 3)
    diffused_n /= np.percentile(diffused_n, 97)
    diffused_n = np.clip(diffused_n, 0, 1)
    plt.figure(None,(7,7))
    plt.title(label="Beginning point")
    vcy.scatter_viz(vlm.embedding[ixs, 0], vlm.embedding[ixs, 1], c=diffused_n, alpha=0.5, s=50, lw=0., edgecolor=None, cmap="viridis_r", rasterized=True)
    plt.axis("off")
    if path:
        plt.savefig(path+"/markov_start_points", bbox_inches="tight")
        
        
        
#%%Velocyto CI iteration functions

def save_smooth_vels(vlm,the_dict, cell_line, replicate, layer,file_name,iteration_name='Iterations'):
    """
    Function which saves the smoothed velocities from a dictionnary
    The function can also be used to save the mean expression values used to
    calculate the velocity as the dictionnaries have the same format.

    Parameters
    ----------
    the_dict : Dictionnary
        Dictionnary containing the smoothed velocities.
    cell_line : string
        a string containing the name of the cell line being saved.
    replicate : string
        a string containing the name of the replicate being saved.
    layer : string
        a string containing the name of the layer (spliced or unspliced).
    file_name : string
        a string with the name to be given to the file that will be saved.
    iteration_name : string
        a string which gives the name of the iteration folder where the date will be
        save. By default 'Iteration' which is expected to house the velocity data.

    Returns
    -------
    None.

    """
    the_file=pd.DataFrame.from_dict(the_dict)
    the_file.index =list(vlm.ca['CellID'])
    # the_file.set_axis(list(vlm.ca['CellID']), inplace=True)
    the_path='data_files/confidence_intervals/'+cell_line+'/'+replicate+'/'+iteration_name+'/'+layer
    
    my_utils.create_folder(the_path)
        
    bf = open(the_path+"/"+file_name+".bin", "wb")
    pickle.dump(the_file, bf, -1)
    bf.close()
    # the_file.to_csv(the_path+"/"+file_name+".csv",index=False)


def save_vlm_values(vlm,cell_line,replicate,layer,file_name):
    """
    Function which saves the unspliced or spliced values of the processed and 
    fildetered loom file. 
    
    Function written by Yohan Lefol

    Parameters
    ----------
    vlm : velocyto.analysis.VelocytoLoom
        The loom file as read by velocyto.
    cell_line : string
        string defining the cell line being used.
    replicate : string
        string defining the replicate being used.
    layer : string
        string indicating if the save target is the splice dor unspliced layer
    file_name : string
        the name that will be given to the file.

    Returns
    -------
    None.

    """
    
    save_path='data_files/confidence_intervals/'+cell_line+'/'+replicate+'/vlm_vals_iters/'+layer
    
    if os.path.isdir(save_path) == False:
        my_utils.create_folder(save_path)
    
    if layer=='spliced':
        vlm_df=pd.DataFrame(vlm.Sx,index=list(vlm.ra['Gene']))
        vlm_df=vlm_df.transpose()
    else:#Layer is unspliced
        vlm_df=pd.DataFrame(vlm.Ux,index=list(vlm.ra['Gene']))
        vlm_df=vlm_df.transpose()
    
    vlm_df.index=list(vlm.ca['CellID'])
    
    bf = open(save_path+'/'+file_name+'.bin', "wb")
    pickle.dump(vlm_df, bf, -1)
    bf.close()
    
    # vlm_df.to_csv(save_path+'/'+file_name+'.csv',index=False)


def save_iteration_data(vlm, dta_to_save, cell_line, replicate, layer, file_name, save_choice):
    """
    Function which saves the velocities, expression data, or mean/smoothed expression from a dictionnary.c

    Parameters
    ----------
    vlm : velocyto.analysis.VelocytoLoom
        The loom file as read by velocyto.
    dta_to_save : Dictionnary
        Dictionnary containing the smoothed velocities.
    cell_line : string
        a string containing the name of the cell line being saved.
    replicate : string
        a string containing the name of the replicate being saved.
    layer : string
        a string containing the name of the layer (spliced or unspliced).
    file_name : string
        the name that will be given to the file.
    save_choice : string
        A string indicating what type of data is being saved. Either 'exp', 'exp_mean',
        or 'vel'.

    Returns
    -------
    None.

    """
    
    if save_choice=='vel':
        folder_name='Velocity_iterations'
    elif save_choice=='exp':
        folder_name='Expression_iterations'
    elif save_choice=='exp_mean':
        folder_name='Smooth_expression_iterations'
    else:
        print('Incorrect save choice given')
        return(None)
    
    save_path='data_files/confidence_intervals/'+cell_line+'/'+replicate+'/'+folder_name+'/'+layer
    
    if os.path.isdir(save_path) == False:
        my_utils.create_folder(save_path)
    
    if save_choice =='exp':
        if layer=='spliced':
            dta_to_save=pd.DataFrame(vlm.Sx,index=list(vlm.ra['Gene']))
            dta_to_save=dta_to_save.transpose()
        else:#Layer is unspliced
            dta_to_save=pd.DataFrame(vlm.Ux,index=list(vlm.ra['Gene']))
            dta_to_save=dta_to_save.transpose()
    else:
        dta_to_save=pd.DataFrame.from_dict(dta_to_save)
    
    dta_to_save.index =list(vlm.ca['CellID'])
    
        
    bf = open(save_path+"/"+file_name+".bin", "wb")
    pickle.dump(dta_to_save, bf, -1)
    bf.close()


def calculate_mean_of_cells_optimized(path,z_val,do_CI,num_iter):
    """
    Calculates the mean velocity of each cell based on the amount of files that were created.
    The function will load the entirety of the iterations in a dictionnary of genes,
    each gene being a 2D array. The confidence intervals and mean velocity are then calculated
    from these 2D arrays and returned.

    Parameters
    ----------
    path : string
        The path to the iterations to calculate the mean velocity.
    z_val : float
        The value to be used to calculate the confidence interval.
    do_CI : boolean
        A boolean indicating if the confidence intervals should be calculated
        and the returned
    num_iter : Integer
        The number of iterations performed. Used in the calculation of the conidence intervals
    

    Returns
    -------
    my_gene_dict : Dictionnary
        Dictionnary containing the calculated mean velocities of the iterations per gene per cell.
    up_CI_dict : Dictionnary
        Dictionnary containing the upper confidence interval for each gene at each cell.
    low_CI_dict : Dictionnary
        Dictionnary containing the lower confidence interval for each gene at each cell.
    my_var_dict : Dictionnary
        Dictionnary containing the variability between iterations for each gene at each cell


    """
    list_of_files=os.listdir(path)
    
    #Iterate over all files to get common genes and number of cells
    col_list=[]
    row_list=[]
    for file in list_of_files:
        temp_df=pd.read_csv(path+"/"+file)
        # print(temp_df)
        if col_list==[]:
            col_list=list(temp_df.columns)
        else:
            col_list=list(set(col_list) & set(list(temp_df.columns)))
        if row_list==[]:
            row_list=list(temp_df.index)
        else:
            row_list=list(set(row_list) & set(list(temp_df.index)))
    #Set up result dictionnaries
    my_gene_dict={}
    if do_CI:
        up_CI_dict={}
        low_CI_dict={}
    

    #Set up the numpy 2D arrays for result storage
    for gene in col_list:
        my_gene_dict[gene]=np.zeros(shape=(len(row_list),len(list_of_files)))
    
    #Iterate over files and store results in respective 2D arrays
    for idx,file in enumerate(list_of_files):
        my_df=pd.read_csv(path+"/"+file)
        for gene in my_gene_dict.keys():
            if gene in my_df.columns and gene !='Unnamed: 0':
                # print(my_df[gene])
                my_gene_dict[gene][:,idx]=my_df[gene].values
    
    #Calculate the variability between the iterations for each cell per gene
    my_var_dict={}
    for gene in my_gene_dict.keys():
        my_var_dict[gene]=np.var(my_gene_dict[gene],axis=1)
    
    #Perform calculations for each gene
    for gene in my_gene_dict.keys():
        if do_CI:
            #Could use np.sqrt(len(my_gene_dict[gene]))? if n is the number of cells?
            #Noting this in case I need to switch it
            ##BELOW ARE THE TWO POSSIBLE LINES OF CODE
            CI = (my_gene_dict[gene].std(axis=1)/np.sqrt(num_iter)) * z_val
            # CI = (my_gene_dict[gene].std(axis=1)/np.sqrt(len(my_gene_dict[gene]))) * z_val
            my_gene_dict[gene]=my_gene_dict[gene].mean(axis=1)
            up_CI_dict[gene]=my_gene_dict[gene]+CI
            low_CI_dict[gene]=my_gene_dict[gene]-CI
        else:
            my_gene_dict[gene]=my_gene_dict[gene].mean(axis=1)
    
    if do_CI:
        return my_gene_dict, up_CI_dict, low_CI_dict, my_var_dict
    else:
        return my_gene_dict