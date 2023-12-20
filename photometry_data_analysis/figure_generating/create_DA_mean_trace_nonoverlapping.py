# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 13:28:57 2023

@author: qianl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import random
import matplotlib as mpl
import seaborn as sns
import re
import csv
import pickle
import sys
# sys.path.insert(0,os.path.join(path,'functions'))
from b_load_processed_mat_save_pickle_per_mouse import Mouse_data
from b_load_processed_mat_save_pickle_per_mouse import pickle_dict
from b_load_processed_mat_save_pickle_per_mouse import load_pickleddata
from scipy.signal import savgol_filter
from sklearn.preprocessing import normalize
import matplotlib.cm as cm  
#%% filter bad trials when there's no water signal
#only keep the trace when the animal actually gets water
def remove_bad_trials(mat,window = [40,50],thres = 5):
    mat_window = mat[:,window[0]:window[1]]
    max_val = mat_window.max(axis = 1)
    ind = max_val>=thres
    return mat[ind,:]
def average_signal_by_trial(multiregion_dict ,region,types,phase,length,num_mouse,session_num,
                            session_of_phase = None,mouse_index = None,rm_bad = True,rm_window = [40,50]):
    key = region
    ave_response_trace = np.full([session_num,length,num_mouse],np.nan) #d1 session, d2 time series, d3 mouse, fill with full trace
    for mouse_id in range(num_mouse):
        for session_of_phase in range(len(phase_dict[phase])):
            try:
                x = multiregion_dict[key][types][mouse_id][:,:,phase_dict[phase][session_of_phase]]
                if rm_bad:
                    x = remove_bad_trials(x,window = [rm_window[0],rm_window[1]])
                ave_response_trace[session_of_phase,:,mouse_id] = np.nanmean(x,axis = 0)
            except:
                pass
    return ave_response_trace
def find_max_response_from_ave_signal(mat):  
    return np.nanmax(mat,axis = 1)
#%%

region_data = load_pickleddata('D:/PhD/Data_Code_Contingency_Uchida/Photometry/pickles/region_based_data_by_group_filtered.pickle')
trialtypes = ['go','go_reward','no_go','unpred_water','go_omit','background','c_reward','c_omit',
              'lk_aligned_c_rw','lk_aligned_go_rw','lk_aligned_unpred_rw',
              'pk_aligned_c_rw','pk_aligned_go_rw','pk_aligned_unpred_rw']
good_regions = load_pickleddata('D:/PhD/Data_Code_Contingency_Uchida/Photometry/pickles/good_regions.pickle')
phase_dict = {'cond_days':[0,1,2,3,4],
              'deg_days':[5,6,7,8,9],
              'rec_days':[10,11,12],
              'ext_days':[13,14,15],
              'finalrec_days':[16,17],
              'c_odor_days':[5,6,7,8,9],
              }    
#%% non-overlapping stacked traces
group = 'deg_group'
region = 'LNacS'
trialtypes = ['go','no_go','go_omit','background','c_reward','c_omit'] # deg
# trialtypes = ['go','no_go','go_omit','background','c_reward','c_omit',  #c
#               ]
for types in trialtypes:
    for phase in phase_dict.keys():
        
        region_group_data = region_data[group]
        num_mouse = len(region_group_data[region][types])
        session_num = len(phase_dict[phase])
        length = 180
        if types in ['go','unpred_water','c_reward']:
            rm_bad = False            
        else:
            rm_bad = False
        average_mat = average_signal_by_trial(region_group_data,region,types,
                                              phase,length,num_mouse,session_num,rm_bad = rm_bad)   
        
        
            # ave_response = find_max_response_from_ave_signal(average_mat)
        # normalize
        # normalizer_peak = np.sum(np.nanmax(average_mat,axis = 1),axis = 0)
        new_mat = average_mat.copy()
        # for i in range(session_num):
        #     for j in range(num_mouse):
        #     # ax[i].plot(np.nanmean(a[i,:,:],axis =1))
        #         new_mat[i,:,j] = average_mat[i,:,j]/normalizer_peak[j]*np.mean(normalizer_peak)
        
        
        
        fig,ax = plt.subplots(1,1,figsize = (4,5))
        plt.title(region+'_'+types+'_'+phase)

        ax.fill_between([40,60], [15,15], [-26,-26],color = 'grey',edgecolor = None, alpha=0.2)
        ax.fill_between([110,111], [15,15], [-26,-26], color = 'blue', alpha=0.4,edgecolor = None,)  
        for i in range(session_num):
            
            ax.axhline(y = i*(-6), color = 'grey', linewidth = 1)
            aa = np.nanmean(new_mat[i,:,:],axis =1) + i*(-6)
            ax.plot(aa,alpha = 0.8,color = 'k',linewidth = 0.5) #subtract the baseline again
            stde = np.nanstd(new_mat[i,:,:],axis = 1)/np.sqrt(new_mat[i,:,:].shape[1])
            ax.fill_between(np.arange(0,180,1), aa-stde, aa+stde,alpha = 0.3,color = 'k')
 
            
            # ax[i].fill_between([40,60], [6,6], [-0.5,-0.5],color = 'purple',edgecolor = None, alpha=0.2)
            # ax[i].fill_between([110,114], [6,6], [-0.5,-0.5], color = 'blue', alpha=0.4,edgecolor = None,)
            # ax[i].axhline(y = 0, color = 'grey', linewidth = 1)
            # aa = np.nanmean(new_mat[i,:,:],axis =1)
            # ax[i].plot(aa,alpha = 0.8,color = 'k') #subtract the baseline again
            # stde = np.nanstd(new_mat[i,:,:],axis = 1)/np.sqrt(new_mat[i,:,:].shape[1])
            # ax[i].fill_between(np.arange(0,180,1), aa-stde, aa+stde,alpha = 0.3,color = 'k')
        plt.yticks(np.arange(0,(session_num)*(-6),-6),np.arange(0,session_num))
        plt.xticks(np.arange(0,180,20),np.arange(-2,7,1))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
            # ax[i].set_xlabel(f'{phase} {i+1}')
            
            # if i == 0:
            #     ax[i].set_ylabel('normalized DA signal(Z-score)')      
        savepath = 'D:/PhD/Figures/photometry/average_stacked'
        # plt.savefig("{0}/{1}_{2}_{3}_{4}.png".format(savepath,region,phase,types,group), bbox_inches="tight", dpi = 300)
        # plt.savefig("{0}/{1}_{2}_{3}_{4}.pdf".format(savepath,region,phase,types,group), bbox_inches="tight", dpi = 300)
        plt.show()
#%% lick aligned water
group = 'deg_group'
region = 'LNacS'
trialtypes = ['pk_aligned_unpred_rw'] # deg
# trialtypes = ['go','no_go','go_omit','background','c_reward','c_omit',  #c
#               'lk_aligned_c_rw','lk_aligned_go_rw']
phase= 'deg_days'
for types in trialtypes:

    
    region_group_data = region_data[group]
    num_mouse = len(region_group_data[region][types])
    session_num = len(phase_dict[phase])

    length = 80
    rm_bad = True

    average_mat = average_signal_by_trial(region_group_data,region,types,
                                          phase,length,num_mouse,session_num,rm_bad = rm_bad)   
    
    
        # ave_response = find_max_response_from_ave_signal(average_mat)
    # normalize
    # normalizer_peak = np.sum(np.nanmax(average_mat,axis = 1),axis = 0)
    new_mat = average_mat.copy()
    # for i in range(session_num):
    #     for j in range(num_mouse):
    #     # ax[i].plot(np.nanmean(a[i,:,:],axis =1))
    #         new_mat[i,:,j] = average_mat[i,:,j]/normalizer_peak[j]*np.mean(normalizer_peak)
    
    
    
    fig,ax = plt.subplots(1,1,figsize = (4,5))
    plt.title(region+'_'+types+'_'+phase)
    ax.fill_between([38,39], [15,15], [-26,-26], color = 'blue', alpha=0.4,edgecolor = None,)  
    for i in range(session_num):
        
        ax.axhline(y = i*(-6), color = 'grey', linewidth = 1)
        aa = np.nanmean(new_mat[i,:,:],axis =1) + i*(-6)
        ax.plot(aa,alpha = 0.8,color = 'k') #subtract the baseline again
 
        
        # ax[i].fill_between([40,60], [6,6], [-0.5,-0.5],color = 'purple',edgecolor = None, alpha=0.2)
        # ax[i].fill_between([110,114], [6,6], [-0.5,-0.5], color = 'blue', alpha=0.4,edgecolor = None,)
        # ax[i].axhline(y = 0, color = 'grey', linewidth = 1)
        # aa = np.nanmean(new_mat[i,:,:],axis =1)
        # ax[i].plot(aa,alpha = 0.8,color = 'k') #subtract the baseline again
        # stde = np.nanstd(new_mat[i,:,:],axis = 1)/np.sqrt(new_mat[i,:,:].shape[1])
        # ax[i].fill_between(np.arange(0,180,1), aa-stde, aa+stde,alpha = 0.3,color = 'k')
    plt.yticks(np.arange(0,(session_num)*(-6),-6),np.arange(0,session_num))
    plt.xticks(np.arange(0,80,20),np.arange(-2,2,1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
        # ax[i].set_xlabel(f'{phase} {i+1}')
        
        # if i == 0:
        #     ax[i].set_ylabel('normalized DA signal(Z-score)')      
    # savepath = 'D:/PhD/Data_Code_Contingency_Uchida/Figures'
    # plt.savefig("{0}/new_{1}_{2}_{3}_{4}.png".format(savepath,region,phase,types,group), bbox_inches="tight", dpi = 300)
    # plt.savefig("{0}/new_{1}_{2}_{3}_{4}.pdf".format(savepath,region,phase,types,group), bbox_inches="tight", dpi = 300)
    plt.show()