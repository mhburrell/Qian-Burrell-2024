# -*- coding: utf-8 -*-
"""
Created on Tue May 30 12:33:12 2023

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
import scipy.stats as stats
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
from matplotlib.colors import LinearSegmentedColormap

# Create a custom colormap for the gradient
def create_colormap(colors, n_segments):
    return LinearSegmentedColormap.from_list('custom_cmap', colors, N=n_segments)
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

#%% convert data to four conditions

conditions = {'conditioning':{},
             'degradation':{},
             'cue-C':{},
             'extinction':{}}

phase_dict = {'cond_days':[0,1,2,3,4],
              'deg_days':[5,6,7,8,9],
              'ext_days':[13,14,15],
              'c_odor_days':[5,6,7,8,9],
              }   

rm_bad = False
region = 'LNacS'
# trialtypes ='go' # deg
# trialtypes = ['go','no_go','go_omit','background','c_reward','c_omit',  #c
#               ]
trialtypes = ['go','no_go','go_omit']
for condition in conditions:
    for types in trialtypes:
        if condition == 'conditioning':
            phase = 'cond_days'
            region_group_data = region_data['deg_group']
            num_mouse = len(region_group_data[region][types])
            session_num = len(phase_dict[phase])
            length = 180
            # if types in ['go','unpred_water','c_reward']:
            #     rm_bad = True            
            # else:
            #     rm_bad = False
            average_mat1 = average_signal_by_trial(region_group_data,region,types,
                                                  phase,length,num_mouse,session_num,rm_bad = rm_bad)  
    
            
            region_group_data = region_data['c_group']
            num_mouse = len(region_group_data[region][types])
            session_num = len(phase_dict[phase])
            length = 180
            # if types in ['go','unpred_water','c_reward']:
            #     rm_bad = True            
            # else:
            #     rm_bad = False
            average_mat2 = average_signal_by_trial(region_group_data,region,types,
                                                  phase,length,num_mouse,session_num,rm_bad = rm_bad)  
            conditions[condition][types]= np.dstack((average_mat1,average_mat2))
            
        elif condition == 'degradation':
            phase = 'deg_days'
            region_group_data = region_data['deg_group']
            num_mouse = len(region_group_data[region][types])
            session_num = len(phase_dict[phase])
            length = 180
            # if types in ['go','unpred_water','c_reward']:
            #     rm_bad = True            
            # else:
            #     rm_bad = False
            average_mat = average_signal_by_trial(region_group_data,region,types,
                                                  phase,length,num_mouse,session_num,rm_bad = rm_bad)  
            conditions[condition][types]=average_mat
        
            
        elif condition == 'cue-C':
            phase = 'c_odor_days'
            region_group_data = region_data['c_group']
            num_mouse = len(region_group_data[region][types])
            session_num = len(phase_dict[phase])
            length = 180
            # if types in ['go','unpred_water','c_reward']:
            #     rm_bad = True            
            # else:
            #     rm_bad = False
            average_mat = average_signal_by_trial(region_group_data,region,types,
                                                  phase,length,num_mouse,session_num,rm_bad = rm_bad)  
            conditions[condition][types] = average_mat
            
        else:
            phase = 'ext_days'
            region_group_data = region_data['deg_group']
            num_mouse = len(region_group_data[region][types])
            session_num = len(phase_dict[phase])
            length = 180
            # if types in ['go','unpred_water','c_reward']:
            #     rm_bad = True            
            # else:
            #     rm_bad = False
            average_mat = average_signal_by_trial(region_group_data,region,types,
                                                  phase,length,num_mouse,session_num,rm_bad = rm_bad)  
            conditions[condition][types] = average_mat
            
#%% plotting bar plots      
resp_types = ['go_omit']
window = [105,140]
index = 4
modes = ['conditioning', 'degradation', 'cue-C']
for resp_type in resp_types:
    array_mean = []
    array_ste = []
    array_response = []
    for phase in modes:
        
        try:
            array = np.sum(conditions[phase][resp_type][index,window[0]:window[1],:],axis = 0)
            
            isnan = np.isnan(array)
            array = array[~isnan]
            response_mean = np.nanmean(array)
            print(response_mean)
            response_ste = np.nanstd(array)/np.sqrt(len(array))
            array_mean.append(response_mean)
            array_response.append(array)
            array_ste.append(response_ste)
        except:
            array_mean.append(np.nan)
            array_response.append(None)
            array_ste.append(np.nan)
    plt.figure(figsize=(4,4))
    plt.bar(np.array([0,1,2]),array_mean,width=0.6,color = 'orchid')
    
    for i, resp_points in enumerate(array_response):
        if len(resp_points)>0:
            plt.scatter(np.ones(len(resp_points))*i, resp_points, facecolor = 'none', edgecolors = 'grey')
            
            plt.errorbar(i, array_mean[i],array_ste[i],color = 'grey', lw = 2.5, capsize = 6, markeredgewidth = 2.5)
    
    plt.xticks(ticks = np.array([0,1,2]),labels = modes)
    plt.xlim([-0.5,2.5])
    # plt.ylim([-0.5,0.25])
    plt.title(resp_type)
    plt.savefig(resp_type+'_auc105-140.pdf',dpi = 300,)
    plt.show()            
            
            
            
            
            
#%% statistical test for DA response in different conditions

#trialtypes = ['go','no_go','go_omit']
# no go response = no go 40:70 auc
types = 'no_go'
group1 = np.sum(conditions['conditioning'][types][-1,45:60,:],axis = 0)
group2 = np.sum(conditions['degradation'][types][-1,45:60,:],axis = 0)
group3 = np.sum(conditions['cue-C'][types][-1,45:60,:],axis = 0)

statistic, pvalue = stats.f_oneway(group1[~np.isnan(group1)], group2[~np.isnan(group2)])
print('----------------group1&2'+types+'^^^'+'--------------------')
print("Statistic:", statistic, ", p-value:", pvalue)
statistic, pvalue = stats.f_oneway(group1[~np.isnan(group1)], group3[~np.isnan(group3)])
print('----------------group1&3'+types+'^^^'+'--------------------')
print("Statistic:", statistic, ", p-value:", pvalue)
statistic, pvalue = stats.f_oneway(group2[~np.isnan(group2)], group3[~np.isnan(group3)])
print('----------------group2&3'+types+'^^^'+'--------------------')
print("Statistic:", statistic, ", p-value:", pvalue)
print(np.mean(group1),np.mean(group2),np.mean(group3))
#%%
# odor A response = go_omit 40:60 peak
types = 'go_omit'
group1 = np.max(conditions['conditioning'][types][-1,40:60,:],axis = 0)
group2 = np.max(conditions['degradation'][types][-1,40:60,:],axis = 0)
group3 = np.max(conditions['cue-C'][types][-1,40:60,:],axis = 0)

statistic, pvalue = stats.f_oneway(group1[~np.isnan(group1)], group2[~np.isnan(group2)])
print('----------------group1&2'+types+'^^^'+'--------------------')
print("Statistic:", statistic, ", p-value:", pvalue)
statistic, pvalue = stats.f_oneway(group1[~np.isnan(group1)], group3[~np.isnan(group3)])
print('----------------group1&3'+types+'^^^'+'--------------------')
print("Statistic:", statistic, ", p-value:", pvalue)
statistic, pvalue = stats.f_oneway(group2[~np.isnan(group2)], group3[~np.isnan(group3)])
print('----------------group2&3'+types+'^^^'+'--------------------')
print("Statistic:", statistic, ", p-value:", pvalue)

#%%
# odor A water response = go 110:130 peak
types = 'go'
group1 = np.max(conditions['conditioning'][types][-1,110:130,:],axis = 0)
group2 = np.max(conditions['degradation'][types][-1,110:130,:],axis = 0)
group3 = np.max(conditions['cue-C'][types][-1,110:130,:],axis = 0)

statistic, pvalue = stats.f_oneway(group1[~np.isnan(group1)], group2[~np.isnan(group2)])
print('----------------group1&2'+types+'^^^'+'--------------------')
print("Statistic:", statistic, ", p-value:", pvalue)
statistic, pvalue = stats.f_oneway(group1[~np.isnan(group1)], group3[~np.isnan(group3)])
print('----------------group1&3'+types+'^^^'+'--------------------')
print("Statistic:", statistic, ", p-value:", pvalue)
statistic, pvalue = stats.f_oneway(group2[~np.isnan(group2)], group3[~np.isnan(group3)])
print('----------------group2&3'+types+'^^^'+'--------------------')
print("Statistic:", statistic, ", p-value:", pvalue)

#%%
# go omission = go omit dip 105:140
types = 'go_omit'
group1 = np.sum(conditions['conditioning'][types][-1,105:140,:],axis = 0)
group2 = np.sum(conditions['degradation'][types][-1,105:140,:],axis = 0)
group3 = np.sum(conditions['cue-C'][types][-1,105:140,:],axis = 0)

statistic, pvalue = stats.f_oneway(group1[~np.isnan(group1)], group2[~np.isnan(group2)])
print('----------------group1&2'+types+'^^^'+'--------------------')
print("Statistic:", statistic, ", p-value:", pvalue)
statistic, pvalue = stats.f_oneway(group1[~np.isnan(group1)], group3[~np.isnan(group3)])
print('----------------group1&3'+types+'^^^'+'--------------------')
print("Statistic:", statistic, ", p-value:", pvalue)
statistic, pvalue = stats.f_oneway(group2[~np.isnan(group2)], group3[~np.isnan(group3)])
print('----------------group2&3'+types+'^^^'+'--------------------')
print("Statistic:", statistic, ", p-value:", pvalue)

print(np.mean(group1),np.mean(group2),np.mean(group3))





















