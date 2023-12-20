# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 17:35:34 2022

@author: qianl
"""


from datetime import datetime
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import random
import matplotlib as mpl
import re
import csv
import pickle
import sys
# sys.path.insert(0,os.path.join(path,'functions'))
from b_load_processed_mat_save_pickle_per_mouse import Mouse_data
from b_load_processed_mat_save_pickle_per_mouse import pickle_dict
from b_load_processed_mat_save_pickle_per_mouse import load_pickleddata
from sklearn.linear_model import LinearRegression
def normalize_mat(mat, axis):
    
    norm = mat.T/np.nansum(mat,axis = axis)
    return norm.T
def search_rise_point(trace,pivot):
    pos_index = [x for x in trace if x >0]
    while pivot >-1:
        if pivot-1 not in pos_index:
            return pivot -1
        pivot -= 1
    return 0
    
    
    
def align_trace(mat, diff_peak_window,pre_time,post_time):
    new_mat = np.full([mat.shape[0],pre_time+post_time],np.nan)
    for i in range(mat.shape[0]):
        if str(mat[i,:][20]) == 'nan':
            continue
        smooth_mat = savgol_filter(mat[i,:], 7, 3)
        diff_trace = np.diff(smooth_mat[diff_peak_window[0]:diff_peak_window[1]])

        rise_max_in_window = np.argmax(diff_trace)+1
        # print(mat[i,diff_peak_window[0]:diff_peak_window[1]][rise_max_in_window])
        if mat[i,diff_peak_window[0]:diff_peak_window[1]][rise_max_in_window] <3: # a temporary way of removing trials without licking
            continue
    
        rise_point = search_rise_point(diff_trace[diff_peak_window[0]:diff_peak_window[1]],rise_max_in_window)
        
        new_mat[i,:] = mat[i,rise_point+diff_peak_window[0]-pre_time:rise_point+diff_peak_window[0]+post_time]
    return new_mat




#%% step 1 change the data structure to brain region based, align the unpredicted reward response and save
# load data
data = load_pickleddata('D:/PhD/Data_Code_Contingency_Uchida/Photometry/pickles/corrected_DA_gcamp_data_combined_zscore_by_ITI.pickle')
# dict for good regions
good_regions = {'AMOT':['FgDA_01','FgDA_03','FgDA_04','FgDA_06','FgDA_07','FgDA_09',
                        'FgDA_C6','FgDA_C4','FgDA_C2','FgDA_C1'],
                'PMOT':['FgDA_01','FgDA_02','FgDA_04','FgDA_05','FgDA_06','FgDA_07','FgDA_09',
                        'FgDA_C6','FgDA_C4','FgDA_C2','FgDA_C1'],
                'ALOT':['FgDA_03','FgDA_04','FgDA_05','FgDA_06','FgDA_09',
                        'FgDA_C6','FgDA_C5','FgDA_C4','FgDA_C2'],
                'PLOT':['FgDA_01','FgDA_02','FgDA_04','FgDA_05','FgDA_06','FgDA_07','FgDA_09','FgDA_05',
                        'FgDA_C6','FgDA_C5','FgDA_C4','FgDA_C2','FgDA_C1'],
                'MNacS':['FgDA_01','FgDA_02','FgDA_03','FgDA_04','FgDA_06','FgDA_07','FgDA_09',
                         'FgDA_C6','FgDA_C4','FgDA_C2','FgDA_C1'],
                'LNacS':['FgDA_01','FgDA_02','FgDA_03','FgDA_04','FgDA_05','FgDA_06','FgDA_07','FgDA_09',
                         'FgDA_C6','FgDA_C5','FgDA_C4','FgDA_C2','FgDA_C1']}


save_path = 'D:/PhD/Data_Code_Contingency_Uchida/Photometry/pickles'
filename = 'good_regions'
pickle_dict(good_regions,save_path,filename)  

#%% lick_filtered data
#'FgDA_01','FgDA_02','FgDA_03','FgDA_04','FgDA_05','FgDA_06','FgDA_07','FgDA_08','FgDA_09'


#%% 我感觉没有必要改成region based，需要改一下以后的code
# trialtypes
ttypes= ['go','no_go','unpred_water','go_omit','background','c_omit','c_reward',
         'lk_aligned_c_rw','lk_aligned_go_rw','lk_aligned_unpred_rw',
         'pk_aligned_c_rw','pk_aligned_go_rw','pk_aligned_unpred_rw']
multiregion_dict_deg = {}
for site, mice in good_regions.items():
    
    multiregion_dict_deg[site] = multiregion_dict_deg.get(site,{})
    for mouse in mice:
        if 'C' not in mouse:
            for types in ttypes:
                multiregion_dict_deg[site][types] = multiregion_dict_deg[site].get(types,[])
    
                try:
                    # if types in ['go','unpred_water']:
                    #     multiregion_dict[site][types].append(data[mouse][site]['signal_gcamp_lick_filtered'][types][:,:,:19])    
                    # else:
                    multiregion_dict_deg[site][types].append(data[mouse][site]['signal_gcamp'][types][:,:,:19])    
                except:
                    pass

filename = 'region_based_data_licking_filtered_deg'
pickle_dict(multiregion_dict_deg,save_path,filename)  


multiregion_dict_C = {}    
for site, mice in good_regions.items():
    
    multiregion_dict_C[site] = multiregion_dict_C.get(site,{})
    for mouse in mice:
        if 'C' in mouse:
            for types in ttypes:
                multiregion_dict_C[site][types] = multiregion_dict_C[site].get(types,[])
    
                try:
                    # if types in ['go','unpred_water']:
                    #     multiregion_dict[site][types].append(data[mouse][site]['signal_gcamp_lick_filtered'][types][:,:,:19])    
                    # else:
                    multiregion_dict_C[site][types].append(data[mouse][site]['signal_gcamp'][types][:,:,:19])    
                except:
                    pass

filename = 'region_based_data_licking_filtered_C'
pickle_dict(multiregion_dict_C,save_path,filename)                     
    
    
region_based_by_group = {}
region_based_by_group['deg_group'] = multiregion_dict_deg
region_based_by_group['c_group'] = multiregion_dict_C
    
filename = 'region_based_data_by_group_filtered'
pickle_dict(region_based_by_group,save_path,filename) 

   
 #%%
region_data = load_pickleddata('D:/PhD/Photometry/DATA/photo-pickles/combined/region_based_data_licking_notyet_filtered_C.pickle')   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    