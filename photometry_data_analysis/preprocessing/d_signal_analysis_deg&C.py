# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 15:14:19 2023

@author: qianl
QC: 
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
import seaborn as sns
def find_consecutive_numbers(input_list):
    consecutive_count = 0
    for index, num in enumerate(input_list):
        if num > 0.5:
            consecutive_count += 1
            if consecutive_count == 3:
                return index - 2
        else:
            consecutive_count = 0
    return None


def movement_correction(signal_g, signal_i):
    ind_nan = np.isnan(signal_g) + np.isnan(signal_i)
    nonnan_signal_g = signal_g[~ind_nan]
    nonnan_signal_i = signal_i[~ind_nan]
    reg = LinearRegression().fit(nonnan_signal_i.reshape(-1, 1), nonnan_signal_g.reshape(-1, 1))
    beta1, beta0 = reg.coef_[0], reg.intercept_[0]
    signal_subtract = beta0 + beta1 * signal_i
    return signal_g - signal_subtract
def get_timepoint_1stlick_aftwater(lickings, timepoint=5.5, time_window=1.5):
    return [
        min([x for x in val if timepoint <= x <= timepoint + time_window], default=np.nan)
        for val in lickings.values()
    ]
#%%
mice = ['FgDA_C1','FgDA_C2','FgDA_C4','FgDA_C5','FgDA_C6','FgDA_C7']
path = 'D:/PhD/Photometry/DATA/photo-pickles'
trialtypes = ['go', 'no_go', 'go_omit', 'background','c_reward','c_omit']+['lk_aligned_go_rw','lk_aligned_c_rw','lk_aligned_unpred_rw']+['pk_aligned_go_rw','pk_aligned_c_rw','pk_aligned_unpred_rw']
align_method = 'first_lick_after_water'

#create new dict
data = {}

# get trials
for mouse_id in mice:
    print('Processing: ', mouse_id)
    #load mouse
    load_path = os.path.join(path,'{0}_stats.pickle'.format(mouse_id))
    mouse = load_pickleddata(load_path)

    # get doric and bpod data
    mouse_trials = mouse.df_bpod_doric
    
    # choose a date
    all_days = mouse.all_days.copy()
    data[mouse_id] = {}

    for site in [key for key in mouse_trials[all_days[0]]['corr_ROI'].keys()]:
        #initialize site and licking data
        data[mouse_id] = {'site':{},'lickings': {'licking_data': {}, 'first_lick_after_water': {}}, site: {'signal_gcamp': {}}}
        for trialtype in trialtypes :
            # Determine dimensions
            dims = [160, 80, len(all_days)] if trialtype in ['lk_aligned_go_rw', 'lk_aligned_c_rw', 'lk_aligned_unpred_rw', 'pk_aligned_go_rw', 'pk_aligned_c_rw', 'pk_aligned_unpred_rw'] else [160, 180, len(all_days)]            
            # Initialize NumPy arrays and lists
            data[mouse_id][site]['signal_gcamp'][trialtype] = np.full(dims, np.nan)
            data[mouse_id]['lickings']['first_lick_after_water'][trialtype] = []
            data[mouse_id]['lickings']['licking_data'][trialtype] = []
        
        for index,day in enumerate(all_days):
            dataframe = mouse_trials[day]['dataframe'].copy()
            whole_ITI_signal = []
            cur_TT = dataframe['Trialtype'] #or 'go_omit' # index of go trials
            for trial in range(len(dataframe)):
                signal_g = dataframe[site].values[trial]
                signal_i = dataframe[site+'_isos'].values[trial]               
                if trialtype == 'background':
                    signal_corrected = movement_correction(signal_g, signal_i)
                    whole_ITI_signal.extend(movement_correction(signal_g, signal_i))
                else:
                    pre_range, post_range = slice(0, 35), slice(150, None)
                    whole_ITI_signal.extend(movement_correction(signal_g[pre_range], signal_i[pre_range]))
                    whole_ITI_signal.extend(movement_correction(signal_g[post_range], signal_i[post_range]))

            std_session = np.nanstd(np.asarray(whole_ITI_signal)) 
            
            for trialtype in trialtypes:             
                is_x_TT = dataframe['Trialtype'] == trialtype #or 'go_omit' # index of go trials
                signal_xTT = dataframe[is_x_TT]
                num_trial = len(signal_xTT[site].values)

                if num_trial == 0:
                    data[mouse_id]['lickings']['first_lick_after_water'][trialtype].append(None)
                    data[mouse_id]['lickings']['licking_data'][trialtype].append(None)                      
                else:
                    lickings = signal_xTT.lickings
                    data[mouse_id]['lickings']['licking_data'][trialtype].append(lickings)   
                    
                    if trialtype in ['go','c_reward','unpred_water']:
                        timepoint_1stlick_aftwater = get_timepoint_1stlick_aftwater(lickings)                        
                        data[mouse_id]['lickings']['first_lick_after_water'][trialtype].append(timepoint_1stlick_aftwater)
                    else:
                        data[mouse_id]['lickings']['first_lick_after_water'][trialtype].append(None)
                            
                for trial in range(num_trial):
                    signal_g = signal_xTT[site].values[trial]
                    signal_i = signal_xTT[site+'_isos'].values[trial]
                    ind_nan = np.isnan(signal_g) + np.isnan(signal_i)
                    if trialtype == 'background':
                        signal_corrected = movement_correction(signal_g, signal_i)    
                    else:
                        nonnan_signal_g = np.concatenate([signal_g[~ind_nan][0:35],signal_g[~ind_nan][150:]]) # select ITI areas
                        nonnan_signal_i = np.concatenate([signal_i[~ind_nan][0:35],signal_i[~ind_nan][150:]])
                        reg = LinearRegression().fit(nonnan_signal_i.reshape(-1,1),nonnan_signal_g.reshape(-1,1))
                        beta1 = reg.coef_[0]
                        beta0 = reg.intercept_[0]       
                        # movement correction
                        signal_subtract = beta0 + beta1*signal_i
                        signal_corrected = signal_g - signal_subtract

                    # Skips the specific conditions where no action is required
                    if np.nanstd(signal_corrected[10:30]) == 0 or sum(np.isnan(signal_corrected)) == len(signal_corrected):
                        pass
                    
                    #minus pre-odor baseline and then zscored by the whole session            
                    values1 = (signal_corrected - np.nanmean(signal_corrected[0:40]))/std_session
                    data[mouse_id][site]['signal_gcamp'][trialtype][trial,:,index] = values1[1:181]
                    #                      
                    if trialtype in ['go','c_reward','unpred_water']:
                        tp = data[mouse_id]['lickings']['first_lick_after_water'][trialtype][index][trial]
                        licks = data[mouse_id]['lickings']['licking_data'][trialtype][index].values[trial]
                        if not np.isnan(tp):
                            #'got_water'
                            rise_index = np.where(np.diff(values1[108:140])>5)[0][0]+110
                            lick_time = rise_index/20
                            idx = np.searchsorted(licks, lick_time, side='left') - 1                          
                            closest_val = licks[idx]# Get the value in the array that is less than the given value
                            if np.nanmean(values1[int(closest_val*20):int(closest_val*20)+5])>3:
                                if trialtype == 'go':
                                    data[mouse_id][site]['signal_gcamp']['pk_aligned_go_rw'][trial,:,index] = values1[int(closest_val*20)-40:int(closest_val*20)+40]- np.nanmean(values1[int(closest_val*20)-20:int(closest_val*20)])        
                                elif trialtype == 'c_reward':
                                        data[mouse_id][site]['signal_gcamp']['pk_aligned_c_rw'][trial,:,index] = values1[int(closest_val*20)-40:int(closest_val*20)+40]- np.nanmean(values1[int(closest_val*20)-20:int(closest_val*20)])
                                elif trialtype == 'unpred_water':
                                        data[mouse_id][site]['signal_gcamp']['pk_aligned_unpred_rw'][trial,:,index] = values1[int(closest_val*20)-40:int(closest_val*20)+40]- np.nanmean(values1[int(closest_val*20)-20:int(closest_val*20)])
                        
                            #first lick                            
                            if trialtype == 'go':
                                data[mouse_id][site]['signal_gcamp']['lk_aligned_go_rw'][trial,:,index] = values1[int(tp*20)-42:int(tp*20)+38] - np.nanmean(values1[int(tp*20)-20:int(tp*20)])
                            elif trialtype == 'c_reward':
                                data[mouse_id][site]['signal_gcamp']['lk_aligned_c_rw'][trial,:,index] = values1[int(tp*20)-42:int(tp*20)+38]- np.nanmean(values1[int(tp*20)-20:int(tp*20)])
                            elif trialtype == 'unpred_water':
                                data[mouse_id][site]['signal_gcamp']['lk_aligned_unpred_rw'][trial,:,index] = values1[int(tp*20)-42:int(tp*20)+38]- np.nanmean(values1[int(tp*20)-20:int(tp*20)])

   #%%         
save_path = path+'/combined'
filename = 'corrected_gcamp_data_C_zscore_by_ITI'
pickle_dict(data,save_path,filename) 

 
#%% analyze degradation group
mice = ['FgDA_01','FgDA_02','FgDA_03',
        'FgDA_04','FgDA_05','FgDA_06',
        'FgDA_07','FgDA_08','FgDA_09','FgDA_C1','FgDA_C2','FgDA_C4','FgDA_C5','FgDA_C6','FgDA_C7']

# mice = ['FgDA_C1']
path = 'D:/PhD/Data_Code_Contingency_Uchida/Photometry/pickles'

data = {}

# get trials


for mouse_id in mice:
    if 'C' in mouse_id:
        trialtypes = ['go', 'no_go', 'go_omit', 'background','c_reward','c_omit']
    else:
        trialtypes = ['go', 'no_go', 'go_omit', 'background','unpred_water']
    print(mouse_id)
    load_path = os.path.join(path,'{0}_stats.pickle'.format(mouse_id))
    mouse = load_pickleddata(load_path)

    # assign two df 
    mouse_trials = mouse.df_bpod_doric
    
    # choose a date
    all_days = mouse.all_days.copy()
    data[mouse_id] = {}
      
    for site in [key for key in mouse_trials[all_days[0]]['corr_ROI'].keys()]:
        
        data[mouse_id][site] = {}
        data[mouse_id][site]['signal_gcamp'] = {}
        data[mouse_id]['lickings'] = {}
        data[mouse_id]['lickings']['licking_data'] = {}
        data[mouse_id]['lickings']['first_lick_after_water'] = {}
        for trialtype in trialtypes +['lk_aligned_go_rw','lk_aligned_c_rw','lk_aligned_unpred_rw']+['pk_aligned_go_rw','pk_aligned_c_rw','pk_aligned_unpred_rw']:
            if trialtype not in ['lk_aligned_go_rw','lk_aligned_c_rw','lk_aligned_unpred_rw']+['pk_aligned_go_rw','pk_aligned_c_rw','pk_aligned_unpred_rw']:
                data[mouse_id][site]['signal_gcamp'][trialtype] = np.full([160,180,len(all_days)], np.nan)               
                data[mouse_id]['lickings']['first_lick_after_water'][trialtype] = []
                data[mouse_id]['lickings']['licking_data'][trialtype] = []
            else:
                data[mouse_id][site]['signal_gcamp'][trialtype] = np.full([160,80,len(all_days)], np.nan)
        
        for index in range(len(all_days)):
            day = all_days[index] 
            dataframe = mouse_trials[day]['dataframe'].copy()
            whole_ITI_signal = []
            for trial in range(len(dataframe)):
                cur_TT = dataframe['Trialtype'] #or 'go_omit' # index of go trials
                
                signal_g = dataframe[site].values[trial]
                signal_i = dataframe[site+'_isos'].values[trial]
                ind_nan = np.isnan(signal_g) + np.isnan(signal_i)
                if trialtype == 'background':
                    nonnan_signal_g = signal_g[~ind_nan]
                    nonnan_signal_i = signal_i[~ind_nan]
                    reg = LinearRegression().fit(nonnan_signal_i.reshape(-1,1),nonnan_signal_g.reshape(-1,1))
                    beta1 = reg.coef_[0]
                    beta0 = reg.intercept_[0]       
                    # movement correction
                    signal_subtract = beta0 + beta1*signal_i
                    signal_corrected = signal_g - signal_subtract
                    whole_ITI_signal += list(signal_corrected)

                else:
                    nonnan_signal_g = np.concatenate([signal_g[~ind_nan][0:35],signal_g[~ind_nan][150:]]) # select ITI areas
                    nonnan_signal_i = np.concatenate([signal_i[~ind_nan][0:35],signal_i[~ind_nan][150:]])
                    reg = LinearRegression().fit(nonnan_signal_i.reshape(-1,1),nonnan_signal_g.reshape(-1,1))
                    beta1 = reg.coef_[0]
                    beta0 = reg.intercept_[0]       
                    # movement correction
                    signal_subtract = beta0 + beta1*signal_i
                    signal_corrected = signal_g - signal_subtract
                    whole_ITI_signal += list(signal_corrected[0:35])
                    whole_ITI_signal += list(signal_corrected[150:])

            
            std_session = np.nanstd(np.asarray(whole_ITI_signal)) 
            
            for trialtype in trialtypes:
                
                is_x_TT = dataframe['Trialtype'] == trialtype #or 'go_omit' # index of go trials
                signal_xTT = dataframe[is_x_TT]
                num_trial = len(signal_xTT[site].values)
                
                # A BLOCK FOR LICKING ALIGNMENT 

                if num_trial == 0:
                    data[mouse_id]['lickings']['first_lick_after_water'][trialtype].append(None)
                    data[mouse_id]['lickings']['licking_data'][trialtype].append(None)   
                    
                else:
                    lickings = signal_xTT.lickings
                    data[mouse_id]['lickings']['licking_data'][trialtype].append(lickings)   
                    if trialtype in ['go','c_reward','unpred_water']:
                        timepoint_1stlick_aftwater = []
                    
             
                        timepoint = 5.5
                        for key, val in lickings.items():
                            
                            valid_lick = [x for x in val if timepoint<=x <=timepoint+1.5]
                            if len(valid_lick) == 0:
                                timepoint_1stlick_aftwater.append(np.nan)
                            else:
                                timepoint_1stlick_aftwater.append(min(valid_lick))
      
                        
                        data[mouse_id]['lickings']['first_lick_after_water'][trialtype].append(timepoint_1stlick_aftwater)
                    else:
                        data[mouse_id]['lickings']['first_lick_after_water'][trialtype].append(None)
                            
                    
                


                
                
                
                for trial in range(num_trial):
                    signal_g = signal_xTT[site].values[trial]
                    signal_i = signal_xTT[site+'_isos'].values[trial]
                    ind_nan = np.isnan(signal_g) + np.isnan(signal_i)
                    if trialtype == 'background':
                        nonnan_signal_g = signal_g[~ind_nan]
                        nonnan_signal_i = signal_i[~ind_nan]
                        reg = LinearRegression().fit(nonnan_signal_i.reshape(-1,1),nonnan_signal_g.reshape(-1,1))
                        beta1 = reg.coef_[0]
                        beta0 = reg.intercept_[0]       
                        # movement correction
                        signal_subtract = beta0 + beta1*signal_i
                        signal_corrected = signal_g - signal_subtract
    
                    else:
                        nonnan_signal_g = np.concatenate([signal_g[~ind_nan][0:35],signal_g[~ind_nan][150:]]) # select ITI areas
                        nonnan_signal_i = np.concatenate([signal_i[~ind_nan][0:35],signal_i[~ind_nan][150:]])
                        reg = LinearRegression().fit(nonnan_signal_i.reshape(-1,1),nonnan_signal_g.reshape(-1,1))
                        beta1 = reg.coef_[0]
                        beta0 = reg.intercept_[0]       
                        # movement correction
                        signal_subtract = beta0 + beta1*signal_i
                        signal_corrected = signal_g - signal_subtract

                        # df/f
                    if np.nanstd(signal_corrected[10:30]) == 0:
                        pass
                        
                    elif sum(np.isnan(signal_corrected)) == len(signal_corrected):
                        
                        pass
                    else: #minus pre-odor baseline and then zscored by the whole session
                        
                        values1 = (signal_corrected - np.nanmean(signal_corrected[0:40]))/std_session
                        data[mouse_id][site]['signal_gcamp'][trialtype][trial,:,index] = values1[1:181]
                        
#                      'got_water':
                        if trialtype in ['go','c_reward','unpred_water']:
                            tp = data[mouse_id]['lickings']['first_lick_after_water'][trialtype][index][trial]
                            if not np.isnan(tp):
                                diff_list = np.diff(values1[110:140])
                                if find_consecutive_numbers(diff_list):
                                    rise_index = find_consecutive_numbers(diff_list) + 110
                                    if trialtype == 'go':
                                        data[mouse_id][site]['signal_gcamp']['pk_aligned_go_rw'][trial,:,index] = values1[int(rise_index)-40:int(rise_index)+40]- np.nanmean(values1[int(rise_index)-20:int(rise_index)])        
                                    elif trialtype == 'c_reward':
                                            data[mouse_id][site]['signal_gcamp']['pk_aligned_c_rw'][trial,:,index] = values1[int(rise_index)-40:int(rise_index)+40]- np.nanmean(values1[int(rise_index)-20:int(rise_index)])
                                    elif trialtype == 'unpred_water':
                                            data[mouse_id][site]['signal_gcamp']['pk_aligned_unpred_rw'][trial,:,index] = values1[int(rise_index)-40:int(rise_index)+40]- np.nanmean(values1[int(rise_index)-20:int(rise_index)])

#                      'first_lick_after_water':
                        if trialtype in ['go','c_reward','unpred_water']:
                            tp = data[mouse_id]['lickings']['first_lick_after_water'][trialtype][index][trial]
                            licks = data[mouse_id]['lickings']['licking_data'][trialtype][index].values[trial]
                            if trialtype == 'go':
                                if not np.isnan(tp):
                                    data[mouse_id][site]['signal_gcamp']['lk_aligned_go_rw'][trial,:,index] = values1[int(tp*20)-42:int(tp*20)+38] - np.nanmean(values1[int(tp*20)-20:int(tp*20)])
                                    
                            elif trialtype == 'c_reward':
                                if not np.isnan(tp):
                                    data[mouse_id][site]['signal_gcamp']['lk_aligned_c_rw'][trial,:,index] = values1[int(tp*20)-42:int(tp*20)+38]- np.nanmean(values1[int(tp*20)-20:int(tp*20)])
                            elif trialtype == 'unpred_water':
                                if not np.isnan(tp):
                                    data[mouse_id][site]['signal_gcamp']['lk_aligned_unpred_rw'][trial,:,index] = values1[int(tp*20)-42:int(tp*20)+38]- np.nanmean(values1[int(tp*20)-20:int(tp*20)])

#%% save separately into two pickels

deg_dict = {}
C_dict = {}
     
for mouse in mice:
    if 'C' in mouse:
        C_dict[mouse] = data[mouse]
    else:
        deg_dict[mouse] = data[mouse]
        
# save data
save_path = path
filename = 'corrected_gcamp_data_deg_zscore_by_ITI'
pickle_dict(deg_dict,save_path,filename)      
filename = 'corrected_gcamp_data_C_zscore_by_ITI'
pickle_dict(C_dict,save_path,filename)    

filename = 'corrected_gcamp_data_combined_zscore_by_ITI'
pickle_dict(data,save_path,filename) 
        




#%%
deg_dict = load_pickleddata('D:/PhD/Data_Code_Contingency_Uchida/Photometry/pickles/corrected_gcamp_data_deg_zscore_by_ITI.pickle')
#%%
save_path = 'D:/PhD/Data_Code_Contingency_Uchida/Photometry/pickles/'
filename = 'corrected_gcamp_data_deg_zscore_by_ITI'
pickle_dict(deg_dict,save_path,filename)   



















#%% the below are tests
TT = 'pk_aligned_unpred_rw'
region = 'LNacS'
mouse_name = 'FgDA_09'

temp = data[mouse_name][region]['signal_gcamp'][TT].copy()
for i in range(temp.shape[2]):

    temp = data[mouse_name][region]['signal_gcamp'][TT].copy()
    trial_num = sum(~np.isnan(temp[:,1,i]))
    if i == 0 :
        a = temp[:trial_num,:,i]
        a[:,0:10] = np.full([trial_num,10],i*5)
        init_mat = a
    else:
        a = temp[:trial_num,:,i]
        a[:,0:10] = np.full([trial_num,10],i*5)
        init_mat = np.vstack((init_mat,a))
plt.figure()
# plt.setp(ax, xticks=np.arange(0,300,1), xticklabels=np.arange(0,len(all_days),1),
# plt.savefig("{0}/heatmap_of_trials_{1}_{2}.png".format(savepath,TT,mouse_name), bbox_inches="tight", dpi = 200)

sns.heatmap(init_mat)
#%%
save_path = path+'/combined'
filename = 'corrected_gcamp_data_deg_zscore_by_ITI'
pickle_dict(data,save_path,filename) 
#%%
data = load_pickleddata('D:/PhD/Photometry/DATA/photo-pickles/combined/corrected_gcamp_data_C_zscore_by_ITI.pickle')


#%%
data = load_pickleddata('D:/PhD/Photometry/DATA/photo-pickles/combined/corrected_gcamp_data_deg_zscore_by_ITI.pickle')
#%%
#test about water alignemnt
trialtype ='unpred_water'
mouse_id = 'FgDA_01'
for trial in range(3,15):
    index = 5
    plt.figure()
    tp = data[mouse_id]['lickings']['first_lick_after_water'][trialtype][index][trial]
    licks = data[mouse_id]['lickings']['licking_data'][trialtype][index].values[trial]
    signal = data[mouse_id]['LNacS']['signal_gcamp']['unpred_water'][trial,:,index]
    plt.plot(np.asarray(licks),np.ones(len(licks)),'o',markersize = 1)
    plt.plot(tp,1,'ro',markersize = 1)
    plt.axvline(x = 5.5)
    plt.plot(np.arange(0,9,0.05),signal)
# if not np.isnan(tp):
#     rise_index = np.where(np.diff(values1[108:140])>1)[0][0]+108
#     lick_time = rise_index/20
#     idx = np.searchsorted(licks, lick_time, side='left') - 1

#     # Get the value in the array that is less than the given value
#     closest_val = licks[idx]
#     if trialtype == 'go':
#         data[mouse_id][site]['signal_gcamp']['lk_aligned_go_rw'][trial,:,index] = values1[int(closest_val*20)-20:int(closest_val*20)+30]- np.nanmean(values1[int(closest_val*20)-20:int(closest_val*20)])        
#     elif trialtype == 'c_reward':
#             data[mouse_id][site]['signal_gcamp']['lk_aligned_c_rw'][trial,:,index] = values1[int(closest_val*20)-20:int(closest_val*20)+30]- np.nanmean(values1[int(closest_val*20)-20:int(closest_val*20)])
#     elif trialtype == 'unpred_water':
#             data[mouse_id][site]['signal_gcamp']['lk_aligned_unpred_rw'][trial,:,index] = values1[int(closest_val*20)-20:int(closest_val*20)+30]- np.nanmean(values1[int(closest_val*20)-20:int(closest_val*20)])


#%%
TT = 'pk_aligned_unpred_rw'
region = 'LNacS'
mouse_name = 'FgDA_01'

temp = data[mouse_name][region]['signal_gcamp'][TT].copy()
for i in range(temp.shape[2]):

    temp = data[mouse_name][region]['signal_gcamp'][TT].copy()
    trial_num = sum(~np.isnan(temp[:,1,i]))
    if i == 0 :
        a = temp[:trial_num,:,i]
        a[:,0:10] = np.full([trial_num,10],i*5)
        init_mat = a
    else:
        a = temp[:trial_num,:,i]
        a[:,0:10] = np.full([trial_num,10],i*5)
        init_mat = np.vstack((init_mat,a))
plt.figure()
# plt.setp(ax, xticks=np.arange(0,300,1), xticklabels=np.arange(0,len(all_days),1),
# plt.savefig("{0}/heatmap_of_trials_{1}_{2}.png".format(savepath,TT,mouse_name), bbox_inches="tight", dpi = 200)

sns.heatmap(init_mat)


