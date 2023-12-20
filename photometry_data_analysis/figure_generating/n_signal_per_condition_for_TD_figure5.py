# -*- coding: utf-8 -*-
"""
Created on Mon May 29 20:14:18 2023

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
sys.path.insert(0,os.path.join('D:/PhD/Data_Code_Contingency_Uchida/Behavior','functions'))
from a1_parse_data_v2 import Mouse_data
from a1_parse_data_v2 import pickle_dict
from a1_parse_data_v2 import load_pickleddata
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
#%% plot stacked response for TD model figure
# extract data

trialtypes = ['go','no_go','go_omit']
for types in trialtypes:
    for phase in conditions:
        new_mat = conditions[phase][types]
        
        # plot
        fig,ax = plt.subplots(1,1,figsize = (8,5))
        plt.title(types+'_'+phase)
        colors = ['red', 'yellow']
        n_segments = 5
        
        # Create the custom colormap
        cmap = create_colormap(colors, n_segments)
        
        ax.fill_between([40,60], [10,10], [-1,-1],color = 'grey',edgecolor = None, alpha=0.2)
        ax.fill_between([110,111], [10,10], [-1,-1], color = 'blue', alpha=0.4,edgecolor = None,)  
        ax.axhline(y = 0, color = 'grey', linewidth = 1)
        for i in range(new_mat.shape[0]):
        
            aa = np.nanmean(new_mat[i,:,:],axis =1) 
            ax.plot(aa,alpha = 1,linewidth = 1, color = cmap(i),label = 'session {}'.format(i)) #subtract the baseline again
            # stde = np.nanstd(new_mat[i,:,:],axis = 1)/np.sqrt(new_mat[i,:,:].shape[1])
            # ax.fill_between(np.arange(0,length,1), aa-stde, aa+stde,alpha = 0.3,color = 'k')
         
            
            # ax[i].fill_between([40,60], [6,6], [-0.5,-0.5],color = 'purple',edgecolor = None, alpha=0.2)
            # ax[i].fill_between([110,114], [6,6], [-0.5,-0.5], color = 'blue', alpha=0.4,edgecolor = None,)
            # ax[i].axhline(y = 0, color = 'grey', linewidth = 1)
            # aa = np.nanmean(new_mat[i,:,:],axis =1)
            # ax[i].plot(aa,alpha = 0.8,color = 'k') #subtract the baseline again
            # stde = np.nanstd(new_mat[i,:,:],axis = 1)/np.sqrt(new_mat[i,:,:].shape[1])
            # ax[i].fill_between(np.arange(0,180,1), aa-stde, aa+stde,alpha = 0.3,color = 'k')
        # plt.yticks(np.arange(0,(session_num)*(-6),-6),np.arange(0,session_num))
        plt.xticks(np.arange(0,160,20),np.arange(-1,7,1))
        plt.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
            # ax[i].set_xlabel(f'{phase} {i+1}')
            
            # if i == 0:
            #     ax[i].set_ylabel('normalized DA signal(Z-score)')      
        savepath = 'D:/PhD/Data_Code_Contingency_Uchida/Figures/latest_overlapped_trace_forTDfigure'
        # plt.savefig("{0}/{1}_{2}.png".format(savepath,phase,types), bbox_inches="tight", dpi = 300)
        # plt.savefig("{0}/{1}_{2}.pdf".format(savepath,phase,types), bbox_inches="tight", dpi = 300)
        plt.show()
        
        
        
#%% plot dots

def get_data(new_mat,window, method):
    sel_window = new_mat[:,window[0]:window[1],:]
    num_mouse = new_mat.shape[2]
    if method == 'peak':
        mean = np.nanmean(np.max(sel_window,axis = 1), axis = 1)
        std = np.nanstd(np.max(sel_window,axis = 1), axis = 1)/np.sqrt(num_mouse)
        data = np.max(sel_window,axis = 1)

    elif method == 'dip':        
        mean = np.nanmean(np.min(sel_window,axis = 1), axis = 1)
        std = np.nanstd(np.min(sel_window,axis = 1), axis = 1)/np.sqrt(num_mouse)
        data = np.min(sel_window,axis = 1)
    elif method == 'auc':
        mean = np.nanmean(np.sum(sel_window,axis = 1), axis = 1)
        std = np.nanstd(np.sum(sel_window,axis = 1), axis = 1)/np.sqrt(num_mouse)
        data = np.sum(sel_window,axis = 1)
        
        
    return mean, std, data

trialtypes = ['no_go','go_omit_odor','go_omit_dip_auc', 'no_go_auc_500ms','go_reward']
for types in trialtypes:
    for phase in conditions:
        if types == 'no_go':
            mean, std, _ = get_data(conditions[phase][types],[40,60],method = 'peak')
        if types == 'no_go_auc_500ms':
            mean, std, _ = get_data(conditions[phase][types[:5]],[45,60],method = 'auc')
        if types == 'go_omit_odor':
            mean, std, _ = get_data(conditions[phase][types[:7]],[40,60],method = 'peak')
        if types == 'go_omit_dip_auc':
            mean, std, _ = get_data(conditions[phase][types[:7]],[115,140],method = 'auc')
        if types == 'go_reward':
            mean, std, _ = get_data(conditions[phase]['go'],[115,125],method = 'peak')
        
        fig,ax = plt.subplots(1,1,figsize = (2,5))
        plt.errorbar(np.arange(len(mean)), mean, yerr=std,  marker='o', ecolor=None, elinewidth=None, capsize=None, capthick=None)
        if types in ['go_omit_odor','go_reward']:
            plt.ylim([-2,10])
        elif types == 'no_go':
            plt.ylim([0,4])
        elif types == 'go_omit_dip_auc':
            plt.ylim([-30,0])
        else: 
            plt.ylim([-15,10])
        savepath = 'D:/PhD/Data_Code_Contingency_Uchida/Figures/quatify response for conditions'
        plt.savefig("{0}/{1}_{2}.png".format(savepath,phase,types), bbox_inches="tight", dpi = 300)
        plt.savefig("{0}/{1}_{2}.pdf".format(savepath,phase,types), bbox_inches="tight", dpi = 300)
        plt.show()


#%% run one way avona of fist and last session for the above figure

trialtypes = ['no_go','go_omit_odor','go_omit_dip', 'no_go_auc']
for types in trialtypes:
    for phase in conditions:
        if types == 'no_go':
            _, _, data = get_data(conditions[phase][types],[40,60],method = 'peak')
            
        if types == 'no_go_auc':
             _, _, data = get_data(conditions[phase][types[:5]],[45,60],method = 'auc')
        if types == 'go_omit_odor':
             _, _, data = get_data(conditions[phase][types[:7]],[40,60],method = 'peak')
        if types == 'go_omit_dip':
             _, _, data = get_data(conditions[phase][types[:7]],[115,140],method = 'auc')
        
        
        statistic, pvalue = stats.ttest_rel(data[-1,:][~np.isnan(data[-1,:])], data[0,:][~np.isnan(data[0,:])])
        print('--------------------'+types+'^^^'+phase+'--------------------')
        print("Statistic:", statistic, ", p-value:", pvalue)


















