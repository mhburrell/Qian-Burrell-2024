# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:08:35 2023

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
import datetime
from datetime import date
import sys
sys.path.insert(0,os.path.join('D:/PhD/Data_Code_Contingency_Uchida/Behavior','functions'))
from a1_parse_data_v2 import Mouse_data
from a1_parse_data_v2 import pickle_dict
from a1_parse_data_v2 import load_pickleddata
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
#%% I want to plot trends of odor response and water response

group = 'c_group'
region = 'LNacS'
trialtypes = ['go','']#['pk_aligned_unpred_rw'] # deg
# trialtypes = ['go','no_go','go_omit','background','c_reward','c_omit',  #c
#               ]
for types in trialtypes:
    for phase in ['cond_days','deg_days']:#phase_dict.keys():
        
        region_group_data = region_data[group]
        num_mouse = len(region_group_data[region][types])
        session_num = len(phase_dict[phase])
        length = 180 #180
        
        rm_bad = False          

        average_mat = average_signal_by_trial(region_group_data,region,types,
                                              phase,length,num_mouse,session_num,rm_bad = rm_bad)   
        
        new_mat = average_mat.copy()

        
        fig,ax = plt.subplots(1,1,figsize = (3,3))
        plt.title(region+'_'+types+'_'+phase)
        mean_odor = []
        mean_water = []
        std_odor = []
        std_water = []
        

        for i in range(session_num):
            
            if types in ["go",'go_omit']:  
                mean_odor.append(np.nanmean(np.max(new_mat[i,40:50,:],axis =0))) # intial activation
                mean_water.append(np.nanmean(np.max(new_mat[i,110:120,:],axis =0)))
                std_odor.append(np.nanstd(np.max(new_mat[i,40:50,:,],axis =0))/np.sqrt(num_mouse))
                std_water.append(np.nanstd(np.max(new_mat[i,110:120,:],axis =0))/np.sqrt(num_mouse))
            elif types == 'no_go':
                mean_odor.append(np.nanmean(np.max(new_mat[i,40:50,:],axis =0)))
                mean_water.append(np.nanmean(np.min(new_mat[i,45:60,:],axis =0)))
                std_odor.append(np.nanstd(np.max(new_mat[i,40:50,:,],axis =0))/np.sqrt(num_mouse))
                std_water.append(np.nanstd(np.min(new_mat[i,45:60,:],axis =0))/np.sqrt(num_mouse))
            elif types in ['pk_aligned_unpred_rw','unpred_water','lk_aligned_unpred_rw']:
                #auc
                mean_water.append(np.nanmean(np.sum(new_mat[i,40:60,:],axis =0)))

                std_water.append(np.nanstd(np.sum(new_mat[i,40:60,:],axis =0))/np.sqrt(num_mouse))
            
        # plt.yticks(np.arange(0,(session_num)*(-6),-6),np.arange(0,session_num))
        if types in ["go",'go_omit']:  
            plt.ylim([0,11])
            plt.xlim([-0.5,4.5])
            savepath = 'D:/PhD/Data_Code_Contingency_Uchida/Figures' 
        
        elif types == 'no_go':
            plt.ylim([-1.2,3])
            plt.xlim([-0.5,4.5])
            savepath = 'D:/PhD/Data_Code_Contingency_Uchida/Figures' 
        elif types in ['pk_aligned_unpred_rw','unpred_water','lk_aligned_unpred_rw']:
            
            # plt.ylim([6,11])
            plt.xlim([-0.5,4.5])
            savepath = 'D:/PhD/Data_Code_Contingency_Uchida/Figures' 
        plt.xlabel('# session')
        x = np.arange(np.max((len(mean_water),len(mean_odor))))
        if len(mean_odor) != 0:
            
            plt.errorbar(x, mean_odor, yerr=std_odor, marker='o', linestyle='dashed',ecolor=None, elinewidth=None, capsize=None, capthick=None,label = 'initial activation' )
        if len(mean_water) != 0:    
            plt.errorbar(x, mean_water, yerr=std_water,  marker='o', linestyle='dashed',ecolor=None, elinewidth=None, capsize=None, capthick=None, label = 'inhibition')
        # plt.plot(x,np.max(new_mat[:,40:60,:],axis =1),color = 'grey',alpha = 0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # plt.legend()
            # ax[i].set_xlabel(f'{phase} {i+1}')
            
            # if i == 0:
            #     ax[i].set_ylabel('normalized DA signal(Z-score)')      
        
        # plt.savefig("{0}/{1}_{2}_{3}_{4}.png".format(savepath,region,phase,types,group), bbox_inches="tight", dpi = 300)
        # plt.savefig("{0}/{1}_{2}_{3}_{4}.pdf".format(savepath,region,phase,types,group), bbox_inches="tight", dpi = 300)
        plt.show()

#%% statistical test
import scipy.stats as stats
unpred_water_peaks  = np.max(new_mat[:,40:70,:],axis =1) # session by mice
F_statistic, p_value = stats.f_oneway(unpred_water_peaks[0,:], unpred_water_peaks[4,:])
print("F-statistic:", F_statistic)
print("P-value:", p_value)

#%% comparision between c group and deg group 1: get data from two groups
group = 'deg_group'
region = 'LNacS'
types = 'go' # deg
# trialtypes = ['go','no_go','go_omit','background','c_reward','c_omit',  #c
phase = 'deg_days'
region_group_data = region_data[group]
num_mouse = len(region_group_data[region][types])
session_num = len(phase_dict[phase])
length = 180 # water 80; otherwise 180
rm_bad = False          
average_mat_deg = average_signal_by_trial(region_group_data,region,types,
                                      phase,length,num_mouse,session_num,rm_bad = rm_bad) 
group = 'c_group'
types = 'go' # deg
# trialtypes = ['go','no_go','go_omit','background','c_reward','c_omit',  #c
phase = 'deg_days'
region_group_data = region_data[group]
num_mouse = len(region_group_data[region][types])
session_num = len(phase_dict[phase])   
average_mat_c = average_signal_by_trial(region_group_data,region,types,
                                      phase,length,num_mouse,session_num,rm_bad = rm_bad) 

#%% run statistical test
import scipy.stats as stats
session_num = 5
for ses_num in range(session_num):
    deg_response  = np.max(average_mat_deg[:,40:70,:],axis =1) # session by mice
    c_response = np.max(average_mat_c[:,40:70,:],axis =1)
    t_statistic, p_value = stats.ttest_ind(deg_response[ses_num,:], c_response[ses_num,:])
    print('degradation vs cue-C day{}'.format(ses_num+1))
    print("t-statistic:", t_statistic)
    print("P-value:", p_value)


#%% group comparison with dots
def create_dot_pot(data_list,group_name,color,savepath, savename):
    fig,ax = plt.subplots(figsize = (1.1*len(data_list),4))
    i = 0
    x = []
    y = []
    for data, name, c in zip(data_list, group_name, color):
        plt.scatter(np.ones(len(data))*i, data, facecolors='none', edgecolors= c, alpha = 1)
        # plt.scatter(i,np.nanmean(data),fa.cecolors='none', edgecolors= 'k',lw = 2)
        plt.errorbar(i, np.nanmean(data), np.std(data)/np.sqrt(len(data)),color = 'grey',lw = 2.5,capsize=6,markeredgewidth=2.5)
        
        x.append(list(np.ones(len(data))*i))
        y.append(list(data))

        if i%2 == 1:
            plt.plot(x,y,color = 'grey',alpha = 0.2)
            x = []
            y = []
        i += 1
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    plt.xlim([-1,len(data_list)])
    plt.ylim([-0.5,17])
    plt.xticks(np.arange(len(data_list)),group_name,rotation=45,ha = 'right')  
    
    plt.ylabel('licking rate (s)')
    plt.xlabel('odor contingency')
    plt.savefig("{0}/{1}_dot_{2}.png".format(savepath,savename,date.today()), bbox_inches="tight", dpi = 300)
    plt.savefig("{0}/{1}_dot_{2}.pdf".format(savepath,savename,date.today()), bbox_inches="tight", dpi = 300)
    
    plt.show()
def pairwise_ttest(data_list):
    for i in range(len(data_list)-1):
        for j in range(i+1,len(data_list)):
            t_statistic, p_value = stats.ttest_ind(data_list[i],data_list[j])
            print(f'Between {i} and {j}:',"t-statistic:", t_statistic, "P-value:", p_value)

def gen_data(region_data, group = 'c_group', region = 'LNacS', phase = 'deg_days',types = 'go', window = [40,50],session_id=4):
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
    output = np.max(average_mat[session_id,window[0]:window[1],:],axis =0)
    return output


deg_session10 = gen_data(region_data, group = 'deg_group',phase = 'deg_days', region = 'LNacS', types = 'go', 
                         window = [40,50],session_id=4)

c_session10 = gen_data(region_data, group = 'c_group',phase = 'c_odor_days', region = 'LNacS', types = 'go', 
                         window = [40,50],session_id=4)
deg_session5 = gen_data(region_data, group = 'deg_group',phase = 'cond_days', region = 'LNacS', types = 'go', 
                         window = [40,50],session_id=4)

c_session5 = gen_data(region_data, group = 'c_group',phase = 'cond_days', region = 'LNacS', types = 'go', 
                         window = [40,50],session_id=4)

savepath = 'D:/PhD/Data_Code_Contingency_Uchida/Figures'
create_dot_pot([deg_session5, deg_session10,  c_session5, c_session10],
               ['deg bef','deg after','c bef','c after'],
               ['#ff8143','#fcb85d','#7570b3','#8da0cb',],
               savepath,savename = 'deg and c group photometry before after degradation compare')
#%% two sided t test
pairwise_ttest([deg_session5, deg_session10,  c_session5, c_session10])


#%% paired t test

t_statistic, p_value = stats.ttest_rel(deg_session10, deg_session5)
print('between deg cond 5 and deg 5')
print("t-statistic:", t_statistic)
print("P-value:", p_value)

t_statistic, p_value = stats.ttest_rel(c_session10, c_session5)
print('between c cond 5 and deg 5')
print("t-statistic:", t_statistic)
print("P-value:", p_value)

deg_session1 = gen_data(region_data, group = 'deg_group',phase = 'cond_days', region = 'LNacS', types = 'go', 
                         window = [40,50],session_id=1)
c_session1 = gen_data(region_data, group = 'c_group',phase = 'cond_days', region = 'LNacS', types = 'go', 
                         window = [40,50],session_id=1)
t_statistic, p_value = stats.ttest_rel(np.concatenate((deg_session5,c_session5)), np.concatenate((deg_session1,c_session1)))
print('between deg+c cond 1 and cond 5')
print("t-statistic:", t_statistic)
print("P-value:", p_value)



#%% for predicted reward
# run statistical test
import scipy.stats as stats
session_num = 5
for ses_num in range(session_num):
    deg_response  = np.max(average_mat_deg[:,110:120,:],axis =1) # session by mice
    c_response = np.max(average_mat_c[:,110:120,:],axis =1)
    t_statistic, p_value = stats.ttest_ind(deg_response[ses_num,:], c_response[ses_num,:])
    print('degradation vs cue-C day{}'.format(ses_num+1))
    print("t-statistic:", t_statistic)
    print("P-value:", p_value)


#%% group comparison with dots
def create_dot_pot(data_list,group_name,color,savepath, savename):
    fig,ax = plt.subplots(figsize = (1.1*len(data_list),4))
    i = 0
    x = []
    y = []
    for data, name, c in zip(data_list, group_name, color):
        plt.scatter(np.ones(len(data))*i, data, facecolors='none', edgecolors= c, alpha = 1)
        # plt.scatter(i,np.nanmean(data),fa.cecolors='none', edgecolors= 'k',lw = 2)
        plt.errorbar(i, np.nanmean(data), np.std(data)/np.sqrt(len(data)),color = 'grey',lw = 2.5,capsize=6,markeredgewidth=2.5)
        
        x.append(list(np.ones(len(data))*i))
        y.append(list(data))

        if i%2 == 1:
            plt.plot(x,y,color = 'grey',alpha = 0.2)
            x = []
            y = []
        i += 1
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    plt.xlim([-1,len(data_list)])
    plt.ylim([-0.5,17])
    plt.xticks(np.arange(len(data_list)),group_name,rotation=45,ha = 'right')  
    
    plt.ylabel('licking rate (s)')
    plt.xlabel('odor contingency')
    plt.savefig("{0}/{1}_dot_{2}.png".format(savepath,savename,date.today()), bbox_inches="tight", dpi = 300)
    plt.savefig("{0}/{1}_dot_{2}.pdf".format(savepath,savename,date.today()), bbox_inches="tight", dpi = 300)
    
    plt.show()
def pairwise_ttest(data_list):
    for i in range(len(data_list)-1):
        for j in range(i+1,len(data_list)):
            t_statistic, p_value = stats.ttest_ind(data_list[i],data_list[j])
            print(f'Between {i} and {j}:',"t-statistic:", t_statistic, "P-value:", p_value)

def gen_data(region_data, group = 'c_group', region = 'LNacS', phase = 'deg_days',types = 'go', window = [40,50],session_id=4):
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
    output = np.max(average_mat[session_id,window[0]:window[1],:],axis =0)
    return output


deg_session10 = gen_data(region_data, group = 'deg_group',phase = 'deg_days', region = 'LNacS', types = 'go', 
                         window = [110,120],session_id=4)

c_session10 = gen_data(region_data, group = 'c_group',phase = 'c_odor_days', region = 'LNacS', types = 'go', 
                         window = [110,120],session_id=4)
deg_session5 = gen_data(region_data, group = 'deg_group',phase = 'cond_days', region = 'LNacS', types = 'go', 
                         window = [110,120],session_id=4)

c_session5 = gen_data(region_data, group = 'c_group',phase = 'cond_days', region = 'LNacS', types = 'go', 
                         window = [110,120],session_id=4)

savepath = 'D:/PhD/Data_Code_Contingency_Uchida/Figures'
create_dot_pot([deg_session5, deg_session10,  c_session5, c_session10],
               ['deg bef','deg after','c bef','c after'],
               ['#ff8143','#fcb85d','#7570b3','#8da0cb',],
               savepath,savename = 'deg and c group photometry before after degradation compare')
#%% two sided t test
pairwise_ttest([deg_session5, deg_session10,  c_session5, c_session10])


#%% paired t test

t_statistic, p_value = stats.ttest_rel(deg_session10, deg_session5)
print('between deg cond 5 and deg 5')
print("t-statistic:", t_statistic)
print("P-value:", p_value)

t_statistic, p_value = stats.ttest_rel(c_session10, c_session5)
print('between c cond 5 and deg 5')
print("t-statistic:", t_statistic)
print("P-value:", p_value)

deg_session1 = gen_data(region_data, group = 'deg_group',phase = 'cond_days', region = 'LNacS', types = 'go', 
                         window = [40,50],session_id=1)
c_session1 = gen_data(region_data, group = 'c_group',phase = 'cond_days', region = 'LNacS', types = 'go', 
                         window = [40,50],session_id=1)
t_statistic, p_value = stats.ttest_rel(np.concatenate((deg_session5,c_session5)), np.concatenate((deg_session1,c_session1)))
print('between deg+c cond 1 and cond 5')
print("t-statistic:", t_statistic)
print("P-value:", p_value)




















