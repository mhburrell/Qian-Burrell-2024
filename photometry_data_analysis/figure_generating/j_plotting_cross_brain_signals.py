# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 21:21:17 2022

@author: qianl
analysis between regions
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
from a1_parse_data_v2 import Mouse_data
from a1_parse_data_v2 import pickle_dict
from a1_parse_data_v2 import load_pickleddata



region_data = load_pickleddata('D:/PhD/Data_Code_Contingency_Uchida/Photometry/pickles/region_based_data_licking_filtered_deg.pickle')
good_regions = load_pickleddata('D:/PhD/Data_Code_Contingency_Uchida/Photometry/pickles/good_regions.pickle')
phase_dict = {'cond_days':[0,1,2,3,4],
              'deg_days':[5,6,7,8,9],
              'rec_days':[10,11,12,13],
              'ext_days':[14,15,16],
              'finalrec_days':[17,18] }    
#%%
def average_signal_by_trial(multiregion_dict ,region,types,phase,length,num_mouse,session_num,
                            session_of_phase = None,mouse_index = None,rm_bad = True,rm_window = [105,130]):
    key = region
    ave_response_trace = np.ones([session_num,length,num_mouse]) #d1 session, d2 time series, d3 mouse, fill with full trace
    for mouse_id in range(num_mouse):
        for session_of_phase in range(len(phase_dict[phase])):
            x = multiregion_dict[key][types][mouse_id][:,:,phase_dict[phase][session_of_phase]]
            if rm_bad:
                x = remove_bad_trials(x,window = [rm_window[0],rm_window[1]])
            print(key,types,mouse_id)
            ave_response_trace[session_of_phase,:,mouse_id] = np.nanmean(x,axis = 0)
    return ave_response_trace
def remove_bad_trials(mat,window = [105,130],thres = 10):
    mat_window = mat[:,window[0]:window[1]]
    max_val = mat_window.max(axis = 1)
    ind = max_val>=thres
    return mat[ind,:]


length = 180
trialtypes_full = ['go']#'no_go','UnpredReward','go_omit','background']
average_trace_dict = {}
for phase in ['cond_days','deg_days']:
    average_trace_dict[phase] = {}
    for region in good_regions.keys():
        average_trace_dict[phase][region] = {}
        for types in trialtypes_full:
            num_mouse = len(region_data[region]['go'])
            session_num = len(phase_dict[phase])
            if types in ['go','UnpredReward']:
                average_mat = average_signal_by_trial(region_data,region,types,
                                              phase,length,num_mouse,session_num,
                                              rm_bad = False,rm_window = [105,130]) 
            else: 
                average_mat = average_signal_by_trial(region_data,region,types,
                                              phase,length,num_mouse,session_num,
                                              rm_bad = False) 
            # # normalizer always based on go trial
            # go_mat = average_signal_by_trial(region_data,region,'go_omit',
            #                               'cond_days',length,num_mouse,5,
            #                               ) 
            normalizer_peak = np.mean(np.nanmax(average_mat,axis = 1),axis = 0)
            new_mat = average_mat.copy()
            for i in range(session_num):
                for j in range(num_mouse):
                # ax[i].plot(np.nanmean(a[i,:,:],axis =1))
                    new_mat[i,:,j] = average_mat[i,:,j]/normalizer_peak[j]*np.mean(normalizer_peak)
            average_trace_dict[phase][region][types] = new_mat
#%% between regions
#average_trace_dict[phase][region][types]
# 
ttype = 'go'
regions = good_regions.keys()
phases = ['cond_days','deg_days']
fig,ax = plt.subplots(len(regions),len(phases) ,figsize = (8,15))
for i,region in enumerate(regions):
    for j,phase in enumerate(phases):
        if ttype == 'go' and phase == 'ext_days':
            ax[i,j].plot(np.nanmean(average_trace_dict[phase][region]['go_omit'][-1,:,:],axis = 1))
        else:
            ax[i,j].plot(np.nanmean(average_trace_dict[phase][region][ttype][-1,:,:],axis = 1))
    
        ax[i,j].spines['top'].set_visible(False)
        ax[i,j].spines['right'].set_visible(False)
        ax[i,j].spines['bottom'].set_visible(False)
        ax[i,j].set_xticks([])
        ax[i,j].spines['left'].set_visible(False)
        ax[i,j].set_ylim([-2,15])
        if j == 0:
            ax[i,j].spines['left'].set_visible(True)
            ax[i,j].set_ylabel(region,fontsize=15)
            ax[i,j].set_yticks(np.array([0,4,8,12]))
            ax[i,j].set_yticklabels(np.array([0,4,8,12]),fontsize=13)
        else:
            ax[i,j].set_yticks([])

        if i == len(regions)-1:
            ax[i,j].spines['bottom'].set_visible(True)
            

            ax[i,j].set_xticks(np.arange(0,180,40))
            
            ax[i,j].set_xticklabels(np.arange(-2,7,2), fontsize=13)

plt.savefig('conditioning and degradation across 6 regions.pdf', dpi = 300)
plt.show()    
    
#%%

ttype = 'go'
phase = 'cond_days'
regions = average_trace_dict[phase].keys()

fig,ax = plt.subplots(len(regions),len(phase_dict[phase]),figsize = (12,8))
for i,region in enumerate(regions):
    for j in range(len(phase_dict[phase])):
        
        ax[i,j].plot(np.nanmean(average_trace_dict[phase][region][ttype][j,:,:],axis = 1))
    
        ax[i,j].spines['top'].set_visible(False)
        ax[i,j].spines['right'].set_visible(False)
        ax[i,j].spines['bottom'].set_visible(False)
        ax[i,j].set_xticks([])
        ax[i,j].spines['left'].set_visible(False)
        ax[i,j].set_ylim([-2,8])
        if j == 0:
            ax[i,j].spines['left'].set_visible(True)
            ax[i,j].set_ylabel(region,fontsize=15)
            ax[i,j].set_yticks(np.array([0,4,8]))
            ax[i,j].set_yticklabels(np.array([0,4,8]),fontsize=13)
        else:
            ax[i,j].set_yticks([])

        if i == len(regions)-1:
            ax[i,j].spines['bottom'].set_visible(True)
            ax[i,j].set_xlabel(f'session {j+1}',fontsize=15)

            ax[i,j].set_xticks(np.arange(0,180,40))
            
            ax[i,j].set_xticklabels(np.arange(-2,7,2), fontsize=13)
# plt.savefig('conditioning all sessions across 6 regions.pdf', dpi = 300)
plt.show()     
    
    
#%%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

ttype = 'go'
phase = 'cond_days'
regions = average_trace_dict[phase].keys()
plt.figure(figsize = (6,5))
mat = np.zeros([6,180])
for i,region in enumerate(regions):
        
    mat[i,:] = np.nanmean(average_trace_dict[phase][region][ttype][4,:,:],axis = 1)   
    
    
# Compute pairwise cosine similarity
cosine_similarities = cosine_similarity(mat)

# Create a heatmap
sns.heatmap(cosine_similarities, annot=True, cmap='plasma', vmin = 0, vmax = 1)
plt.xticks(np.arange(mat.shape[0])+0.5,good_regions.keys())
plt.yticks(np.arange(mat.shape[0])+0.5,good_regions.keys())
# Show the plot
# plt.savefig('similarity across session session 5 degradation.pdf', dpi = 300)
plt.show()


#%% k mean cluster

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

ttype = 'go'
phase = 'cond_days'
regions = average_trace_dict[phase].keys()
# Example data: 6 samples with 3 features each
# You should replace this with your actual data
mat = np.zeros([6,180*5])
for i,region in enumerate(regions):
    for j in range(5):
        mat[i,j*180:(j+1)*180] = np.nanmean(average_trace_dict[phase][region][ttype][j,:,:],axis = 1)  

# Apply k-means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=0).fit(mat)
# Annotate each point with the corresponding site label
data = np.zeros([6,180])
for i,region in enumerate(regions):
        
    data[i,:] = np.nanmean(average_trace_dict[phase][region][ttype][4,:,:],axis = 1)   
for i, x in enumerate(data):
    plt.text(x[46], x[116], f'Site {i+1}', fontsize=9)
# Print the cluster labels for each site
print("Cluster labels for the brain sites:", kmeans.labels_)

# Optional: Visualize the clusters (assuming 2D data for simplicity; adapt as needed)
plt.scatter([x[46] for x in data], [x[116] for x in data], c=kmeans.labels_, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering of Brain Sites')
plt.show()


#%%
for i in range(40,50):
    print(data[0,i])


    
    
    
    
    
    
    
    
    
    