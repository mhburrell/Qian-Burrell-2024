# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:10:12 2023

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
import seaborn as sns
#%% creatingheatmap for each of individual mouse
savepath = 'D:/PhD/Data_Code_Contingency_Uchida/Figures'
data = load_pickleddata('D:/PhD/Data_Code_Contingency_Uchida/Photometry/pickles/corrected_gcamp_data_deg_zscore_by_ITI.pickle')

#%%
#FgDA_07
# TT = 'lk_aligned_unpred_rw'
TT_dict = {'no_go':{'vmin':-4,'vmax' : 6},'go':{'vmin':-5,'vmax' : 12},'lk_aligned_unpred_rw':{'vmin':-5,'vmax' : 12}}
region = 'LNacS'
# no-go (-1)-10; go and unpred rw -1-20, 
mice= ['FgDA_08']

phase = {'cond':[0,1,2,3,4],
           'deg':[5,6,7,8,9],
 # 'c':[5,6,7,8,9],
           # 'rec':[10,11,12],
           # 'ext':[13,14,15]
         }
TT = 'go'
# TT = 'pk_aligned_unpred_rw'
# mice = ['FgDA_08'] 


vmax = 20
vmin = -5
for condition in phase.keys():
    sns.color_palette("vlag", as_cmap=True)
    session = phase[condition]
    for mouse_name in mice:
        # mouse_name = 'FgDA_C7'
        temp = data[mouse_name][region]['signal_gcamp'][TT].copy()
        cum_trialnum = [0]
        for i,ses in enumerate(session): #temp.shape[2]
        
            temp = data[mouse_name][region]['signal_gcamp'][TT].copy()
            trial_num = sum(~np.isnan(temp[:,1,ses]))
            cum_trialnum.append(trial_num)
            if i == 0 :
                a = temp[:trial_num,:,ses]
                a[:,0:10] = np.full([trial_num,10],i*vmax/len(session)) #这个根据不同需求改
                init_mat = a
            else:
                a = temp[:trial_num,:,ses]
                a[:,0:10] = np.full([trial_num,10],i*vmax/len(session))
                init_mat = np.vstack((init_mat,a))
        
        plt.subplots(figsize  =(5.5,3))
        plt.title(mouse_name+TT+condition)
        # plt.setp(ax, xticks=np.arange(0,300,1), xticklabels=np.arange(0,len(all_days),1),
        
        
        sns.heatmap(init_mat,vmin = vmin,vmax = vmax, cmap = 'coolwarm',center=0) 
        plt.xticks(np.arange(20,180,20),np.arange(-1,7,1),rotation =0)
        plt.yticks(np.cumsum(cum_trialnum[:-1]),np.arange(len(cum_trialnum)-1))
        plt.xlabel('Time to odor(s)')
        plt.savefig("{0}/{3}_{1}_{2}.png".format(savepath,TT,mouse_name,condition), bbox_inches="tight", dpi = 300)
        plt.savefig("{0}/{3}_{1}_{2}.pdf".format(savepath,TT,mouse_name,condition), bbox_inches="tight", dpi = 300)
        plt.show()
