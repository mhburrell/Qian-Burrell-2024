# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 18:36:00 2022

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

#%%

# mice = ['FgDA_01','FrgD1_01','FrgD2_01','FrgD2_02']
# mice = ['FgDA_02','FgDA_03']
mice = ['FgDA_01','FgDA_02','FgDA_03',
        'FgDA_04','FgDA_05','FgDA_06',
        'FgDA_07','FgDA_08','FgDA_09',]

path = 'D:/PhD/Photometry/DATA/photo-pickles'

trialtypes = ['go', 'no_go', 'go_omit', 'background','unpred_water']

data = {}


# get trials


for mouse_id in mice:
    print(mouse_id)
    load_path = os.path.join(path,'{0}_stats.pickle'.format(mouse_id))
    mouse = load_pickleddata(load_path)
    
    #event plot with trials and iscorerct data
    
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
        for trialtype in trialtypes:
            data[mouse_id][site]['signal_gcamp'][trialtype] = np.full([160,180,len(all_days)], np.nan)
            data[mouse_id]['lickings']['first_lick_after_water'][trialtype] = []
            data[mouse_id]['lickings']['licking_data'][trialtype] = []
        
        for index in range(len(all_days)):
            day = all_days[index] 
            dataframe = mouse_trials[day]['dataframe'].copy()
            whole_session_corrected_signal = []
            for trialtype in trialtypes:
                
                is_x_TT = dataframe['Trialtype'] == trialtype #or 'go_omit' # index of go trials
                signal_xTT = dataframe[is_x_TT]
                num_trial = len(signal_xTT[site].values)
                
                try: 
                    concat_func_signal_xTT = np.concatenate(signal_xTT[site].values)
                    concat_isos_signal_xTT = np.concatenate(signal_xTT[site+'_isos'].values)
                    # remove nan value
                    
                    ind_nan = np.isnan(concat_func_signal_xTT) + np.isnan(concat_isos_signal_xTT)
                    concat_func_signal_xTT = concat_func_signal_xTT[~ind_nan]
                    concat_isos_signal_xTT = concat_isos_signal_xTT[~ind_nan]
                    
                    reg = LinearRegression().fit(concat_isos_signal_xTT.reshape(-1,1),concat_func_signal_xTT.reshape(-1,1))
                    beta1 = reg.coef_[0]
                    beta0 = reg.intercept_[0]
                
                
                
                    for trial in range(num_trial):
                        # movement correction
                        signal_subtract = beta0 + beta1*signal_xTT[site+'_isos'].values[trial]
                        signal_corrected = signal_xTT[site].values[trial] - signal_subtract
                        whole_session_corrected_signal += list(signal_corrected)
                except:
                    pass
            
            std_session = np.nanstd(np.asarray(whole_session_corrected_signal)) 
            
            for trialtype in trialtypes:
                
                is_x_TT = dataframe['Trialtype'] == trialtype #or 'go_omit' # index of go trials
                signal_xTT = dataframe[is_x_TT]
                num_trial = len(signal_xTT[site].values)
                
                # A BLOCK FOR LICKING ALIGNMENT 
                if num_trial == 0:
                    data[mouse_id]['lickings']['first_lick_after_water'][trialtype].append(None)
                    data[mouse_id]['lickings']['licking_data'][trialtype].append(None)   
                    
                else:
                    lickings = dataframe[is_x_TT].lickings
                    data[mouse_id]['lickings']['licking_data'][trialtype].append(lickings)   
                    if trialtype in ['go','c_reward','unpred_water']:
                        timepoint_1stlick_aftwater = []
                    
             
                        timepoint = 5.51
                        for key, val in lickings.items():
                            
                            valid_lick = [x for x in val if timepoint<=x <=timepoint+1.5]
                            if len(valid_lick) == 0:
                                timepoint_1stlick_aftwater.append(np.nan)
                            else:
                                timepoint_1stlick_aftwater.append(min(valid_lick))
      
                        
                        data[mouse_id]['lickings']['first_lick_after_water'][trialtype].append(timepoint_1stlick_aftwater)
                    else:
                        data[mouse_id]['lickings']['first_lick_after_water'][trialtype].append(None)
                        
                
                
                try: 
                    concat_func_signal_xTT = np.concatenate(signal_xTT[site].values)
                    concat_isos_signal_xTT = np.concatenate(signal_xTT[site+'_isos'].values)
                    # remove nan value
                    
                    ind_nan = np.isnan(concat_func_signal_xTT) + np.isnan(concat_isos_signal_xTT)
                    concat_func_signal_xTT = concat_func_signal_xTT[~ind_nan]
                    concat_isos_signal_xTT = concat_isos_signal_xTT[~ind_nan]
                    
                    reg = LinearRegression().fit(concat_isos_signal_xTT.reshape(-1,1),concat_func_signal_xTT.reshape(-1,1))
                    beta1 = reg.coef_[0]
                    beta0 = reg.intercept_[0]
                
                
                
                    for trial in range(num_trial):
                        # movement correction
                        signal_subtract = beta0 + beta1*signal_xTT[site+'_isos'].values[trial]
                        signal_corrected = signal_xTT[site].values[trial] - signal_subtract
                            # df/f
                        if np.nanstd(signal_corrected[10:30]) == 0:
                            data[mouse_id][site]['signal_gcamp'][trialtype][trial,:,index] = np.zeros([1,180])
                            
                        elif sum(np.isnan(signal_corrected)) == len(signal_corrected):
                            
                            pass
                        else: #minus pre-odor baseline and then zscored by the whole session
                            
                            values1 = (signal_corrected - np.nanmean(signal_corrected[0:40]))/std_session
                            data[mouse_id][site]['signal_gcamp'][trialtype][trial,:,index] = values1[1:181]
                except:
                    pass
   #%%         
save_path = path+'/combined'
filename = 'corrected_gcamp_data_deg'
pickle_dict(data,save_path,filename)           
#%% load existing data
folder_name = 'round_20221111-right-pilot_c-odor'
data = load_pickleddata('D:/PhD/Photometry/DATA/{}/processed/corrected_gcamp_data.pickle'.format(folder_name))
#%%
mouse_name = 'FgDA_07'

mouse = load_pickleddata('D:/PhD/Photometry/DATA/{}/processed/{}_stats.pickle'.format(folder_name,mouse_name))
group = mouse.group
#%%
import matplotlib.cm as cm
import matplotlib as mpl
rc = {"axes.spines.left" : False,
      "axes.spines.right" : False,
      "axes.spines.bottom" : False,
      "axes.spines.top" : False,
}
plt.rcParams.update(rc)

mpl.rcParams['lines.linewidth'] = 4

gap = 5
# mice = ['FgDA_01','FrgD1_01','FrgD2_01','FrgD2_02']
# mice = ['FgDA_09']
# mice = ['FrgD1_01']



    

sample_points = 180
water_point = int(sample_points/9*5.5)
odor_on_point = int(sample_points/9*2)
odor_off_point = int(sample_points/9*3)


trialtypes = ['go', 'no_go', 'go_omit', 'unpred_water']
# trialtypes = ['unpred_water']
for mouse_name in mice:
    mouse = load_pickleddata('D:/PhD/Photometry/DATA/photo-pickles/{}_stats.pickle'.format(mouse_name))
    all_days = mouse.all_days.copy()
    group = mouse.group
    
    if mouse_name != 'FgDA_09':
        cond_days = [0,1,2,3,4]
        deg_days = [5,6,7,8,9]
        rec_days = [10,11,12]
        ext_days = [13,14,15]
        finalrec_days = [16,17]
    else:
        cond_days = [0,1,2,3,4]
        deg_days = [5,6,7,8,9]
    
    
    for TT in trialtypes:
        if TT in ['go']:
            water_point = int(sample_points/9*5.5)
            odor_on_point = int(sample_points/9*2)
            odor_off_point = int(sample_points/9*3)
        elif TT in ['go_omit','no_go']:
            odor_on_point = int(sample_points/9*2)
            odor_off_point = int(sample_points/9*3)
            water_point = np.nan
        elif TT == 'unpred_water':
            water_point = int(sample_points/9*5.5)
            odor_on_point = np.nan
            odor_off_point = np.nan
        else:
            water_point = np.nan
            odor_on_point = np.nan
            odor_off_point = np.nan
            
        nsites =  len([key for key in data[mouse_name].keys()])-1  # remove lickings
        fig,ax = plt.subplots(nsites,5,sharex = True, sharey = True,figsize = (25,6*nsites))
        
        plt.setp(ax, xticks=np.arange(0,180,20), xticklabels=np.arange(0,9,1),
                yticks=np.arange(0,8,5))
        
        for num,site in enumerate([key for key in data[mouse_name].keys() if key != 'lickings']):
            
            across_day_signal_persite = np.nanmean(data[mouse_name][site]['signal_gcamp'][TT],axis = 0)
            if TT =='go':
                across_day_signal_persite_omit = np.nanmean(data[mouse_name][site]['signal_gcamp']['go_omit'],axis = 0)
            else:
                across_day_signal_persite_omit = np.nanmean(data[mouse_name][site]['signal_gcamp'][TT],axis = 0)
            if mouse_name != 'FgDA_09':
                for index,i in enumerate(cond_days):
                    ax[num,0].plot(across_day_signal_persite[:,i]+gap*index,color =cm.cool(index/float(5)),label = all_days[i])
                    ax[num,0].legend(frameon=False)
                    ax[num,0].set_ylim(-10,gap*5+20)
                    ax[num,0].set_title('conditioning',fontsize = 25)
                    ax[num,0].set_ylabel(site,fontsize = 35)
                    ax[num,0].axvline(x = odor_on_point,linewidth=0.2, color=(0, 0, 0, 0.75))
                    ax[num,0].axvline(x = odor_off_point,linewidth=0.2, color=(0, 0, 0, 0.75))
                    ax[num,0].axvline(x = water_point,linewidth=0.5, color=(0.17578125, 0.67578125, 0.83203125,1))
                for index,i in enumerate(deg_days):
                    ax[num,1].plot(across_day_signal_persite[:,i]+gap*index,color =cm.cool(index/float(5)),label = all_days[i])
                    ax[num,1].legend(frameon=False)
                    ax[num,1].set_title('degradation',fontsize = 20)
                    ax[num,1].axvline(x = odor_on_point,linewidth=0.2, color=(0, 0, 0, 0.75))
                    ax[num,1].axvline(x = odor_off_point,linewidth=0.2, color=(0, 0, 0, 0.75))
                    ax[num,1].axvline(x = water_point,linewidth=0.5, color=(0.17578125, 0.67578125, 0.83203125,1))                
                for index,i in enumerate(rec_days):
                    ax[num,2].plot(across_day_signal_persite[:,i]+gap*index,color =cm.cool(index/float(5)),label = all_days[i])
                    ax[num,2].legend(frameon=False)
                    ax[num,2].set_title('recovery',fontsize = 20)
                    ax[num,2].axvline(x = odor_on_point,linewidth=0.2, color=(0, 0, 0, 0.75))
                    ax[num,2].axvline(x = odor_off_point,linewidth=0.2, color=(0, 0, 0, 0.75))
                    ax[num,2].axvline(x = water_point,linewidth=0.5, color=(0.17578125, 0.67578125, 0.83203125,1))            
                for index,i in enumerate(ext_days):
                    ax[num,3].plot(across_day_signal_persite_omit[:,i]+gap*index,color =cm.cool(index/float(5)),label = all_days[i])
                    ax[num,3].legend(frameon=False)
                    ax[num,3].set_title('extinction',fontsize = 20)
                    ax[num,3].axvline(x = odor_on_point,linewidth=0.2, color=(0, 0, 0, 0.75))
                    ax[num,3].axvline(x = odor_off_point,linewidth=0.2, color=(0, 0, 0, 0.75))
                    ax[num,3].axvline(x = water_point,linewidth=0.5, color=(0.17578125, 0.67578125, 0.83203125,1))            
                for index,i in enumerate(finalrec_days):
                    ax[num,4].plot(across_day_signal_persite[:,i]+gap*index,color =cm.cool(index/float(5)),label = all_days[i])
                    ax[num,4].legend(frameon=False)
                    ax[num,4].set_title('final recovery',fontsize = 20)
                    ax[num,4].axvline(x = odor_on_point,linewidth=0.2, color=(0, 0, 0, 0.75))
                    ax[num,4].axvline(x = odor_off_point,linewidth=0.2, color=(0, 0, 0, 0.75))
                    ax[num,4].axvline(x = water_point,linewidth=0.5, color=(0.17578125, 0.67578125, 0.83203125,1))        
            else:
                for index,i in enumerate(cond_days):
                    ax[num,0].plot(across_day_signal_persite[:,i]+gap*index,color =cm.cool(index/float(5)),label = all_days[i])
                    ax[num,0].legend(frameon=False)
                    ax[num,0].set_ylim(-10,gap*5+40)
                    ax[num,0].set_title('conditioning',fontsize = 25)
                    ax[num,0].set_ylabel(site,fontsize = 35)
                    ax[num,0].axvline(x = odor_on_point,linewidth=0.2, color=(0, 0, 0, 0.75))
                    ax[num,0].axvline(x = odor_off_point,linewidth=0.2, color=(0, 0, 0, 0.75))
                    ax[num,0].axvline(x = water_point,linewidth=0.5, color=(0.17578125, 0.67578125, 0.83203125,1))
                for index,i in enumerate(deg_days):
                    ax[num,1].plot(across_day_signal_persite[:,i]+gap*index,color =cm.cool(index/float(5)),label = all_days[i])
                    ax[num,1].legend(frameon=False)
                    ax[num,1].set_title('degradation',fontsize = 20)
                    ax[num,1].axvline(x = odor_on_point,linewidth=0.2, color=(0, 0, 0, 0.75))
                    ax[num,1].axvline(x = odor_off_point,linewidth=0.2, color=(0, 0, 0, 0.75))
                    ax[num,1].axvline(x = water_point,linewidth=0.5, color=(0.17578125, 0.67578125, 0.83203125,1))           
                
        
        savepath = 'D:/PhD/Figures/photometry'.format(path.split('/')[-1])
        plt.savefig("{0}/{1}_{2}_{3}.png".format(savepath,'deg',mouse_name,TT), bbox_inches="tight", dpi = 72)
        plt.show()

































#%%
# DA analysis
# response to go odor and no go odor

mouse_name = 'FgDA_07'
good_sites = ['MNacS','PMOT','PLOT','LNacS','AMOT','ALOT']

odor_mat = np.full([len(good_sites),len(all_days)], np.nan)
ratio_mat = np.full([len(good_sites),len(all_days)], np.nan)
TT = 'go'
for k,site in enumerate(good_sites):
    for i in range(len(all_days)):
        if i in [14,15,16] and TT == 'go':
            across_day_mean = np.nanmean(data[mouse_name]['go_omit']['signal_gcamp'][site],axis = 0)
        else:
            across_day_mean = np.nanmean(data[mouse_name][TT]['signal_gcamp'][site],axis = 0)
    
        odor_peak = np.max(across_day_mean[:,i][40:70])-np.max(across_day_mean[:,i][35:40])
        water_peak = np.max(across_day_mean[:,i][105:125])-np.max(across_day_mean[:,i][95:100])
        ratio = (np.abs(water_peak)-np.abs(odor_peak))/np.abs(odor_peak)
    
        odor_mat[k,i] = odor_peak
        ratio_mat[k,i] = ratio
        
        
#%% plot ratio DA
rc = {"axes.spines.left" : True,
      "axes.spines.right" : False,
      "axes.spines.bottom" : True,
      "axes.spines.top" : False,
}
plt.rcParams.update(rc)
fig,ax = plt.subplots(figsize = (5,5))

for i in range(len(good_sites)):
    array = ratio_mat[i]
    array = np.insert(array,5,np.nan)
    array = np.insert(array,11,np.nan)
    array = np.insert(array,16,np.nan)
    array = np.insert(array,20,np.nan)
    plt.plot(array,'-o',label = good_sites[i],linewidth = 1,color = cm.Set2(i),alpha = 0.7)
plt.axhline(y = 0,linewidth = 1,color = 'grey')
plt.ylabel('(water response-odor response)/odor response (Zscore)')
plt.xlabel('# sessions')
plt.ylim([-2,14])
plt.xticks(ticks = np.arange(len(all_days)+3), 
           labels = ['cond1','cond2','cond3','cond4','cond5','','deg1','deg2','deg3','deg4','deg5','','rec1','rec2','rec3','pre-ext','','ext1','ext2','ext3','','frec1','frec2']
           ,rotation = 45)

plt.legend()
savepath = 'D:/PhD/Photometry/DATA/{}/figures'.format(folder_name)

# plt.savefig("{0}/{1}_water_vs_odor_ratio.png".format(savepath,mouse_name), bbox_inches="tight", dpi = 200)
plt.show()

#%% plot odor response  DA
fig,ax = plt.subplots(figsize = (6,7))

for i in range(len(good_sites)):
    array = odor_mat[i]
    array = np.insert(array,5,np.nan)
    array = np.insert(array,11,np.nan)
    array = np.insert(array,16,np.nan)
    array = np.insert(array,20,np.nan)
    plt.plot(array,'-o',label = good_sites[i],linewidth = 1,color = cm.Set2(i))
plt.axhline(y = 0,linewidth = 1,color = 'grey')
plt.ylabel('odor response (Zscore)')
plt.xlabel('# sessions')
plt.ylim([0,8])
plt.xticks(ticks = np.arange(len(all_days)+4), 
           labels = ['cond1','cond2','cond3','cond4','cond5','','deg1','deg2','deg3','deg4','deg5','','rec1','rec2','rec3','pre-ext','','ext1','ext2','ext3','','frec1','frec2'],
           rotation = 45)

plt.legend()
savepath = 'D:/PhD/Photometry/DATA/round_20220307/figures'

plt.savefig("{0}/{1}_odor_response.png".format(savepath,mouse_name), bbox_inches="tight", dpi = 200)
plt.show()

#%% plot correlation between brain regions
import seaborn as sns

matrix = {}
for site in good_sites:
    signals = []
    for i,day in enumerate(all_days):
        if i in [14,15,16] and TT == 'go':
            across_day_mean = np.nanmean(data[mouse_name]['go_omit']['signal_gcamp'][site],axis = 0)
        else:
            across_day_mean = np.nanmean(data[mouse_name][TT]['signal_gcamp'][site],axis = 0)
        signals.append(across_day_mean[1:,i])
    matrix[site] = np.concatenate(signals)
    
corr_mat = pd.DataFrame(matrix)    
plt.figure(figsize = (5,4))
sns.heatmap(corr_mat.corr(),annot=True)
plt.savefig("{0}/DA_brainregion_correlation.png".format(savepath), bbox_inches="tight", dpi = 200)
plt.show()


#%% cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
mouse_name = 'FgDA_01'
good_sites = ['AMOT','PMOT','ALOT','PLOT','MNacS','LNacS',]
all_days = mouse.all_days.copy()
matrix = {}
TT = 'go'
for site in good_sites:
    signals = []
    for i,day in enumerate(all_days):
        if i in [14,15,16] and TT == 'go':
            across_day_mean = np.nanmean(data[mouse_name]['go_omit']['signal_gcamp'][site],axis = 0)
        else:
            across_day_mean = np.nanmean(data[mouse_name][TT]['signal_gcamp'][site],axis = 0)
        signals.append(across_day_mean[1:,i])
    matrix[site] = np.concatenate(signals)

df = pd.DataFrame(matrix)
#%%
cosine_mat = cosine_similarity(df.T)
plt.figure(figsize = (5,4))
sns.heatmap(cosine_mat,annot=True,xticklabels=good_sites, yticklabels=good_sites)

savepath = 'D:/PhD/Photometry/DATA/round_20220307/figures'
plt.savefig("{0}/DA_brainregion_correlation_cosine2.png".format(savepath), bbox_inches="tight", dpi = 200)
plt.show()
#%%
task = ['cond1','cond2','cond3','cond4','cond5','deg1','deg2','deg3','deg4','deg5','rec1','rec2','rec3','pre-ext','ext1','ext2','ext3','finalrec1','finalrec2']
matrix = {}
for i,day in enumerate(all_days):
    signals = []
    
    for site in good_sites:
        if i in [14,15,16] and TT == 'go':
            across_day_mean = np.nanmean(data[mouse_name]['go_omit']['signal_gcamp'][site],axis = 0)
        else:
            across_day_mean = np.nanmean(data[mouse_name][TT]['signal_gcamp'][site],axis = 0)
        signals.append(across_day_mean[1:,i])
    matrix[task[i]] = np.concatenate(signals)
    
corr_mat = pd.DataFrame(matrix)    
plt.figure(figsize = (5,4))

sns.heatmap(corr_mat.corr(),cmap = 'coolwarm')
plt.savefig("{0}/DA_days_correlation_fulltrace.png".format(savepath), bbox_inches="tight", dpi = 200)
plt.show()

#%%
matrix = {}
for i,day in enumerate(all_days):
    signals = []
    
    for site in good_sites:
        if i in [14,15,16] and TT == 'go':
            across_day_mean = np.nanmean(data[mouse_name]['go_omit']['signal_gcamp'][site],axis = 0)
        else:
            across_day_mean = np.nanmean(data[mouse_name][TT]['signal_gcamp'][site],axis = 0)
        signals.append(across_day_mean[20:70,i])
    matrix[task[i]] = np.concatenate(signals)
    
corr_mat = pd.DataFrame(matrix)    
plt.figure(figsize = (5,4))

sns.heatmap(corr_mat.corr(),cmap = 'coolwarm')
plt.savefig("{0}/DA_days_correlation_odorpart.png".format(savepath), bbox_inches="tight", dpi = 200)
plt.show()

#%% heatmap of trials 先别save了
TT = 'unpred_water'
region = 'LNacS'
mouse_name = 'FgDA_09'
temp = data[mouse_name][region]['signal_gcamp'][TT]
for i in range(temp.shape[2]):

    temp = data[mouse_name][region]['signal_gcamp'][TT]
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
#%% waterfall of one session
mouse_name = 'FgDA_01'
temp = data[mouse_name]['go']['signal_gcamp']['AMOT']
i = 10
trial_num = sum(~np.isnan(temp[:,1,i]))
a = temp[:trial_num,:,i]

fig,ax = plt.subplots(figsize = (4,1))


sns.heatmap(a,ax = ax)
plt.setp(ax, yticks=np.arange(0,trial_num,10),yticklabels=np.arange(0,trial_num,10), xticks=np.arange(3,183,20),xticklabels=np.arange(0,9,1))
plt.show()
#%% othertrials
mouse_name = 'FrgD2_02'
for i in range(temp.shape[2]):
    
    temp = data[mouse_name]['go_omit']['signal_gcamp']['AMOT']
    
    trial_num = sum(~np.isnan(temp[:,1,i]))
    if i == 0 :
        a = temp[:trial_num,:,i]
        a[:,0:10] = np.full([trial_num,10],i*1)
        init_mat = a
    else:
        a = temp[:trial_num,:,i]
        a[:,0:10] = np.full([trial_num,10],i*1)
        init_mat = np.vstack((init_mat,a))


sns.heatmap(init_mat)





#%%

container1 = []
TT = 'go'
site = 'PLOT'
dates = [5,6,7,8,9]
temp = data[mouse_name][TT]['signal_gcamp'][site]
trial_num = sum(~np.isnan(temp[:,1,i]))
for t in range(trial_num):
    for j in dates:
        container1.append(np.max(temp[t,40:70,j]))

x = np.arange(len(container1))

container2 = []
TT = 'go'
site = 'PMOT'
dates = [5,6,7,8,9]
temp = data[mouse_name][TT]['signal_gcamp'][site]
trial_num = sum(~np.isnan(temp[:,1,i]))
for t in range(trial_num):
    for j in dates:
        container2.append(np.max(temp[t,40:70,j]))

x2 = np.arange(len(container2))








plt.scatter(x,container1)
plt.scatter(x2,container2)
























