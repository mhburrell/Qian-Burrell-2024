# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 19:20:43 2022

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
group = 'c_group'
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
            rm_bad = True            
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
    savepath = 'D:/PhD/Data_Code_Contingency_Uchida/Figures'
    plt.savefig("{0}/new_{1}_{2}_{3}_{4}.png".format(savepath,region,phase,types,group), bbox_inches="tight", dpi = 300)
    plt.savefig("{0}/new_{1}_{2}_{3}_{4}.pdf".format(savepath,region,phase,types,group), bbox_inches="tight", dpi = 300)
    plt.show()









#%% I want to plot trends of odor response and water response

group = 'deg_group'
region = 'LNacS'
trialtypes = ['pk_aligned_unpred_rw'] # deg
# trialtypes = ['go','no_go','go_omit','background','c_reward','c_omit',  #c
#               ]
for types in trialtypes:
    for phase in ['deg_days']:#phase_dict.keys():
        
        region_group_data = region_data[group]
        num_mouse = len(region_group_data[region][types])
        session_num = len(phase_dict[phase])
        length = 80 #180
        
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
        x = np.arange(np.max(len(mean_water),len(mean_odor)))
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
        
        plt.savefig("{0}/{1}_{2}_{3}_{4}.png".format(savepath,region,phase,types,group), bbox_inches="tight", dpi = 300)
        plt.savefig("{0}/{1}_{2}_{3}_{4}.pdf".format(savepath,region,phase,types,group), bbox_inches="tight", dpi = 300)
        plt.show()

#%% statistical test
import scipy.stats as stats
unpred_water_peaks  = np.max(new_mat[:,40:70,:],axis =1) # session by mice
F_statistic, p_value = stats.f_oneway(unpred_water_peaks[0,:], unpred_water_peaks[4,:])
print("F-statistic:", F_statistic)
print("P-value:", p_value)

#%%
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
    plt.ylim([-0.1,20])
    plt.xticks(np.arange(len(data_list)),group_name,rotation=45,ha = 'right')  
    
    plt.ylabel('licking rate (s)')
    plt.xlabel('odor contingency')
    plt.savefig("{0}/{1}_dot_{2}.png".format(savepath,savename,date.today()), bbox_inches="tight", dpi = 300)
    plt.savefig("{0}/{1}_dot_{2}.pdf".format(savepath,savename,date.today()), bbox_inches="tight", dpi = 300)
    
    plt.show()
def pairwise_anova(data_list):
    for i in range(len(data_list)-1):
        for j in range(i+1,len(data_list)):
            F_statistic, p_value = stats.f_oneway(data_list[i],data_list[j])
            print(f'Between {i} and {j}:',"F-statistic:", F_statistic, "P-value:", p_value)

def gen_data(region_data, group = 'c_group', region = 'LNacS', phase = 'deg_days',types = 'go', window = [40,50],session_id=4):
    region_group_data = region_data[group]
    num_mouse = len(region_group_data[region][types])
    session_num = len(phase_dict[phase])
    length = 180
    if types in ['go','unpred_water','c_reward']:
        rm_bad = True            
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
pairwise_anova([deg_session5, deg_session10,  c_session5, c_session10])










#%%
group = 'deg_group'
region = 'LNacS'
trialtypes = ['lk_aligned_unpred_rw'] # deg
# trialtypes = ['go','no_go','go_omit','background','c_reward','c_omit',  #c
#               ]
for types in trialtypes:
    for phase in phase_dict.keys():
        
        region_group_data = region_data[group]
        num_mouse = len(region_group_data[region][types])
        session_num = len(phase_dict[phase])
        length = 80
        
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
            

            mean_odor.append(np.nanmean(np.max(new_mat[i,40:50,:],axis =0))) # intial activation

            std_odor.append(np.nanstd(np.max(new_mat[i,40:50,:,],axis =0))/np.sqrt(num_mouse))



            
        # plt.yticks(np.arange(0,(session_num)*(-6),-6),np.arange(0,session_num))
        if types in ["go",'go_omit','lk_aligned_unpred_rw']:  
            plt.ylim([0,11])
            plt.xlim([-0.5,4.5])
            savepath = 'D:/PhD/Figures/photometry/odor reponse and water response' 
        
        elif types == 'no_go':
            plt.ylim([-1.2,3])
            plt.xlim([-0.5,4.5])
            savepath = 'D:/PhD/Figures/photometry/initial_resposne_and_inhibition' 
        plt.xlabel('# session')
        x = np.arange(len(mean_odor))
        plt.errorbar(x, mean_odor, yerr=std_odor, marker='o', linestyle='dashed',ecolor=None, elinewidth=None, capsize=None, capthick=None,label = 'initial activation' )
 
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # plt.legend()
            # ax[i].set_xlabel(f'{phase} {i+1}')
            
            # if i == 0:
            #     ax[i].set_ylabel('normalized DA signal(Z-score)')      
        
        # plt.savefig("{0}/{1}_{2}_{3}_{4}.png".format(savepath,region,phase,types,group), bbox_inches="tight", dpi = 300)
        # plt.savefig("{0}/{1}_{2}_{3}_{4}.pdf".format(savepath,region,phase,types,group), bbox_inches="tight", dpi = 300)
        plt.show()



#%% overlappend traces
from matplotlib.colors import LinearSegmentedColormap

# Create a custom colormap for the gradient
def create_colormap(colors, n_segments):
    return LinearSegmentedColormap.from_list('custom_cmap', colors, N=n_segments)

# phase_dict = {
#               'deg_days':[4,5,6,7,8,9],

#               'ext_days':[12,13,14,15],
#               'c_odor_days':[4,5,6,7,8,9],
#               }  

phase_dict = {
              'cond_days':[0,1,2,3,4],
              } 

# Define the gradient colors for the lines
colors = ['red', 'yellow']
n_segments = 5

# Create the custom colormap
cmap = create_colormap(colors, n_segments)

group = 'c_group'
region = 'LNacS'
trialtypes = ['go','no_go','go_omit','background','c_reward','c_omit'] # c
# trialtypes = ['go','no_go','go_omit','background','pk_aligned_unpred_rw'] #deg
for types in trialtypes:
    for phase in phase_dict.keys():
        
        region_group_data = region_data[group]
        num_mouse = len(region_group_data[region][types])
        session_num = len(phase_dict[phase])
        if types == 'pk_aligned_unpred_rw':
            length = 80
        else:
            length = 180
        if types in ['go','unpred_water','c_reward']:
            rm_bad = True            
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
        
        
        
        fig,ax = plt.subplots(1,1,figsize = (8,5))
        plt.title(region+'_'+types+'_'+phase)

        ax.fill_between([20,40], [10,10], [-1,-1],color = 'grey',edgecolor = None, alpha=0.2)
        ax.fill_between([90,91], [10,10], [-1,-1], color = 'blue', alpha=0.4,edgecolor = None,)  
        ax.axhline(y = 0, color = 'grey', linewidth = 1)
        for i in range(session_num):

            aa = np.nanmean(new_mat[i,20:,:],axis =1) 
            ax.plot(aa,alpha = 1,linewidth = 1, color = cmap(i),label = 'session {}'.format(i+4)) #subtract the baseline again
            # stde = np.nanstd(new_mat[i,:,:],axis = 1)/np.sqrt(new_mat[i,:,:].shape[1])
            # ax.fill_between(np.arange(0,180,1), aa-stde, aa+stde,alpha = 0.3,color = 'k')
 
            
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
        savepath = 'D:/PhD/Data_Code_Contingency_Uchida/Figures/average_overlapped'
        plt.savefig("{0}/{1}_{2}_{3}_{4}.png".format(savepath,region,phase,types,group), bbox_inches="tight", dpi = 300)
        plt.savefig("{0}/{1}_{2}_{3}_{4}.pdf".format(savepath,region,phase,types,group), bbox_inches="tight", dpi = 300)
        plt.show()











#%% plot each averaged signal across mice
fig,ax = plt.subplots(5,9,sharex = True, sharey = True)
for i in range(5):
    for j in range(9):
        ax[i,j].plot(new_mat[i,:,j])
#%% inspect unpred water response (heatmap)


mouse_ind = 8
region_data2 = region_data.copy()
temp = region_data2['deg_group']['LNacS']['lk_aligned_unpred_rw'][mouse_ind]
for i in range(temp.shape[2]):

    temp = region_data2['deg_group']['LNacS']['lk_aligned_unpred_rw'][mouse_ind]
    trial_num = sum(~np.isnan(temp[:,1,i]))
    if i == 0 :
        a = temp[:trial_num,:,i]
        a[:,0:10] = np.full([trial_num,10],i*5)
        a = remove_bad_trials(a,window = [40,50])
        init_mat = a
    else:
        a = temp[:trial_num,:,i]
        a[:,0:10] = np.full([trial_num,10],i*5)
        a = remove_bad_trials(a,window = [40,50])
        init_mat = np.vstack((init_mat,a))
plt.figure()
# plt.setp(ax, xticks=np.arange(0,300,1), xticklabels=np.arange(0,len(all_days),1),
# plt.savefig("{0}/heatmap_of_trials_{1}_{2}.png".format(savepath,TT,mouse_name), bbox_inches="tight", dpi = 200)

sns.heatmap(init_mat)





#%% plot last day conditioning and last day degradation




length = 180
trialtypes_full = ['go','no_go','go_omit','c_omit','c_reward']
groups = ['deg_group','c_group']
average_trace_dict = {}
for group in groups:
    average_trace_dict[group] = {}
    region_group_data = region_data[group]
    if group == 'deg_group':
        phase_dict = phase_dict = {'cond_days':[0,1,2,3,4],
              'deg_days':[5,6,7,8,9],
              'rec_days':[10,11,12],
              'ext_days':[13,14,15],
              'finalrec_days':[16,17],
              }  
    else:
        phase_dict = phase_dict = {'cond_days':[0,1,2,3,4],
              'c_odor_days':[5,6,7,8,9],
              }  
    
    for phase in phase_dict.keys():
        average_trace_dict[group][phase] = {}
        for region in good_regions.keys():
            average_trace_dict[group][phase][region] = {}
            for types in trialtypes_full:
                if group == 'deg_group':
                    num_mouse = len([x for x in good_regions[region] if 'C' not in x])
                else:
                    num_mouse = len([x for x in good_regions[region] if 'C' in x])
                session_num = len(phase_dict[phase])
                if types in ['go','unpred_water','c_reward']:
                    average_mat = average_signal_by_trial(region_group_data,region,types,
                                                  phase,length,num_mouse,session_num,
                                                  rm_bad = True,rm_window = [105,130]) 
                else: 
                    average_mat = average_signal_by_trial(region_group_data,region,types,
                                                  phase,length,num_mouse,session_num,
                                                  rm_bad = False) 
                # # normalizer always based on go trial
                # go_mat = average_signal_by_trial(region_data,region,'go_omit',
                #                               'cond_days',length,num_mouse,5,
                #                               ) 
                normalizer_peak = np.mean(np.nanmax(average_mat,axis = 1),axis = 0)
                new_mat = average_mat.copy()
                # for i in range(session_num):
                #     for j in range(num_mouse):
                #     # ax[i].plot(np.nanmean(a[i,:,:],axis =1))
                #         new_mat[i,:,j] = average_mat[i,:,j]/normalizer_peak[j]*np.mean(normalizer_peak)
                average_trace_dict[group][phase][region][types] = new_mat

save_path = 'D:/PhD/Photometry/DATA/photo-pickles/combined'
filename = 'avg_region_based_data_by_group_notyet_filtered'
pickle_dict(average_trace_dict,save_path,filename) 

#%% plot

# condition, region LNacS, session 5, go trial



def plot_single_session_average_trace(data, phase,region,types,session_id,CS,US,group,figsize = (5,4),ylim=[-1,7]):
    aa = np.nanmean(data[phase][region][types][session_id,:,:],axis = 1)
    fig,ax = plt.subplots(figsize = (figsize[0],figsize[1]))
    ax.plot(aa,color = 'k',linewidth = 2)
    plt.xticks(np.arange(0,180,20),np.arange(-2,7,1))
    plt.xlabel('Time from odor onset(s)')
    plt.ylabel('Signals(Zscores)')
    ax.set_ylim([ylim[0],ylim[1]])
    
    # filled area
    stde = np.nanstd(data[phase][region][types][session_id,:,:],axis = 1)/np.sqrt(data[phase][region][types][session_id,:,:].shape[1])
    ax.fill_between(np.arange(0,180,1), aa-stde, aa+stde,alpha = 0.2,color = 'k')
    ymin, ymax = ax.get_ylim()
    if CS:       
        # vertical lines
        ax.fill_between([40,60], [7,7], [-0.5,-0.5],color = 'purple',edgecolor = None, alpha=0.2)
    
    if US:     
        # vertical lines
        ax.fill_between([110,114], [7,7], [-0.5,-0.5], color = 'blue', alpha=0.4,edgecolor = None,)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    savepath = 'D:/PhD/Figures/photometry'
    plt.savefig("{0}/{1}_{2}_{3}_{4}.png".format(savepath,region,phase,types,group), bbox_inches="tight", dpi = 300)
    plt.savefig("{0}/{1}_{2}_{3}_{4}.eps".format(savepath,region,phase,types,group), bbox_inches="tight", dpi = 300)
    plt.savefig("{0}/{1}_{2}_{3}_{4}.pdf".format(savepath,region,phase,types,group), bbox_inches="tight", dpi = 300)
    plt.show()

ymax = 7
group = 'deg_group'
# regions = ['AMOT','PMOT','ALOT','PLOT','MNacS','lNacS']
regions = ['LNacS']
trialtypes_full = ['go','no_go','go_omit','c_omit','c_reward']
for region in regions:
    for types in trialtypes_full:
        try:
            plot_single_session_average_trace(data=average_trace_dict[group], phase = 'cond_days',
                                              region = region,types = types,session_id = 4,
                                              CS = True,US = True,
                                              figsize = (5,4),ylim=[-1,ymax],group = group)
        except:
            pass
        try:
            plot_single_session_average_trace(data=average_trace_dict[group], phase = 'deg_days',
                                              region = region,types = types,session_id = 4,
                                              CS = True,US = True,
                                              figsize = (5,4),ylim=[-1,ymax],group = group)
        except:
            pass
        try:            
            plot_single_session_average_trace(data=average_trace_dict[group], phase = 'rec_days',
                                              region = region,types = types,session_id = 2,
                                              CS = True,US = True,
                                              figsize = (5,4),ylim=[-1,ymax],group = group)
        except:
            pass
        try:            
            plot_single_session_average_trace(data=average_trace_dict[group], phase = 'ext_days',
                                              region = region,types = types,session_id = 2,
                                              CS = True,US = True,
                                              figsize = (5,4),ylim=[-1,ymax],group = group)
        except:
            pass
        try:            
            plot_single_session_average_trace(data=average_trace_dict[group], phase = 'finalrec_days',
                                              region = region,types = types,session_id = 1,
                                              CS = True,US = True,
                                              figsize = (5,4),ylim=[-1,ymax],group = group)
        except:
            pass
        try:            
            plot_single_session_average_trace(data=average_trace_dict[group], phase = 'c_odor_days',
                                              region = region,types = types,session_id = 1,
                                              CS = True,US = True,
                                              figsize = (5,4),ylim=[-1,ymax],group = group)
        except:
            pass



#%%
def plot_multi_session_average_trace(data, phase,region,types,CS=None,US = None,figsize = (4,7),ylim=[-1,7],gap = 7):
     
    fig,ax = plt.subplots(figsize = (figsize[0],figsize[1]))
    i = 0
    sessions = data[phase][region][types].shape[0]
    for session_id in range(sessions):
        aa = np.nanmean(data[phase][region][types][session_id,:,:],axis = 1)
        ax.plot(aa+i,color = cm.cool(session_id/float(sessions)))
        std = np.nanstd(data[phase][region][types][session_id,:,:],axis = 1)
        ax.fill_between(np.arange(0,180,1), aa+i-std, aa+i+std,alpha = 0.2,color = cm.cool(session_id/float(sessions)))
        i += gap
    plt.xticks(np.arange(0,180,20),np.arange(-2,7,1))
    plt.xlabel('Time from odor onset(s)')
    plt.ylabel('Signals(Zscores)')
    # ax.set_ylim([ylim[0],ylim[1]])
    
    # filled area
    
    ymin, ymax = ax.get_ylim()
    if CS:       
        # vertical lines
        ax.vlines(x=40, ymin=ymin, ymax=ymax, colors='tab:orange', ls='--', lw=2)
    if US:     
        # vertical lines
        ax.vlines(x=110, ymin=ymin, ymax=ymax, colors='tab:blue', ls='--', lw=2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([0,5])
    plt.show()
    
plot_multi_session_average_trace(data=average_trace_dict, phase = 'cond_days',
                                  region = 'LNacS',types = 'go',
                                  CS=None,US = None,figsize = (4,6),
                                  ylim=[-1,7],gap =16)

plot_multi_session_average_trace(data=average_trace_dict, phase = 'deg_days',
                                  region = 'LNacS',types = 'go',
                                  CS=None,US = None,figsize = (4,6),
                                  ylim=[-1,7],gap = 12)

plot_multi_session_average_trace(data=average_trace_dict, phase = 'rec_days',
                                  region = 'LNacS',types = 'go',
                                  CS=None,US = None,figsize = (4,4.8),
                                  ylim=[-1,7],gap = 14)

plot_multi_session_average_trace(data=average_trace_dict, phase = 'ext_days',
                                  region = 'LNacS',types = 'go_omit',
                                  CS=None,US = None,figsize = (4,3.6),
                                  ylim=[-1,7],gap = 12)

plot_multi_session_average_trace(data=average_trace_dict, phase = 'finalrec_days',
                                  region = 'LNacS',types = 'go',
                                  CS=None,US = None,figsize = (4,2.4),
                                  ylim=[-1,7],gap = 12)

#%% plot multiple trialtypes

def plot_multi_session_multitype_average_trace(data, phase,region,types,CS=None,US = None,figsize = (4,7),ylim=[-1,7],gap = 7):
    type_color = {'go':'orange','go_omit':'yellow','no_go':'green','background':'grey','UnpredReward':'blue'}
    fig,ax = plt.subplots(figsize = (figsize[0],figsize[1]))
    i = 0
    sessions = data[phase][region][types[0]].shape[0]
    for session_id in range(sessions):
        for ttype in types:
            aa = np.nanmean(data[phase][region][ttype][session_id,:,:],axis = 1)
            ax.plot(aa+i,color = type_color[ttype], label = ttype if session_id == 0 else '')
            # std = np.nanstd(data[phase][region][ttype][session_id,:,:],axis = 1)
            # ax.fill_between(np.arange(0,180,1), aa+i-std, aa+i+std,alpha = 0.2,color = type_color[ttype])
        i += gap
    plt.xticks(np.arange(0,180,20),np.arange(-2,7,1))
    plt.xlabel('Time from odor onset(s)')
    plt.ylabel('Signals(Zscores)')
    # ax.set_ylim([ylim[0],ylim[1]])
    
    # filled area
    
    ymin, ymax = ax.get_ylim()
    if CS:       
        # vertical lines
        ax.vlines(x=40, ymin=ymin, ymax=ymax, colors='tab:orange', ls='--', lw=2)
    if US:     
        # vertical lines
        ax.vlines(x=110, ymin=ymin, ymax=ymax, colors='tab:blue', ls='--', lw=2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([0,5])
    plt.legend(bbox_to_anchor=(1.1, 0.5),frameon = False,loc = 'center left')
    plt.show()


plot_multi_session_multitype_average_trace(data=average_trace_dict, phase = 'cond_days',
                                  region = 'LNacS',types = ['go','go_omit','no_go'],
                                  CS=None,US = None,figsize = (4,6),
                                  ylim=[-1,7],gap = 10)

plot_multi_session_multitype_average_trace(data=average_trace_dict, phase = 'deg_days',
                                  region = 'LNacS',types = ['go','go_omit','no_go','UnpredReward'],
                                  CS=None,US = None,figsize = (4,6),
                                  ylim=[-1,7],gap = 10)

plot_multi_session_multitype_average_trace(data=average_trace_dict, phase = 'rec_days',
                                  region = 'LNacS',types = ['go','go_omit','no_go'],
                                  CS=None,US = None,figsize = (4,4.8),
                                  ylim=[-1,7],gap = 10)

plot_multi_session_multitype_average_trace(data=average_trace_dict, phase = 'ext_days',
                                  region = 'LNacS',types = ['go_omit','no_go'],
                                  CS=None,US = None,figsize = (4,3.6),
                                  ylim=[-1,7],gap = 10)

plot_multi_session_multitype_average_trace(data=average_trace_dict, phase = 'finalrec_days',
                                  region = 'LNacS',types = ['go','go_omit','no_go'],
                                  CS=None,US = None,figsize = (4,2.4),
                                  ylim=[-1,7],gap = 10)



#%% plotting
# go omit
plot_multi_session_average_trace(data=average_trace_dict, phase = 'cond_days',
                                  region = 'LNacS',types = 'go_omit',
                                  CS=None,US = None,figsize = (4,6),
                                  ylim=[-1,7],gap =16)

plot_multi_session_average_trace(data=average_trace_dict, phase = 'deg_days',
                                  region = 'LNacS',types = 'go_omit',
                                  CS=None,US = None,figsize = (4,6),
                                  ylim=[-1,7],gap = 12)

plot_multi_session_average_trace(data=average_trace_dict, phase = 'rec_days',
                                  region = 'LNacS',types = 'go_omit',
                                  CS=None,US = None,figsize = (4,4.8),
                                  ylim=[-1,7],gap = 14)

plot_multi_session_average_trace(data=average_trace_dict, phase = 'ext_days',
                                  region = 'LNacS',types = 'go_omit',
                                  CS=None,US = None,figsize = (4,3.6),
                                  ylim=[-1,7],gap = 12)

plot_multi_session_average_trace(data=average_trace_dict, phase = 'finalrec_days',
                                  region = 'LNacS',types = 'go_omit',
                                  CS=None,US = None,figsize = (4,2.4),
                                  ylim=[-1,7],gap = 12)

#%% plotting
# unpredReward


plot_multi_session_average_trace(data=average_trace_dict, phase = 'deg_days',
                                  region = 'LNacS',types = 'UnpredReward',
                                  CS=None,US = None,figsize = (4,6),
                                  ylim=[-1,7],gap = 12)




plot_single_session_average_trace(data=average_trace_dict, phase = 'deg_days',
                                  region = 'LNacS',types = 'UnpredReward',session_id = 4,
                                  CS = True,US = True,
                                  figsize = (5,4),ylim=[-1,ymax])



#%% notstacked
def plot_multi_session_average_trace_overlay(data, phase,region,types,CS=None,US = None,figsize = (4,7),ylim=[-1,7],gap = 7):
     
    fig,ax = plt.subplots(figsize = (figsize[0],figsize[1]))
    i = 0
    sessions = data[phase][region][types].shape[0]
    for session_id in range(sessions):
        aa = np.nanmean(data[phase][region][types][session_id,:,:],axis = 1)
        ax.plot(aa+i,color = cm.cool(session_id/float(sessions)))
        # std = np.nanstd(data[phase][region][types][session_id,:,:],axis = 1)
        # ax.fill_between(np.arange(0,180,1), aa+i-std, aa+i+std,alpha = 0.2,color = cm.cool(session_id/float(sessions)))
    plt.xticks(np.arange(0,180,20),np.arange(-2,7,1))
    plt.xlabel('Time from odor onset(s)')
    plt.ylabel('Signals(Zscores)')
    # ax.set_ylim([ylim[0],ylim[1]])
    
    # filled area
    
    ymin, ymax = ax.get_ylim()
    if CS:       
        # vertical lines
        ax.vlines(x=40, ymin=ymin, ymax=ymax, colors='tab:orange', ls='--', lw=2)
    if US:     
        # vertical lines
        ax.vlines(x=110, ymin=ymin, ymax=ymax, colors='tab:blue', ls='--', lw=2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([0,5])
    plt.show()
        
plot_multi_session_average_trace_overlay(data=average_trace_dict, phase = 'deg_days',
                                  region = 'MNacS',types = 'UnpredReward',
                                  CS=None,US = None,figsize = (4,6),
                                  ylim=[-1,7],gap = 12)

#%% 12/11/2022 CHECK initial resposne to water
for mouse in range(5):
    if mouse in [3,4]:
        sns.heatmap(region_data['LNacS']['go'][mouse][:40,:,1])
    else:   
        sns.heatmap(region_data['LNacS']['go'][mouse][:40,:,0])
    plt.show()
