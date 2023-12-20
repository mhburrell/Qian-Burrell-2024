# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:01:38 2022

@author: qianl

QC: clear
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


#%%
def event_plot(df,save_dir,mouse_id = '', exp_date = '', filename = 'test',save = False, width = 3.5, figuresize = [12,20],gocolor = '#EBA0B1',nogocolor = '#F9CD69', rewardcolor = '#3083D1',lickcolor = 'grey'):
    lineoffsets2 = 1
    linelengths2 = 1
      
    # create a horizontal plot
    figure = plt.figure()
    df_copy = df.reset_index(inplace=False)

    for i, row in df_copy.iterrows():
        plt.hlines(i, row['go'][0][0], row['go'][0][1],color = gocolor,alpha = 1,
                   linewidth = width,label = 'go odor' if i ==0 else '')
        plt.hlines(i, row['water'][0][0], row['water'][0][1],color = rewardcolor,
                   linewidth = width,label = 'water' if i ==0 else '')
        plt.hlines(i, row['no_go'][0][0], row['no_go'][0][1],color = nogocolor,alpha = 1,
                   linewidth = width,label = 'no go odor' if i ==0 else '')
        plt.hlines(i, row['go_omit'][0][0], row['go_omit'][0][1],color = gocolor,alpha = 1,
                   linewidth = width)
        # plt.hlines(i, row['background'][0][0], row['background'][0][1],color = 'grey',alpha = 1,
        #            linewidth = width)
        plt.hlines(i, 0, row['TrialEnd'][0][1],color = 'grey',alpha = 0.2,
                   linewidth = width)
        try:
            plt.hlines(i, row['c_odor'][0][0], row['c_odor'][0][1],alpha = 1,
                        linewidth = width,label = 'control odor' if i ==0 else '')
        except:
            pass
        
    lickings = []
    for token in df_copy.lickings.values:
        if len(token) == 0:
            lickings.append([])
        else:
            lickings.append(token)
    
    plt.eventplot(lickings, colors=lickcolor, lineoffsets=lineoffsets2,linelengths=linelengths2,alpha = 0.5, label = 'licking')
    
    ax = plt.subplot(111)    
    ax.spines["top"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    draw_loc = ax.get_xlim()[1]
    draw_loc2 = ax.get_ylim()[1]
     
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()
    plt.tick_params(axis="both", which="both", bottom=False, left = False, top = False, right = False,
                    labelbottom="on", labelleft="on",labelsize = 14)   
    plt.ylabel('Trials',fontsize = 18)
    plt.xlabel('Time(s)',fontsize = 18)
    ax.set_ylim(bottom=-1,ymax = len(df_copy.index))
    ax.set_xlim(left=-0.2)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),prop={'size': 14},loc='upper center', bbox_to_anchor=(0.5, 1.07),
          frameon=False,fancybox=False, shadow=False, ncol=3)

    plt.suptitle('{} eventplot on {}'.format(mouse_id, exp_date),fontsize = 20, y = 0.96)
    
    
    if save:
        try:
            savepath = "{1}/{0}/{2}".format(mouse_id,save_dir,date.today())
            os.makedirs(savepath)
        except:
            pass
        figure.set_size_inches(figuresize[0], figuresize[1])
    
        plt.savefig("{0}/{1}_{2}.png".format(savepath,exp_date,filename), bbox_inches="tight", dpi = 100)

        
        #    print('error while saving')
    
    plt.show()
    return figure



#-------------------------
#%% import data

mouse_names = ['FgDA_01','FgDA_02','FgDA_03','FgDA_04','FgDA_05','FgDA_06','FgDA_07','FgDA_08']

path = 'D:/PhD/Photometry/DATA/degradation/5_conditions'
for mouse_id in mouse_names:    
    
    load_path = os.path.join(path,'processed/{0}_stats.pickle'.format(mouse_id))
    mouse = load_pickleddata(load_path)
    
    #event plot with trials and iscorerct data
    
    # assign two df 
    mouse_trials = mouse.df_bpod_doric
    
    # choose a date
    all_days = mouse.all_days
    print('-----------------------------------------------------------')
    print('there are ', len(all_days),'days, condition days is ', len([x for x in mouse.training_type if x == 'cond']))
    for index in range(len(all_days)):
        print('you are looking at day ', all_days[index],'training type is ',mouse.training_type[index] )
        day = all_days[index] 
        
        # get dataframe
        # dataframe = mouse_trials[str(index)+'_'+day]['dataframe'].copy() # need to be hard copy
        dataframe = mouse_trials[day]['dataframe'].copy()

        # plot
        save_dir = os.path.join(path,'figures')
        figure = event_plot(dataframe,mouse_id = mouse.mouse_id,exp_date = day,filename = 'all_trials',save = True,save_dir = save_dir)
        
        # only choose go trials for above day
        is_go_trials = dataframe['Trialtype'] == 'go' #or 'go_omit' # index of go trials
        merged_go_trials = dataframe[is_go_trials] # select out the go trials from the merged dataset
        merged_go_trials.index = range(len(merged_go_trials)) # reindex the index; otherwise the index will be original index like 1,4,14,23,etc
        
        # plot
        figure = event_plot(merged_go_trials,mouse_id = mouse.mouse_id,exp_date = day,filename = 'go_trials',save = True, save_dir = save_dir,width = 3,figuresize = [12,15])
        
        # only choose no_go trials for above day
        is_nogo_trials = dataframe['Trialtype'] == 'no_go' # index of no_go trials
        merged_nogo_trials = dataframe[is_nogo_trials] # select out the no_go trials from the merged dataset
        merged_nogo_trials.index = range(len(merged_nogo_trials)) # reindex the index; otherwise the index will be original index like 1,4,14,23,etc
        # plot
        figure = event_plot(merged_nogo_trials,mouse_id = mouse.mouse_id,exp_date = day,filename = 'no_go_trials',save = True, save_dir = save_dir,width = 3,figuresize = [12,15])
        
        # only choose empty trials for above day
        is_background_trials = dataframe['Trialtype'] == 'background' # index of background trials
        merged_background_trials = dataframe[is_background_trials] # select out the background trials from the merged dataset
        merged_background_trials.index = range(len(merged_background_trials)) # reindex the index; otherwise the index will be original index like 1,4,14,23,etc
        # plot
        figure = event_plot(merged_background_trials,mouse_id = mouse.mouse_id,exp_date = day,filename = 'background_trials',save = True, save_dir = save_dir,width = 3,figuresize = [12,15])
    
    
    
 

                       