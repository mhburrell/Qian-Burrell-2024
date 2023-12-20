# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 15:29:38 2022

@author: qianl

QC: clear
"""
import scipy
import numpy as np
import pandas as pd
import matplotlib as plt
from os.path import dirname, join as pjoin
import scipy.io as sio
import math
from datetime import datetime
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



class Mouse_data:
    def __init__(self,mouse_id, protocol, filedir, group = 'T', implant_side = 'L'):
        """
        Initialize the MouseExperiment class with various attributes.
        
        Parameters:
        - mouse_id: ID of the mouse
        - protocol: Experimental protocol
        - filedir: Directory for storing files
        - group: Experimental group, default is 'T', treatment group
        - implant_side: Side of implant, default is 'L', left side
        """
        # Basic Info
        self.mouse_id = mouse_id
        self.protocol = protocol
        self.filedir = filedir
        self.group = group
        self.implant_side = implant_side

        # File Info
        self.filename = ''
        self.selected_filename = ''
        self.all_days = []

        # Behavioral Data
        self.training_type = []
        self.df_trials = {}
        self.trialtypes = []
        self.df_trials_iscorrect = {}
        self.df_trials_lick = {}
        self.df_eventcode = {}

        # Statistics
        self.p_hit = {}
        self.p_correj = {}
 
        # Licking Data
        self.licking_actionwindow = {}
        self.licking_latency = {}
        self.licking_baselicking = {}
        self.stats = {}

        # Event-related Data
        self.event_data = ''
        
        # Timing Variables
        self.odor_bef = 2.0
        self.odor_on = 1.0
        self.delay = 2.5
        self.rew_after = 8
        self.avg_ITI = 2
        

    def read_filename(self):
        filedir = pjoin(self.filedir, '{}/{}/Session Data/processed/'.format(self.mouse_id,self.protocol))
    
        filename = []
        for dirpath, dirnames, files in os.walk(filedir): # can walk through all levels down
        #     print(f'Found directory: {dirpath}')
            for f_name in files:
                if f_name.endswith('.mat'):
                    filename.append(dirpath+'/'+f_name)
                    print(f_name)
        print('---------------------------------------------')    
        print('The files have been loaded from the following paths')
        
        self.filename = filename
        

        
    def create_dataset(self): #{'date':df of eventcode} format from original CSV
        date_list = []
        df = {}
        df_2 = {}
        task = []
        
        for session, file in enumerate(self.filename):
            
            perdate_df = {}
            date = re.search(r"(\d{8})",file[-50:-1]).group(0) # extract date: must be like format 2020-02-10
            
            date_list.append(date) # create a list of emperiment date
            
            train_type = os.path.split(file)[-1][len(self.mouse_id)+len(self.protocol)+11:-4]
            
            perdate_df['task'] = train_type
            task.append(train_type) ###
            print(file)
                
            
            mat_contents = sio.loadmat(file) 
            perdate_df['mat_contents'] = mat_contents
            perdate_df['num_ROI'] = mat_contents['MSFPAcq']['ROI'][0,0][0,0]
            perdate_df['num_CAM'] = mat_contents['MSFPAcq']['CAM'][0,0][0,0]
            perdate_df['num_EXC'] = mat_contents['MSFPAcq']['EXC'][0,0][0,0]
            perdate_df['num_FrameRate'] = mat_contents['MSFPAcq']['FrameRate'][0,0][0,0]
            trial_num = mat_contents['Events'].shape[1]
            perdate_df['trial_num'] = trial_num
            perdate_df['session'] = session
            # suppose we keep ROI 0-7,10-11, each list means Time, ROI0, ROI1,...etc
            list_ROIs_gcamp = [[] for _ in range(perdate_df['num_ROI'] +1)]
            list_ROIs_isos = [[] for _ in range(perdate_df['num_ROI'] +1)]
            
            #if perdate_df['num_CAM']== 1 and perdate_df['num_EXC'] == 2:
            for trialnum in range(trial_num):
                for field in range(perdate_df['num_ROI']+1):
                    if field == 0 :
                        list_ROIs_isos[field].append(mat_contents['MSFP']['Time'][0][trialnum][0][0][0]-mat_contents['MSFP']['Time'][0][trialnum][0][0][0][0])
                        list_ROIs_gcamp[field].append(mat_contents['MSFP']['Time'][0][trialnum][0][1][0]-mat_contents['MSFP']['Time'][0][trialnum][0][1][0][0])
                    else: 
                        list_ROIs_isos[field].append(mat_contents['MSFP']['Fluo'][0][trialnum][0][0][:,field-1])
                        list_ROIs_gcamp[field].append(mat_contents['MSFP']['Fluo'][0][trialnum][0][1][:,field-1])
            
    
    
            # ROI correspondence table
            if perdate_df['num_ROI'] == 6 and self.implant_side == 'L':
                corr_ROI = {'AMOT':0,'PMOT':4,'ALOT':1,'PLOT':5,'MNacS':2,'LNacS':3}
                perdate_df['corr_ROI'] = corr_ROI
            elif perdate_df['num_ROI'] == 6 and self.implant_side == 'R':
                corr_ROI = {'AMOT':1,'PMOT':0,'ALOT':4,'PLOT':5,'MNacS':2,'LNacS':3}
                perdate_df['corr_ROI'] = corr_ROI
                
            elif perdate_df['num_ROI'] == 12:
                corr_ROI = {'AMOT':9,'PMOT':8,'ALOT':1,'PLOT':4,'MNacS':5,'LNacS':0,'NacC':2, 'AmLOT':3, 'Pir':10, 'VP':11}
                perdate_df['corr_ROI'] = corr_ROI
            
            if self.group == 'C':
                if  perdate_df['num_ROI'] == 6:
                
                    d = {
                         'TrialStart':mat_contents['States']['TrialStart'][0],
                             'Foreperiod':mat_contents['States']['Foreperiod'][0],
                        'go':mat_contents['States']['go'][0],
                        'no_go':mat_contents['States']['no_go'][0],
                            'go_omit':mat_contents['States']['go_omit'][0],
                            'c_odor':mat_contents['States']['c_odor'][0],
                            'c_odor_omit':mat_contents['States']['c_odor_omit'][0],
                        'background':mat_contents['States']['background'][0],
                        'Trace':mat_contents['States']['Trace'][0],
                        'ITI':mat_contents['States']['ITI'][0],
                        'UnpredReward':mat_contents['States']['UnexpectedReward'][0],
                        'water':mat_contents['States']['Reward'][0],
                        'TrialEnd':mat_contents['States']['TrialEnd'][0],
                         'AMOT': list_ROIs_gcamp[corr_ROI['AMOT']+1], 'PMOT': list_ROIs_gcamp[corr_ROI['PMOT']+1], 'ALOT':list_ROIs_gcamp[corr_ROI['ALOT']+1],
                                 'PLOT':list_ROIs_gcamp[corr_ROI['PLOT']+1],'MNacS':list_ROIs_gcamp[corr_ROI['MNacS']+1],'LNacS':list_ROIs_gcamp[corr_ROI['LNacS']+1],
                                 'Doric_Time_EXC1':list_ROIs_gcamp[0],
                                 'AMOT_isos': list_ROIs_isos[corr_ROI['AMOT']+1], 'PMOT_isos': list_ROIs_isos[corr_ROI['PMOT']+1], 
                                 'ALOT_isos':list_ROIs_isos[corr_ROI['ALOT']+1],'PLOT_isos':list_ROIs_isos[corr_ROI['PLOT']+1],
                                 'MNacS_isos':list_ROIs_isos[corr_ROI['MNacS']+1],
                                 'LNacS_isos':list_ROIs_isos[corr_ROI['LNacS']+1],
                                 }
                    
                elif perdate_df['num_ROI'] == 12:
                    d = {
                         'TrialStart':mat_contents['States']['TrialStart'][0],
                             'Foreperiod':mat_contents['States']['Foreperiod'][0],
                        'go':mat_contents['States']['go'][0],
                        'no_go':mat_contents['States']['no_go'][0],
                            'go_omit':mat_contents['States']['go_omit'][0],
                            'c_odor':mat_contents['States']['c_odor'][0],
                            'c_odor_omit':mat_contents['States']['c_odor_omit'][0],
                        'background':mat_contents['States']['background'][0],
                        'Trace':mat_contents['States']['Trace'][0],
                        'ITI':mat_contents['States']['ITI'][0],
                        'UnpredReward':mat_contents['States']['UnexpectedReward'][0],
                        'water':mat_contents['States']['Reward'][0],
                        'TrialEnd':mat_contents['States']['TrialEnd'][0],
                         'AMOT': list_ROIs_gcamp[corr_ROI['AMOT']+1], 'PMOT': list_ROIs_gcamp[corr_ROI['PMOT']+1], 'ALOT':list_ROIs_gcamp[corr_ROI['ALOT']+1],
                                 'PLOT':list_ROIs_gcamp[corr_ROI['PLOT']+1],'MNacS':list_ROIs_gcamp[corr_ROI['MNacS']+1],'LNacS':list_ROIs_gcamp[corr_ROI['LNacS']+1],
                                 'NacC':list_ROIs_gcamp[corr_ROI['NacC']+1],'AmLOT':list_ROIs_gcamp[corr_ROI['AmLOT']+1],
                                 'Pir':list_ROIs_gcamp[corr_ROI['Pir']+1],'VP':list_ROIs_gcamp[corr_ROI['VP']+1],
                                 
                                 'Doric_Time_EXC1':list_ROIs_gcamp[0],
                                 'AMOT_isos': list_ROIs_isos[corr_ROI['AMOT']+1], 'PMOT_isos': list_ROIs_isos[corr_ROI['PMOT']+1], 
                                 'ALOT_isos':list_ROIs_isos[corr_ROI['ALOT']+1],'PLOT_isos':list_ROIs_isos[corr_ROI['PLOT']+1],
                                 'MNacS_isos':list_ROIs_isos[corr_ROI['MNacS']+1],
                                 'LNacS_isos':list_ROIs_isos[corr_ROI['LNacS']+1],
                                 'NacC_isos':list_ROIs_isos[corr_ROI['NacC']+1],'AmLOT_isos':list_ROIs_isos[corr_ROI['AmLOT']+1],
                                 'Pir_isos':list_ROIs_isos[corr_ROI['Pir']+1],'VP_isos':list_ROIs_isos[corr_ROI['VP']+1],
                                 }
                d_2 = {
                     'TrialStart':[x[0] for x in mat_contents['States']['TrialStart'][0]],
                         'Foreperiod':[x[0] for x in mat_contents['States']['Foreperiod'][0]],
                    'go_odor':[x[0] for x in mat_contents['States']['go'][0]],
                    'nogo_odor':[x[0] for x in mat_contents['States']['no_go'][0]],
                        'go_omit':[x[0] for x in mat_contents['States']['go_omit'][0]],
                        'control_odor':[x[0] for x in mat_contents['States']['c_odor'][0]],
                        'control_odor_omit':[x[0] for x in mat_contents['States']['c_odor_omit'][0]],
                    'background':mat_contents['States']['background'][0],
    
                    'UnpredReward':[x[0] for x in mat_contents['States']['UnexpectedReward'][0]],
                    'water_on':[x[0][0] for x in mat_contents['States']['Reward'][0]],
                    'water_off':[x[0][1] for x in mat_contents['States']['Reward'][0]],
                    'trial_end':[x[0][1] for x in mat_contents['States']['TrialEnd'][0]]}
            else: 
                if  perdate_df['num_ROI'] == 6:
                
                    d = {
                         'TrialStart':mat_contents['States']['TrialStart'][0],
                             'Foreperiod':mat_contents['States']['Foreperiod'][0],
                        'go':mat_contents['States']['go'][0],
                        'no_go':mat_contents['States']['no_go'][0],
                            'go_omit':mat_contents['States']['go_omit'][0],
                            
                        'background':mat_contents['States']['background'][0],
                        'Trace':mat_contents['States']['Trace'][0],
                        'ITI':mat_contents['States']['ITI'][0],
                        'UnpredReward':mat_contents['States']['UnexpectedReward'][0],
                        'water':mat_contents['States']['Reward'][0],
                        'TrialEnd':mat_contents['States']['TrialEnd'][0],
                         'AMOT': list_ROIs_gcamp[corr_ROI['AMOT']+1], 'PMOT': list_ROIs_gcamp[corr_ROI['PMOT']+1], 'ALOT':list_ROIs_gcamp[corr_ROI['ALOT']+1],
                                 'PLOT':list_ROIs_gcamp[corr_ROI['PLOT']+1],'MNacS':list_ROIs_gcamp[corr_ROI['MNacS']+1],'LNacS':list_ROIs_gcamp[corr_ROI['LNacS']+1],
                                 'Doric_Time_EXC1':list_ROIs_gcamp[0],
                                 'AMOT_isos': list_ROIs_isos[corr_ROI['AMOT']+1], 'PMOT_isos': list_ROIs_isos[corr_ROI['PMOT']+1], 
                                 'ALOT_isos':list_ROIs_isos[corr_ROI['ALOT']+1],'PLOT_isos':list_ROIs_isos[corr_ROI['PLOT']+1],
                                 'MNacS_isos':list_ROIs_isos[corr_ROI['MNacS']+1],
                                 'LNacS_isos':list_ROIs_isos[corr_ROI['LNacS']+1],
                                 }
                    
                elif perdate_df['num_ROI'] == 12:
                    d = {
                         'TrialStart':mat_contents['States']['TrialStart'][0],
                             'Foreperiod':mat_contents['States']['Foreperiod'][0],
                        'go':mat_contents['States']['go'][0],
                        'no_go':mat_contents['States']['no_go'][0],
                            'go_omit':mat_contents['States']['go_omit'][0],
                            
                        'background':mat_contents['States']['background'][0],
                        'Trace':mat_contents['States']['Trace'][0],
                        'ITI':mat_contents['States']['ITI'][0],
                        'UnpredReward':mat_contents['States']['UnexpectedReward'][0],
                        'water':mat_contents['States']['Reward'][0],
                        'TrialEnd':mat_contents['States']['TrialEnd'][0],
                         'AMOT': list_ROIs_gcamp[corr_ROI['AMOT']+1], 'PMOT': list_ROIs_gcamp[corr_ROI['PMOT']+1], 'ALOT':list_ROIs_gcamp[corr_ROI['ALOT']+1],
                                 'PLOT':list_ROIs_gcamp[corr_ROI['PLOT']+1],'MNacS':list_ROIs_gcamp[corr_ROI['MNacS']+1],'LNacS':list_ROIs_gcamp[corr_ROI['LNacS']+1],
                                 'NacC':list_ROIs_gcamp[corr_ROI['NacC']+1],'AmLOT':list_ROIs_gcamp[corr_ROI['AmLOT']+1],
                                 'Pir':list_ROIs_gcamp[corr_ROI['Pir']+1],'VP':list_ROIs_gcamp[corr_ROI['VP']+1],
                                 
                                 'Doric_Time_EXC1':list_ROIs_gcamp[0],
                                 'AMOT_isos': list_ROIs_isos[corr_ROI['AMOT']+1], 'PMOT_isos': list_ROIs_isos[corr_ROI['PMOT']+1], 
                                 'ALOT_isos':list_ROIs_isos[corr_ROI['ALOT']+1],'PLOT_isos':list_ROIs_isos[corr_ROI['PLOT']+1],
                                 'MNacS_isos':list_ROIs_isos[corr_ROI['MNacS']+1],
                                 'LNacS_isos':list_ROIs_isos[corr_ROI['LNacS']+1],
                                 'NacC_isos':list_ROIs_isos[corr_ROI['NacC']+1],'AmLOT_isos':list_ROIs_isos[corr_ROI['AmLOT']+1],
                                 'Pir_isos':list_ROIs_isos[corr_ROI['Pir']+1],'VP_isos':list_ROIs_isos[corr_ROI['VP']+1],
                                 }
                d_2 = {
                     'TrialStart':[x[0] for x in mat_contents['States']['TrialStart'][0]],
                         'Foreperiod':[x[0] for x in mat_contents['States']['Foreperiod'][0]],
                    'go_odor':[x[0] for x in mat_contents['States']['go'][0]],
                    'nogo_odor':[x[0] for x in mat_contents['States']['no_go'][0]],
                        'go_omit':[x[0] for x in mat_contents['States']['go_omit'][0]],
                        
                    'background':mat_contents['States']['background'][0],
    
                    'UnpredReward':[x[0] for x in mat_contents['States']['UnexpectedReward'][0]],
                    'water_on':[x[0][0] for x in mat_contents['States']['Reward'][0]],
                    'water_off':[x[0][1] for x in mat_contents['States']['Reward'][0]],
                    'trial_end':[x[0][1] for x in mat_contents['States']['TrialEnd'][0]]}
                     
            
            df_trial = pd.DataFrame(data = d)  
            df_trial_2 = pd.DataFrame(data = d_2)  
            
            # add trialtype to the dataframe
            trialtype = []
            for index, row in df_trial_2.iterrows():
                if not math.isnan(row['go_odor'][0]):
                    trialtype.append('go')
                elif not math.isnan(row['nogo_odor'][0]):
                    trialtype.append('no_go')
                elif not math.isnan(row['go_omit'][0]):
                    trialtype.append('go_omit')
                    df_trial_2.at[index, 'go_odor'] =row['go_omit']
                elif not math.isnan(row['background'][0][0]):
                    trialtype.append('background')
                elif not math.isnan(row['UnpredReward'][0]):
                    trialtype.append('unpred_water')
                elif not math.isnan(row['control_odor'][0]):
                    trialtype.append('c_reward')
                elif not math.isnan(row['control_odor_omit'][0]):
                    trialtype.append('c_omit')
                    df_trial_2.at[index, 'control_odor'] =row['control_odor_omit']

            
            df_trial.insert(0,'Trialtype',trialtype)
            df_trial_2.insert(0,'trialtype',trialtype)
            df_trial['lickings'] = [x[0] if len(x) != 0 else x for x in mat_contents['Events']['Port1Out'][0]]
            df_trial_2['licking'] = [x[0] if len(x) != 0 else x for x in mat_contents['Events']['Port1Out'][0]]
            df_trial['id'] = [self.mouse_id]*trial_num
            df_trial['trialnum'] = np.arange(trial_num)
            df_trial['session'] = [session] *trial_num
            df_trial['task'] = [perdate_df['task']] *trial_num
            df_trial['group'] = [self.group] *trial_num
            
            perdate_df['dataframe'] = df_trial

            
            df.update({str(session) +'_'+ date:perdate_df}) # create the dict of key: date and value: data dataframe
            
            df_2.update({str(session) +'_'+ date:df_trial_2})
        self.df_bpod_doric = df #individual mouse event code data
        self.df_trials = df_2
        date_format = '%Y-%m-%d'
        index = np.argsort(date_list)
        
        self.all_days = [str(i) + '_' + date_list[i] for i in index]
        self.training_type = [task[i] for i in index]
        
        print('---------------------------------------------')
        print('{0} has data from these days: {1}'.format(self.mouse_id,list(zip(self.all_days,self.training_type))))


    def create_trial_iscorrect(self): # create dataframe with trial number, correct or rewarded or not only for conditioning period
        for index , date in enumerate(self.all_days):    
            value = self.df_trials[date]
            new_df = self.eval_trials_correct(value)
            new_df.insert(0,'trialtype',value['trialtype'])
            self.df_trials_iscorrect[date] = new_df
            print('create_trial_iscorrect done!')
            
    def eval_trials_correct(self, df):
        
        is_correct = []
        is_rewarded = []
        for index, row in df.iterrows():
            if row['trialtype'] == 'go':
                is_rewarded.append(1)
                if any(x > row['go_odor'][0] and x < row['go_odor'][1]+self.delay for x in row['licking']):
                    is_correct.append(1)
                else:
                    is_correct.append(0)
                
            elif row['trialtype'] == 'no_go':
                is_rewarded.append(0)
                if any(x > row['nogo_odor'][0] and x < row['nogo_odor'][1]+self.delay for x in row['licking']):
                    is_correct.append(0)
                else:
                    is_correct.append(1)
            
            elif row['trialtype'] == 'c_reward':
                is_rewarded.append(1)
                if any(x > row['control_odor'][0] and x < row['water_on'] for x in row['licking']):
                    is_correct.append(1)
                else:
                    is_correct.append(0)
            elif row['trialtype'] == 'c_omit':
                is_rewarded.append(0)
                if any(x > row['control_odor'][0] and x < row['control_odor'][1]+2*self.delay for x in row['licking']):
                    is_correct.append(1)
                else:
                    is_correct.append(0)
            
            elif row['trialtype'] == 'background':
                is_rewarded.append(0)
                if any(x > 0 and x < row['trial_end'] for x in row['licking']):
                    is_correct.append(0)
                else:
                    is_correct.append(1)
                
            elif row['trialtype'] == 'go_omit':
                is_rewarded.append(0)
                if any(x > row['go_odor'][0] and x < row['go_odor'][1]+self.delay for x in row['licking']):
                    is_correct.append(1)
                else:
                    is_correct.append(0)
            elif row['trialtype'] in ['unpred_water','close_unpred_water','far_unpred_water']:
                is_rewarded.append(1)
                is_correct.append(np.nan)
        d = {'is_Correct':is_correct,'is_Rewarded':is_rewarded}
        new_df = pd.DataFrame(d)
        return new_df

    
    def create_trial_lick(self):
        for index , date in enumerate(self.all_days):
            value = self.df_trials[date]            
            new_df = self.lick_stats(value)               
            new_df.insert(0,'trialtype',value['trialtype'])
            self.df_trials_lick[date] = new_df
            print('lick stats done!')        
            
    def lick_stats(self, df):
        lick_num = []
        lick_rate = []
        lick_latent_odor = []
        lick_latent_rew = []
        lick_duration = []
        lick_rate_anti = []
        lick_rate_aftr = []

        for _, row in df.iterrows():
            trialtype = row['trialtype']
            licking = row['licking']

            valid_licking = [x for x in licking if self.is_valid_lick(x, row, trialtype)]
            rate_anti, rate_aftr = self.calculate_rates(licking, row, trialtype)
            num = len(valid_licking)
            latency_odor, latency_rew = self.calculate_latencies(valid_licking, row, trialtype)
            duration = self.calculate_duration(valid_licking, row, trialtype)

            lick_num.append(num)
            lick_rate.append(num / (self.odor_on + self.delay + self.rew_after))
            lick_latent_odor.append(latency_odor)
            lick_latent_rew.append(latency_rew)
            lick_duration.append(duration)
            lick_rate_anti.append(rate_anti)
            lick_rate_aftr.append(rate_aftr)

        d = {
            'lick_num_whole_trial': lick_num,
            'lick_rate_whole_trial': lick_rate,
            'latency_to_odor': lick_latent_odor,
            'latency_to_rew': lick_latent_rew,
            'anti_duration': lick_duration,
            'rate_antici': lick_rate_anti,
            'rate_after': lick_rate_aftr,
        }
        new_df = pd.DataFrame(d)
        return new_df

    def is_valid_lick(self, x, row, trialtype):
        if trialtype == 'background':
            return 0 < x < row['trial_end']
        elif trialtype in ['unpred_water', 'close_unpred_water', 'far_unpred_water']:
            return 0 < x < row['trial_end']
        elif trialtype == 'no_go':
            return row['nogo_odor'][0] < x < row['nogo_odor'][0] + self.odor_on + self.delay
        else:
            return row['go_odor'][0] < x < row['go_odor'][0] + self.odor_on + self.delay + self.rew_after

    def calculate_rates(self, licking, row, trialtype):
        anti_window = self.odor_on + self.delay
        if trialtype == 'background':
            return len(licking) / row['trial_end'], np.nan
        elif trialtype in ['unpred_water', 'close_unpred_water', 'far_unpred_water']:
            return len(licking) / min(self.rew_after, row['trial_end'] - row['water_off']), len(licking) / row['trial_end']
        elif trialtype == 'no_go':
            return len(licking) / anti_window, np.nan
        else:
            anti = [i for i in licking if i > row['go_odor'][0] and i < row['go_odor'][1] + self.delay]
            aftr = [i for i in licking if i > row['water_on'] and i < row['water_off'] + self.rew_after]
            return len(anti) / anti_window, len(aftr) / self.rew_after

    def calculate_latencies(self, valid_licking, row, trialtype):
        if len(valid_licking) != 0:
            if trialtype == 'no_go':
                return min(valid_licking) - row['nogo_odor'][0], np.nan
            elif trialtype in ['unpred_water', 'close_unpred_water', 'far_unpred_water']:
                return np.nan, min(self.rew_after, row['trial_end'] - row['water_off'])
            else:
                return min(valid_licking) - row['go_odor'][0], min(valid_licking) - row['water_on']
        else:
            if trialtype == 'no_go':
                return self.odor_on + self.delay, np.nan
            elif trialtype in ['unpred_water', 'close_unpred_water', 'far_unpred_water']:
                return np.nan, self.rew_after
            else:
                return self.odor_on + self.delay, self.rew_after

    def calculate_duration(self, valid_licking, row, trialtype):
        if trialtype == 'background':
            return np.nan
        elif len(valid_licking) != 0:
            return max(valid_licking) - min(valid_licking)
        else:
            return np.nan

def pickle_dict(df, path, filename):
    try:
        os.makedirs(path) # create the path first
    except FileExistsError:
        print('the path exist.')
        
    filepath = os.path.join(path, f'{filename}.pickle')
    with open(filename, 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved data to {filepath}')


def load_pickleddata(filename):
    
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data

#%% main code
if __name__ == '__main__':
    
    is_save = True
    load_path = 'D:/PhD/Photometry/DATA/C-odor'
    implant_dict = {'FgDA_01':'L','FgDA_02':'L','FgDA_03':'L',
                    'FgDA_04':'L','FgDA_05':'L','FgDA_06':'R',
                    'FgDA_07':'R','FgDA_08':'R','FgDA_09':'L',
                    'FgDA_C1':'L','FgDA_C2':'L','FgDA_C3':'L','FgDA_C4':'L',
                    'FgDA_C5':'R','FgDA_C6':'L','FgDA_C7':'L',
                    'FgDA_W1':'L','FgDA_W2':'L','FgDA_S3':'L','FgDA_S4':'L',}
    group_dict = {'FgDA_01':'D','FgDA_02':'D','FgDA_03':'D',
                    'FgDA_04':'D','FgDA_05':'D','FgDA_06':'D',
                    'FgDA_07':'D','FgDA_08':'D','FgDA_09':'D',
                    'FgDA_C1':'C','FgDA_C2':'C','FgDA_C3':'C','FgDA_C4':'C',
                    'FgDA_C5':'C','FgDA_C6':'C','FgDA_C7':'C',
                    'FgDA_W1':'C','FgDA_W2':'C','FgDA_S3':'C','FgDA_S4':'C',}
    
    # load file  
    mouse_names = ['FgDA_C1','FgDA_C2','FgDA_C3','FgDA_C4','FgDA_C5','FgDA_C6','FgDA_C7'] # C-odor 
    for mouse_name in mouse_names:
        mouse = Mouse_data(mouse_name, protocol = 'Selina_C5D5R3E5R3',filedir = load_path,group = group_dict[mouse_name],implant_side = implant_dict[mouse_name]) #### group 
        mouse.read_filename()
        #parse data
        mouse.create_dataset()
        mouse.create_trial_iscorrect()    
        mouse.create_trial_lick()   
        if is_save:
            #save data by pickle
            save_path = os.path.join(load_path, 'processed')            
            filename = f'{mouse_name}_stats.pkl'
            pickle_dict(mouse,save_path,filename)
            
   