#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 12:12:39 2022

@author: mobeets
"""
import numpy as np
import scipy.stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#%%

class Condition:
    def __init__(self, name, values, color, linestyle='-'):
        self.name = name
        self.values = values
        self.color = color
        self.linestyle = linestyle
        
    def __str__(self):
        return f'{self.name}'
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.__str__()})'

    @staticmethod
    def get_values(x):
        raise NotImplementedError("Condition must define get_values")
    
    def matches(self, x):
        values = self.__class__.get_values(x)
        return all([(x == y) if ~np.isnan(y) else np.isnan(x) for x,y in zip(values, self.values)])

def condition_matcher(trial, conditions):
    try:
        return next(c for c in conditions if c.matches(trial))
    except:
        return conditions[0]

def get_exemplars(data, conditions, index_in_episode=0, ignore_trials_after_omissions=True):
    exemplars = []
    for y, condition in enumerate(conditions):
        cdatas = [d for t,d in enumerate(data) if condition.matches(d) and
                  (d.index_in_episode == index_in_episode if index_in_episode is not None else True)
                  and (t == 0 or (data[t-1].y.max() > 0 or not ignore_trials_after_omissions))]
        exemplar = sorted(cdatas, key=lambda d: d.iti)
        if len(exemplar) > 0:
            exemplar = exemplar[0] # find minimum ITI
        else:
            print("Could not find any trials matching condition: {}".format(condition))
            continue
        exemplar.condition = condition
        exemplars.append(exemplar)
    return exemplars

class TraceCondition(Condition):
    def get_values(x):
        return (x.cue, x.isi if x.y.max() > 0 else np.nan)

odorA_clr = np.array([229,106,179])/255 # odor a
odorB_clr = np.array([113, 142, 255])/255 #odor B
odorC_clr = np.array([254, 206, 95])/255 # odor C?
odorD_clr = np.array([169, 169, 169])/255 # omission

ODOR_COLORS = [odorA_clr, odorB_clr, odorC_clr, odorD_clr]

def get_conditions(data):
    conditions = []
    
    cues = np.unique([d.cue for d in data])
    for cue in cues:
        isis = np.unique([d.isi for d in data if d.cue == cue and d.y.max() > 0])
        if any([d.y.max() == 0 for d in data if d.cue == cue]): # omission trials exist
            isis = np.hstack([isis[~np.isnan(isis)], [np.nan]])
        
        for i, isi in enumerate(isis):
            is_omission = np.isnan(isi)
            if cue == 0:
                if is_omission:
                    clr = np.array([169, 169, 169])/255 #omission
                else:
                    # color is linear interpolation between three anchor colors
                    cisis = isis[~np.isnan(isis)]
                    p = np.argmax(isi == cisis)/len(cisis)
                    clr_l = np.array([229,106,179])/255 #odor A
                    clr_m = np.array([169, 169, 169])/255
                    clr_u = np.array([28, 227, 255])/255
                    if p < 0.5:
                        q = p/0.5
                        clr = q*clr_m + (1-q)*clr_l
                    else:
                        q = (p-0.5)/0.5
                        clr = q*clr_u + (1-q)*clr_m
            else:
                if ~np.isnan(cue):
                    clr = ODOR_COLORS[cue]
                else:
                    clr = ODOR_COLORS[-1]
            if ~np.isnan(cue):
                cue_name = ['A', 'B', 'C', 'D'][cue]
            else: # invisible cue
                cue_name = 'C'
            if is_omission:
                name = f'odor {cue_name} (omission)'
                lnstl = '--'
            else:
                name = f'odor {cue_name}, isi {isi}'
                lnstl = '-'
            cond = TraceCondition(name, [cue, isi], clr, lnstl)
            conditions.append(cond)
    return conditions

def get_conditions_contingency(data, mode):
    conditions = get_conditions(data)[::-1]
    for condition in conditions:
        if 'odor A' in condition.name:
            condition.name = condition.name.split(', isi')[0]
        elif 'odor B' in condition.name:
            condition.name = 'nogo'
        elif 'odor C' in condition.name:
            if mode.lower() == 'conditioning':
                condition.name = 'background'
                condition.color = np.array([130, 130, 130])/255  # background color
            elif mode.lower() == 'degradation':
                if 'isi' in condition.name:
                    condition.name = 'unexp. reward'
                    condition.color = np.array([181, 45, 234])/255
                else:
                    condition.name = 'background'
                    condition.color = np.array([130, 130, 130])/255
            elif mode.lower() == 'cue-c':
                if 'isi' in condition.name:
                    condition.name = condition.name.split(', isi')[0]
                else:
                    condition.name = 'background'
                    condition.color = np.array([130, 130, 130])/255
    return conditions

#%% PCA

def fit_pca(data, key='Z', handleNan=False):
    Z = np.vstack([x.__dict__[key] for x in data])
    if handleNan: # replace any nans with column mean
        ixNan = np.any(np.isnan(Z), axis=1)
        Z[ixNan,:] = np.nanmean(Z, axis=0)
    pca = PCA(n_components=Z.shape[-1])
    pca.fit(Z)
    return (pca, ixNan) if handleNan else pca

def apply_pca(trials, pca, key='Z'):
    for trial in trials:
        trial.Z_pc = pca.transform(trial.__dict__[key])
    return trials
