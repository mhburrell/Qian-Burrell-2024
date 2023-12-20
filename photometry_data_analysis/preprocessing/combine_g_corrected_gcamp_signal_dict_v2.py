# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 18:06:37 2022

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

#%% load existing data
data1 = load_pickleddata('D:/PhD/Data_Code_Contingency_Uchida/Photometry/pickles/corrected_gcamp_data_C_zscore_by_ITI.pickle') #DA_02,03,04,05
data2 = load_pickleddata('D:/PhD/Data_Code_Contingency_Uchida/Photometry/pickles/corrected_gcamp_data_deg_zscore_by_ITI.pickle') #DA_01

#%% operations
# data1 is always the larger one
data1.update(data2)



#%% save pickle

save_path = 'D:/PhD/Data_Code_Contingency_Uchida/Photometry/pickles/'
filename = 'corrected_DA_gcamp_data_combined_zscore_by_ITI'
pickle_dict(data1,save_path,filename)   