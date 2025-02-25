#This code:
# 1. imports simulated trial data
# 2. converts the data to a cue matrix, context matrix and reward vector
# 3. performs temporal difference learning
# 4. saves the result

import numpy as np
import pandas as pd
from td_sim.import_simulated import import_rep_group, shape_data_csc
from td_sim.td import temporal_difference_learning
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data_file = "simulated_trials.parquet"


# 1: CSC, no ITI representation
# 2: CSC, context representation during ITI
# 3: CSC, during ITI and ISI

#there are 25 reps, and three test groups in each rep

#run TD learning for each rep and test group

alpha = 0.1
#try different values of gamma from 0.5 to 0.975 in steps of 0.025
discount_factor = np.arange(0.5,1.0,0.025)

for rep, testgroup, model_switch,gamma in itertools.product(range(1,26),range(1,4),range(1,4),discount_factor):
    print("rep: {}, testgroup: {}, model_switch: {}, gamma: {}".format(rep,testgroup,model_switch,gamma))
    df=import_rep_group(data_file, rep,testgroup)
    if model_switch == 1:
        cue_matrix, context_matrix, rewards = shape_data_csc(df)
    elif model_switch == 2:
        cue_matrix, context_matrix, rewards = shape_data_csc(df)
        #add context_matrix to cue_matrix
        cue_matrix = np.concatenate((cue_matrix,context_matrix),axis=1)
    elif model_switch == 3:
        cue_matrix, context_matrix, rewards = shape_data_csc(df,threshold=200)

    initial_weights = np.zeros((cue_matrix.shape[1],1))
    w,rpe,value = temporal_difference_learning(cue_matrix,rewards,alpha,gamma,initial_weights)

    #write rpe, value to a parquet file
    df = pd.DataFrame({'rpe':rpe[:,0],'value':value[:,0]})
    #add a column t, which is the epoch number, starting at 1
    df['t'] = df.index + 1
    df['rep'] = rep
    df['testgroup'] = testgroup
    df['model_switch'] = model_switch
    df['alpha'] = alpha
    df['gamma'] = gamma
    df.to_parquet('td_sim_rep_{}_testgroup_{}_gamma_{}_model_switch_{}.parquet'.format(rep,testgroup,gamma,model_switch))