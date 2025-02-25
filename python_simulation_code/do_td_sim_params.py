import numpy as np
import pandas as pd
import os
from td_sim.import_simulated import import_rep_group, shape_data_csc
from td_sim.td import temporal_difference_learning
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse

def run_td_learning(rep, model_switch, testgroup, data_file='simulated_trials.parquet', alpha=0.1, gamma=0.925):
    output_file = 'td_sim_rep_{}_testgroup_{}_gamma_{}_model_switch_{}.parquet'.format(rep,
                                                                                       testgroup,
                                                                                       gamma,
                                                                                       model_switch)
    if os.path.exists(output_file):
        print(f"File '{output_file}' already exists. Skipping computation.")
        return 

    print("rep: {}, testgroup: {}, model_switch: {}, gamma: {}".format(rep, testgroup, model_switch, gamma))
    df = import_rep_group(data_file, rep, testgroup)

    if model_switch == 1:
        cue_matrix, context_matrix, rewards = shape_data_csc(df)
    elif model_switch == 2:
        cue_matrix, context_matrix, rewards = shape_data_csc(df)
        # Add context_matrix to cue_matrix
        cue_matrix = np.concatenate((cue_matrix, context_matrix), axis=1)
    elif model_switch == 3:
        cue_matrix, context_matrix, rewards = shape_data_csc(df, threshold=200)

    initial_weights = np.zeros((cue_matrix.shape[1], 1))
    w, rpe, value = temporal_difference_learning(cue_matrix, rewards, alpha, gamma, initial_weights)

    # Write rpe, value to a dataframe, then save to parquet
    df_out = pd.DataFrame({'rpe': rpe[:,0], 'value': value[:,0]})
    # Add a column t, which is the epoch number, starting at 1
    df_out['t'] = df_out.index + 1
    df_out['rep'] = rep
    df_out['testgroup'] = testgroup
    df_out['model_switch'] = model_switch
    df_out['alpha'] = alpha
    df_out['gamma'] = gamma
    df_out.to_parquet(output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run temporal difference learning simulation.')
    parser.add_argument('rep', type=int, help='Repetition number')
    parser.add_argument('model_switch', type=int, help='model switch')
    parser.add_argument('testgroup', type=int, help='Test group number)')
    parser.add_argument('gamma', type=float, help='discount factor')
    parser.add_argument('alpha', type=float, help='learning rate')

    args = parser.parse_args()

    run_td_learning(args.rep, args.model_switch, args.testgroup, gamma=args.gamma, alpha=args.alpha)
