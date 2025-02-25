import numpy as np
import pandas as pd
from td_sim.import_simulated import import_rep_group, shape_data_csc, shape_data_microstimuli
from td_sim.td import temporal_difference_learning
import itertools
import warnings
import argparse

warnings.simplefilter(action='ignore', category=FutureWarning)

def run_td_learning(rep, sigma, tau, n_stimuli, data_file='simulated_trials.parquet', alpha=0.1, gamma=0.95):
    """
    Run temporal difference learning for a specific rep, sigma, tau, and n_stimuli.
    
    Parameters:
    - rep: int, repetition number
    - sigma: float, sigma value
    - tau: float, tau value
    - n_stimuli: int, number of stimuli
    - data_file: str, path to the data file (default 'simulated_trials.parquet')
    - alpha: float, learning rate (default 0.1)
    - gamma: float, discount factor (default 0.975)
    """
    for testgroup, model_switch in itertools.product(range(1, 4), range(1, 2)):
        print("rep: {}, testgroup: {}, model_switch: {}, gamma: {}, sigma: {}, tau: {}, n_stimuli: {}".format(rep, testgroup, model_switch, gamma, sigma, tau, n_stimuli))
        df = import_rep_group(data_file, rep, testgroup)
        # Get events, phase, testgroup as numpy arrays int32
        if model_switch == 1:
            cue_matrix, context_matrix, rewards = shape_data_microstimuli(df, tau, n_stimuli, sigma)
        elif model_switch == 2:
            cue_matrix, context_matrix, rewards = shape_data_microstimuli(df, tau, n_stimuli, sigma)
            # Add context_matrix to cue_matrix
            cue_matrix = np.concatenate((cue_matrix, context_matrix), axis=1)

        initial_weights = np.zeros((cue_matrix.shape[1], 1))
        w, rpe, value = temporal_difference_learning(cue_matrix, rewards, alpha, gamma, initial_weights)

        # Write rpe, value to a parquet file
        result_df = pd.DataFrame({'rpe': rpe[:, 0], 'value': value[:, 0]})
        # Add a column t, which is the epoch number, starting at 1
        result_df['t'] = result_df.index + 1
        result_df['rep'] = rep
        result_df['testgroup'] = testgroup
        result_df['model_switch'] = model_switch
        result_df['alpha'] = alpha
        result_df['gamma'] = gamma
        result_df['sigma'] = sigma
        result_df['tau'] = tau
        result_df['n_stimuli'] = n_stimuli
        result_df.to_parquet('td_sim_rep_{}_testgroup_{}_gamma_{}_model_switch_{}_sigma_{}_tau_{}_n_stimuli_{}.parquet'.format(rep, testgroup, gamma, model_switch, sigma, tau, n_stimuli))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run temporal difference learning simulation.')
    parser.add_argument('rep', type=int, help='Repetition number')
    parser.add_argument('sigma', type=float, help='Sigma value')
    parser.add_argument('tau', type=float, help='Tau value')
    parser.add_argument('n_stimuli', type=int, help='Number of stimuli')

    args = parser.parse_args()

    run_td_learning(args.rep, args.sigma, args.tau, args.n_stimuli)
