import numpy as np
import pandas as pd
from td_sim.import_simulated import import_rep_group, shape_data_csc, shape_data_microstimuli
from td_sim.td import temporal_difference_learning
import itertools
import warnings
import argparse
from multiprocessing import Pool
import os

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
    - gamma: float, discount factor (default 0.95)
    """
    for testgroup, model_switch in itertools.product(range(1, 4), range(1, 2)):
        out_filename = (
            f"td_sim_rep_{rep}_testgroup_{testgroup}_gamma_{gamma}"
            f"_model_switch_{model_switch}_sigma_{sigma}_tau_{tau}_n_stimuli_{n_stimuli}.parquet"
        )
        
        # Check if file already exists
        if os.path.exists(out_filename):
            print(f"File {out_filename} already exists. Skipping.")
            continue
            
        
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
        result_df.to_parquet(out_filename)

def worker(params):
    """
    Unpack the tuple of parameters and call `run_td_learning`.
    """
    rep, sigma, tau, n_stimuli = params
    run_td_learning(rep, sigma, tau, n_stimuli, 
                    data_file='simulated_trials.parquet',
                    alpha=0.1,
                    gamma=0.95)

def main():
    reps = range(1, 26)  # 1 through 25
    sigmas = np.linspace(0.02, 0.2, 5)   # e.g., 5 points between 0.02 and 0.2
    taus = np.linspace(0.8, 0.99, 5)    # e.g., 5 points between 0.8 and 0.99
    n_stimuli_list = [100] #[5, 10, 20, 50]

    # Generate all combinations using itertools.product
    # Each element of param_list is a tuple (rep, sigma, tau, n_stimuli)
    param_list = list(itertools.product(reps, sigmas, taus, n_stimuli_list))

    # Use a Pool to parallelize
    # By default Pool() uses as many processes as you have CPU cores
    with Pool() as pool:
        pool.map(worker, param_list)

if __name__ == "__main__":
    main()