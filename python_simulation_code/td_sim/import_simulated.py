#code for importing and formating simulated experimental data

import numpy as np
import pandas as pd

def import_rep_group(file_name,rep,group):
    #data is in a parquet file
    #import the data which matchs rep = rep and testgroup = group
    #returns a dataframe
    df = pd.read_parquet(file_name)
    df = df[df['rep']==rep]
    df = df[df['testgroup']==group]
    return df

def shape_data_csc(df,threshold=20):
    #convert events to a (t x n_cues) matrix and 1 x n_cues reward vector
    # df.events contains a list of events
    # 1: nothing, 2:reward, 3:cue 1, 4:cue 2, 5:cue 3
    t = df.shape[0]
    events = df['events'].values
    #convert to int
    events = events.astype(int)
    X = np.eye(5)[events-1]

    #reward is the second column of X
    rewards = X[:,1]
    #cues are the third, fourth and fifth columns of X
    cues = X[:,2:5]

    # Initialize a matrix with the same shape as one_hot_encoded filled with large values
    time_since_last_cue = np.full(cues.shape, np.inf)

    # Iterate over each column (event) in the matrix
    for col in range(cues.shape[1]):
        last_one_idx = -np.inf  # Initialize with negative infinity to handle the first occurrence
        for row in range(cues.shape[0]):
            if cues[row, col] == 1:
                last_one_idx = row
            time_since_last_cue[row, col] = row - last_one_idx if last_one_idx != -np.inf else np.inf

    # Convert the infinite values to zero
    time_since_last_cue[time_since_last_cue == np.inf] = 0 

    #truncate to threshold
    time_since_last_cue[time_since_last_cue>threshold] = 0

    # Initialize the vectors with zeros
    column_of_smallest_non_zero = np.zeros(time_since_last_cue.shape[0], dtype=int)
    smallest_non_zero_value = np.zeros(time_since_last_cue.shape[0])

    # Iterate over each row to find the column with the smallest non-zero value and the value itself
    for i, row in enumerate(time_since_last_cue):
        # Filter out zeros and find the minimum value
        non_zeros = row[row > 0]
        if non_zeros.size > 0:
            min_val = non_zeros.min()
            min_col = np.where(row == min_val)[0][0]  # Get the column index of the minimum value
            column_of_smallest_non_zero[i] = min_col + 1  # +1 to make it 1-based instead of 0-based
            smallest_non_zero_value[i] = min_val

        # Number of rows (t) and number of columns (number of events x threshold)
    n_rows = time_since_last_cue.shape[0]
    n_columns = 3 * threshold

    # Initialize the one-hot encoded matrix with zeros
    csc_matrix = np.zeros((n_rows, n_columns))

    # Populate the matrix based on the calculated position
    for i in range(n_rows):
        if column_of_smallest_non_zero[i] != 0 and smallest_non_zero_value[i] != 0:  # Check for non-zero values
            position = (column_of_smallest_non_zero[i] - 1) * threshold + int(smallest_non_zero_value[i]) - 1
            csc_matrix[i, position] = 1

    #generate context_vector
    #if phase = 1 then context = 1
    #otherwise context = testgroup
    context_vector = np.ones(t)
    context_vector[df['phase'].values==2] = df['testgroup'].values[df['phase'].values==2]

    #convert to int
    context_vector = context_vector.astype(int)

    #convert to one hot encoding
    context_matrix = np.eye(3)[context_vector-1]

    return csc_matrix, context_matrix, rewards

