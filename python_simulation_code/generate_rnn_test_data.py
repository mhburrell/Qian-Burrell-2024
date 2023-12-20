#%%
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
from copy import deepcopy
from valuernn.tasks.contingency import Contingency
from valuernn import model as vmodel
from valuernn import train as vtrain
from rnn_analysis.analysis import get_conditions_contingency, get_exemplars
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%%
def make_exp(seed,is_test=False):
    Exps = {}
    modes = ['conditioning', 'degradation', 'cue-c']

    np.random.seed(seed)
    for mode in modes:
        if is_test:
            E = Contingency(mode=mode,ntrials=1000,ntrials_per_episode=20,jitter=1,t_padding=250,rew_times=[8,8,8])
        else:
            E = Contingency(mode=mode,ntrials=1000,ntrials_per_episode=20,jitter=1,rew_times=[8,8,8])
        Exps[mode] = E

    return Exps    

def make_rnn(hidden_size,seed):
    np.random.seed(seed)
    gamma = 0.83
    model = vmodel.ValueRNN(input_size=4, hidden_size=hidden_size, gamma=gamma)
    model.to(device)
    return(model)

def make_dataloader(Exps,mode):
    batch_size = 1000
    E = Exps[mode]
    dataloader = vtrain.make_dataloader(E,batch_size=batch_size)
    return(dataloader)

def probe_model(model,Exps,mode):
    dataloader = make_dataloader(Exps,mode)
    responses = vtrain.probe_model(model,dataloader)
    return(responses)

def load_weights(model,weights):
    model.load_state_dict(weights)
    return(model)

def extract_response(response):
    # Directly extract attributes
    X = response.X
    y = response.y
    Z = response.Z
    rpe = np.append(response.rpe, 0)
    value = response.value

    # Prepare data for DataFrame
    data = {
        **{f'X{i+1}': X[:, i] for i in range(X.shape[1])},
        'y': y.flatten(),
        **{f'Z{i+1}': Z[:, i] for i in range(Z.shape[1])},
        'rpe': rpe.flatten(),
        'value': value.flatten(),
    }

    df = pd.DataFrame(data)
    
    return df

def get_responses(response_list):
    # Using list comprehension to create list of dataframes
    dfs = [extract_response(response).assign(response_index=i) for i, response in enumerate(response_list)]
    
    # Concatenate all dataframes at once
    responses = pd.concat(dfs, ignore_index=True)
    responses['t'] = responses.index+1
    
    return responses

#%%
if __name__ == '__main__':
   #read in weight files from current folder
   #files are named as follows:
   #(hidden_size)_(seed)_(mode)_weights.pt
    #example: 100_1_conditioning_weights.pt

    #find all .pt files in current folder
    import glob
    files = glob.glob('*.pt')
    #create a table of all the files
    table = pd.DataFrame(files,columns=['file'])
    #extract the hidden size, seed, and mode from the file name
    table['hidden_size'] = table.file.apply(lambda x: x.split('_')[0])
    table['seed'] = table.file.apply(lambda x: x.split('_')[1])
    table['mode'] = table.file.apply(lambda x: x.split('_')[-2])
    table['naive'] = table.file.apply(lambda x: 'naive' in x)

    #iterate over rows of the table
    for i, row in tqdm(table.iterrows()):
        #print file name
        print(row.file)
        #extract the hidden size, seed, and mode
        hidden_size = int(row.hidden_size)
        seed = int(row.seed)
        mode = row['mode']
        naive = row.naive
        #create the experiment
        exp_test = make_exp(seed,is_test=True)
        exp_real = make_exp(seed,is_test=False)
        #create the model
        model = make_rnn(hidden_size,seed)
        #load the weights
        model = load_weights(model,torch.load(row.file,map_location=torch.device('cpu')))
        #probe the model
        #if mode is initial, then test on conditioning and degradation and cue-c
        #otherwise test on the mode
        #e.g. if mode is conditioning, then test on conditioning
        if mode == 'initial':
            responses = probe_model(model,exp_test,'conditioning')
            responses = get_responses(responses)
            responses['seed'] = seed
            responses['hidden_size'] = hidden_size
            responses['condition'] = 'conditioning'
            responses['test'] = True
            responses['trained'] = False
            responses['naive'] = naive
            responses.to_parquet(f'{hidden_size}_{seed}_conditioning_naive{naive}_initial.parquet')

            responses = probe_model(model,exp_test,'degradation')
            responses = get_responses(responses)
            responses['seed'] = seed
            responses['hidden_size'] = hidden_size
            responses['condition'] = 'degradation'
            responses['test'] = True
            responses['trained'] = False
            responses['naive'] = naive
            responses.to_parquet(f'{hidden_size}_{seed}_degradation_naive{naive}_initial.parquet')

            responses = probe_model(model,exp_test,'cue-c')
            responses = get_responses(responses)
            responses['seed'] = seed
            responses['hidden_size'] = hidden_size
            responses['condition'] = 'cue-c'
            responses['test'] = True
            responses['trained'] = False
            responses['naive'] = naive
            responses.to_parquet(f'{hidden_size}_{seed}_cue-c_naive{naive}_initial.parquet')

            responses = probe_model(model,exp_real,'conditioning')
            responses = get_responses(responses)
            responses['seed'] = seed
            responses['hidden_size'] = hidden_size
            responses['condition'] = 'conditioning'
            responses['test'] = False
            responses['trained'] = False
            responses['naive'] = naive
            responses.to_parquet(f'{hidden_size}_{seed}_conditioning_real_naive{naive}_initial.parquet')

            responses = probe_model(model,exp_real,'degradation')
            responses = get_responses(responses)
            responses['seed'] = seed
            responses['hidden_size'] = hidden_size
            responses['condition'] = 'degradation'
            responses['test'] = False
            responses['trained'] = False
            responses['naive'] = naive
            responses.to_parquet(f'{hidden_size}_{seed}_degradation_real_naive{naive}_initial.parquet')

            responses = probe_model(model,exp_real,'cue-c')
            responses = get_responses(responses)
            responses['seed'] = seed
            responses['hidden_size'] = hidden_size
            responses['condition'] = 'cue-c'
            responses['test'] = False
            responses['trained'] = False
            responses['naive'] = naive
            responses.to_parquet(f'{hidden_size}_{seed}_cue-c_real_naive{naive}_initial.parquet')

        else:
            responses = probe_model(model,exp_test,mode)
            responses = get_responses(responses)
            responses['seed'] = seed
            responses['hidden_size'] = hidden_size
            responses['condition'] = mode
            responses['test'] = True
            responses['trained'] = True
            responses['naive'] = naive
            responses.to_parquet(f'{hidden_size}_{seed}_{mode}_naive{naive}_pretrained.parquet')

            responses = probe_model(model,exp_real,mode)
            responses = get_responses(responses)
            responses['seed'] = seed
            responses['hidden_size'] = hidden_size
            responses['condition'] = mode
            responses['test'] = False
            responses['trained'] = True
            responses['naive'] = naive
            responses.to_parquet(f'{hidden_size}_{seed}_{mode}_real_naive{naive}_pretrained.parquet')
# %%
