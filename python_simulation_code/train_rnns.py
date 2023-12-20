#%%
import torch
import sys
import os
import numpy as np
import pandas as pd
from copy import deepcopy
from valuernn.tasks.contingency import Contingency
from valuernn import model as vmodel
from valuernn import train as vtrain
from rnn_analysis.analysis import get_conditions_contingency, get_exemplars
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#the first argument is the seed
input_seed = int(sys.argv[1])

#%%
def make_exp(seed,is_test=False):
    Exps = {}
    modes = ['conditioning', 'degradation', 'cue-c']

    np.random.seed(seed)
    for mode in modes:
        if is_test:
            E = Contingency(mode=mode,ntrials=1000,ntrials_per_episode=20,jitter=0,rew_times=[8,8,8])
        else:
            E = Contingency(mode=mode,ntrials=10000,ntrials_per_episode=20,jitter=1,rew_times=[8,8,8])
        Exps[mode] = E

    return Exps    

def make_rnn(hidden_size,seed):
    np.random.seed(seed)
    gamma = 0.83
    model = vmodel.ValueRNN(input_size=4, hidden_size=hidden_size, gamma=gamma)
    model.to(device)
    return(model)

def make_dataloader(Exps,mode):
    batch_size = 12
    E = Exps[mode]
    dataloader = vtrain.make_dataloader(E,batch_size=batch_size)
    return(dataloader)

def probe_model(model,Exps,mode):
    dataloader = make_dataloader(Exps,mode)
    responses = vtrain.probe_model(model,dataloader)
    return(responses)

def train_model(model,Exps,mode):
    lr = 0.001
    dataloader = make_dataloader(Exps,mode)
    scores,test_scores,weights = vtrain.train_model(model,dataloader,print_every=10,epochs = 400,save_every=1, lr=lr)
    ix = np.argmin(scores) # use best model
    print(ix)
    best_weights = weights[ix]
    return(scores)

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
# %%
if __name__ == '__main__':
    seeds = np.arange(200)
    hidden_sizes = [5,10,15,20,25,30,40,50]*25
    # on every second repetition, load the weights from the previous repetition
    for seed, hidden_size in tqdm(zip(seeds, hidden_sizes)):
        #if the seed + input seed mod 17 is not 0, skip
        if (seed+input_seed)%53 != 0:
            continue
        #print the seed and hidden size
        print(seed,hidden_size)
        Exps = make_exp(seed)
        #Exps_test = make_exp(seed*100,is_test=True)
        model = make_rnn(hidden_size,seed)

        initial_weights = deepcopy(model.state_dict())
        #write weights to file
        torch.save(initial_weights,f'{hidden_size}_{seed}_initial_weights.pt')


        #check if file exists, if so, load, otherwise train and save
        if os.path.isfile(f'{hidden_size}_{seed}_conditioning_weights.pt'):
            cond_weights = torch.load(f'{hidden_size}_{seed}_conditioning_weights.pt')
            model = load_weights(model,cond_weights)
        else:
            cond_scores = train_model(model,Exps,'conditioning')
            cond_weights = deepcopy(model.state_dict())
            #write weights and scores to separate files
            torch.save(cond_weights,f'{hidden_size}_{seed}_conditioning_weights.pt')
            np.save(f'{hidden_size}_{seed}_conditioning_scores.npy',cond_scores)


        if os.path.isfile(f'{hidden_size}_{seed}_degradation_weights.pt'):
            degrade_weights = torch.load(f'{hidden_size}_{seed}_degradation_weights.pt')
        else:
            degrade_scores = train_model(model,Exps,'degradation')
            degrade_weights = deepcopy(model.state_dict())
            #write weights and scores to separate files
            torch.save(degrade_weights,f'{hidden_size}_{seed}_degradation_weights.pt')
            np.save(f'{hidden_size}_{seed}_degradation_scores.npy',degrade_scores)

        if os.path.isfile(f'{hidden_size}_{seed}_cue-c_weights.pt'):
            cuec_weights = torch.load(f'{hidden_size}_{seed}_cue-c_weights.pt')
        else:
            #reload conditioning weights
            model = load_weights(model,cond_weights)
            cuec_scores = train_model(model,Exps,'cue-c')
            cuec_weights = deepcopy(model.state_dict())
            #write weights and scores to separate files
            torch.save(cuec_weights,f'{hidden_size}_{seed}_cue-c_weights.pt')
            np.save(f'{hidden_size}_{seed}_cue-c_scores.npy',cuec_scores)


        if os.path.isfile(f'{hidden_size}_{seed}_naive_degradation_weights.pt'):
            naive_degrade_weights = torch.load(f'{hidden_size}_{seed}_naive_degradation_weights.pt')
        else:
            #reload initial weights for naive degradation
            model = load_weights(model,initial_weights)
            naive_degrade_scores = train_model(model,Exps,'degradation')
            naive_degrade_weights = deepcopy(model.state_dict())
            #write weights and scores to separate files
            torch.save(naive_degrade_weights,f'{hidden_size}_{seed}_naive_degradation_weights.pt')
            np.save(f'{hidden_size}_{seed}_naive_degradation_scores.npy',naive_degrade_scores)

        #reload initial weights for naive cue-c
        if os.path.isfile(f'{hidden_size}_{seed}_naive_cue-c_weights.pt'):
            naive_cuec_weights = torch.load(f'{hidden_size}_{seed}_naive_cue-c_weights.pt')
        else:
            model = load_weights(model,initial_weights)
            naive_cuec_scores = train_model(model,Exps,'cue-c')
            naive_cuec_weights = deepcopy(model.state_dict())
            #write weights and scores to separate files
            torch.save(naive_cuec_weights,f'{hidden_size}_{seed}_naive_cue-c_weights.pt')
            np.save(f'{hidden_size}_{seed}_naive_cue-c_scores.npy',naive_cuec_scores)


