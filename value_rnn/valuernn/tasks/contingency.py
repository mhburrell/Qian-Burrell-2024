#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:31:01 2022

@author: mobeets
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from .trial import Trial, get_itis
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Contingency(Dataset):
    def __init__(self, 
                mode='conditioning',
                rew_times=[9, 9, 9],
                rew_sizes=[1, 1, 1],
                # cue_shown=[True, True, False],
                rew_probs=None, cue_shown=None,
                cue_probs=[0.4, 0.2, 0.4],
                jitter=1,
                ntrials=1000,
                ntrials_per_episode=20,
                iti_min=20, iti_p=1/4, iti_max=0, iti_dist='geometric',
                t_padding=0, include_reward=True,
                include_unique_rewards=True,
                omission_trials_have_duration=True,
                include_null_input=False,
                long_cues=False):
        self.ntrials = ntrials
        self.cue_probs = cue_probs
        self.rew_sizes = rew_sizes
        self.cue_shown = cue_shown
        self.jitter = jitter
        self.mode = mode
        self.rew_times = rew_times
        self.rew_probs = rew_probs
        self.long_cues = long_cues
        if self.mode not in [None, 'conditioning', 'degradation', 'cue-c','garr2023']:
            raise Exception("Invalid mode. Must be one of [None, 'conditioning', 'degradation', 'cue-c','garr2023']")
        if self.mode is not None:
            if self.rew_probs is not None or self.cue_shown is not None:
                raise Exception("If setting mode, cannot set rew_probs or cue_shown")
            else:
                if self.mode.lower() == 'conditioning':
                    self.rew_probs = [0.75, 0, 0]
                    self.cue_shown = [True, True, False]
                    self.nrewards = 1
                elif self.mode.lower() == 'degradation':
                    self.rew_probs = [0.75, 0, 0.75]
                    self.cue_shown = [True, True, False]
                    self.nrewards = 1
                elif self.mode.lower() == 'cue-c':
                    self.rew_probs = [0.75, 0, 0.75]
                    self.cue_shown = [True, True, True]
                    self.nrewards = 1
                elif self.mode == 'garr2023':
                    self.cue_probs = [0.5, 0.5, 0]
                    self.rew_sizes = [[1,0], [0,1], [0,1]]
                    self.nrewards = len(self.rew_sizes[0])
                    self.rew_probs = [0.5, 0.5, 0.5]
                    self.cue_shown = [True, True, False]
                else:
                    raise Exception("Unrecognized mode")

        assert self.rew_probs is not None
        self.ncues_shown = sum(self.cue_shown)
        self.ncues = len(self.cue_probs)
        self.ntrials_per_cue = np.round(np.array(self.cue_probs) * self.ntrials).astype(int)
        self.ntrials_per_episode = ntrials_per_episode
        
        self.iti_min = iti_min
        self.iti_max = iti_max # n.b. only used if iti_dist == 'uniform'
        self.iti_p = iti_p
        self.iti_dist = iti_dist
        self.t_padding = t_padding
        self.include_reward = include_reward
        self.include_unique_rewards = include_unique_rewards
        self.include_null_input = include_null_input
        self.omission_trials_have_duration = omission_trials_have_duration
        self.make_trials()
        if not all(self.cue_shown) and not all([self.cue_shown[i] for i in range(self.cue_shown.index(False)-1)]):
            raise Exception("All hidden cues must be listed last")
        if self.iti_max != 0 and self.iti_dist != 'uniform':
            raise Exception("Cannot set iti_max>0 unless iti_dist == 'uniform'")
            
    def make_trial(self, cue, iti):
        rew_prob = self.rew_probs[cue]
        rew_size = self.rew_sizes[cue] if np.random.rand() <= rew_prob else 0
        if self.nrewards > 1 and rew_size == 0:
            rew_size = [0]*self.nrewards

        isi = int(self.rew_times[cue])
        if rew_size == 0 and not self.omission_trials_have_duration:
            isi = 0
        if isi > 0 and self.jitter > 0:
            isi += np.random.choice(np.arange(-self.jitter, self.jitter+1))
        assert isi >= 0
        
        # n.b. including input for all cues, even if they're not shown
        return Trial(cue, iti, isi, rew_size, self.cue_shown[cue], self.ncues, self.t_padding, self.include_reward, self.include_null_input)

    def shuffle_trials_proportionally(self):
        trials_per_block = 20

        #ncues is the number of cues
        #cue_probs is the probability of each cue
        #ntrials is the number of trials

        #calculate number of blocks
        nblocks = int(self.ntrials / trials_per_block)

        #calculate number of trials per cue per block
        trials_per_cue_per_block = np.round(np.array(self.cue_probs) * trials_per_block).astype(int)

        #create the sequence of trials for one block
        block = np.hstack([c*np.ones(n).astype(int) for c,n in zip(range(trials_per_block), trials_per_cue_per_block)])

        #for each block, shuffle the trials and add to the list of trials
        self.cues = np.hstack([np.random.permutation(block) for i in range(nblocks)])
    
    
    def make_trials(self, cues=None, ITIs=None):
        if cues is None and self.mode != 'garr2023':
            # create all trials
            #self.cues = np.hstack([c*np.ones(n).astype(int) for c,n in zip(range(self.ncues), self.ntrials_per_cue)])
            
            # shuffle trial order
            #np.random.shuffle(self.cues)
            self.shuffle_trials_proportionally()
        elif cues is None and self.mode == 'garr2023':
            self.cues = np.hstack([c*np.ones(n).astype(int) for c,n in zip(range(self.ncues), self.ntrials_per_cue)])
            np.random.shuffle(self.cues)
            #between each trial, add a number of cue 2 trials
            #draw the number of cue 2 trials from a geometric distribution with mean 12
            #the geometric distribution is parameterized by p = 1/mean

            #draw the number of cue 2 trials
            n_cue2_trials = np.random.geometric(1/12, len(self.cues))
            cue_list = []
            for i in range(len(self.cues)):
                cue_list.append(self.cues[i])
                cue_list.extend([2]*n_cue2_trials[i])
            self.cues = np.array(cue_list)


        else:
            self.cues = cues
        
        # ITI per trial
        self.ITIs = get_itis(self) if ITIs is None else ITIs
        
        # make trials
        self.trials = [self.make_trial(cue, iti) for cue, iti in zip(self.cues, self.ITIs)]

        if self.long_cues:
            #for each trial, set the cue to be one for the length of the ISI
            #the cues are the first two columns of trial.X
            for trial in self.trials:
                if trial.cue == 0:
                    trial.X[trial.iti:(trial.iti+trial.isi),0] = 1
                elif trial.cue == 1:
                    trial.X[trial.iti:(trial.iti+trial.isi),1] = 1
        
        # stack trials to make episodes
        self.episodes = self.make_episodes(self.trials, self.ntrials_per_episode)
    
    def make_episodes(self, trials, ntrials_per_episode):
        # concatenate multiple trials in each episode
        episodes = []
        for t in np.arange(0, len(trials)-ntrials_per_episode+1, ntrials_per_episode):
            episode = trials[t:(t+ntrials_per_episode)]
            for ti, trial in enumerate(episode):
                trial.index_in_episode = ti
            episodes.append(episode)
        return episodes
    
    def __getitem__(self, index):
        episode = self.episodes[index]
        X = np.vstack([trial.X for trial in episode])
        y = np.vstack([trial.y for trial in episode])
        trial_lengths = [len(trial) for trial in episode]
        return (torch.from_numpy(X).to(device), torch.from_numpy(y).to(device), trial_lengths, episode)
    
    def __len__(self):
        return len(self.episodes)
