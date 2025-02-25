function [trial_table] = simulate_experiment_rnn(nTrials,idExperiment)
%simulate_experiment generates the sequence of states and rewards for use
%in TD learning

%Inputs:
%   INPUTS:
%       nTrials: number of trials to generate
%       idExperiment: id of the experiment to generate the data for
%   OUTPUTS:
%       states: vector of state identity
%       events: vector of reward identity
%             1: null, 2: reward, 3 or higher: CS

params = struct();
params.ITImean = 2; %mean ITI exponential distribution
params.p = 0.75; %probability of reward
params.belief = 1; %use belief state configuration

params.dt = 0.5; %time step

cue_length = 1;
reward_delay = 2.5;
ISI_length = cue_length + reward_delay;
params.states.ISI = (1:floor(ISI_length/params.dt)+2)';
params.events.ISI = ones(numel(params.states.ISI),1);
%replace first and last ISI event with -99 and 0, respectively
params.events.ISI(1) = -99;
params.events.ISI(end) = 0;
%replace -99 with CS identity and 0 with reward or null inside of trialGenerator
max_isi_state = max(params.states.ISI);

post_reward = 4;
pre_reward = 2;
wait = post_reward + pre_reward;
params.states.ITIwait = (max_isi_state:max_isi_state+floor(wait/params.dt))';
params.events.ITIwait = ones(numel(params.states.ITIwait),1);

%generate trial types
switch idExperiment
    case 'conditioning'
        trialDistribution = {'CSp','CSp','CSm','blank','blank'};
    case 'degradation'
        trialDistribution = {'CSp','CSp','CSm','degrade','degrade'};
    case 'odor C'
        trialDistribution = {'CSp','CSp','CSm','CSp2','CSp2'};
    case 'extinction'
        trialDistribution = {'CSpExtinct','CSpExtinct','CSm','blank','blank'};
end

%generate trial sequence
trialSequence = randsample(trialDistribution,nTrials,true);

%generate states and events
states = max(params.states.ITIwait);

numTrials = numel(trialSequence);
statesCell = cell(numTrials, 1);
eventsCell = cell(numTrials, 1);
trialNumbers = zeros(numTrials, 1);
trialTypes = cell(numTrials, 1);

for i = 1:numTrials
    [s, e] = simulate_trial(trialSequence{i}, params);
    statesCell{i} = s';
    eventsCell{i} = e';
    trialNumbers(i) = i;
    trialTypes{i} = trialSequence{i};
end

% Create table in one go
tt = table(statesCell, eventsCell, trialNumbers, trialTypes, ...
                    'VariableNames', {'states', 'events', 'trialnumber', 'trialtype'});

L = cellfun(@numel,tt.states);
trial_table = repelem(tt,L,1);
trial_table.states = vertcat(tt.states{:});
trial_table.events = vertcat(tt.events{:});


states = trial_table.states;
unique_states = unique(states);
%ensure unique_states is in order
unique_states = sort(unique_states);
for i = 1:length(unique_states)
    states(states==unique_states(i)) = i;
end
trial_table.states = states;