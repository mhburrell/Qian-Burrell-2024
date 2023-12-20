function [states, events] = simulate_trial_garr(trialType,params)

%This function generates individual trials, including trailing ITI time
ITIfinalState = max(params.states.ITIwait);
%select a random number from an exponential distribution with mean ITI
ITIlength = exprnd(params.ITImean);
nITI = ceil(ITIlength/1); %time step is 1s
ITIs = repmat(ITIfinalState,1,nITI);
ITIe = ones(1,nITI);

preCue_states = repmat(ITIfinalState,2,1);
preCue_events = ones(1,2);

pRand = rand;
change_state = [];

%generate the trial
switch trialType
    case 'CSp'
        states = quickConcat(preCue_states, params.states.ISI,params.states.ITIwait,ITIs);
        events = quickConcat(preCue_events, params.events.ISI, params.events.ITIwait, ITIe);
        %replace cue and reward events
        events(events==-99)=3; %cue identity
        %return 2 for reward, 1 for no reward based on probability p
        if pRand < params.p
            e_idx = find(events==0);
            events(events==0)=1; % reward
            events(e_idx+1)=2;
        else
            events(events==0)=1; % no reward
        end
    case 'CSp2'
        states = quickConcat(preCue_states, params.states.ISI_2,params.states.ITIwait,ITIs);
        events = quickConcat(preCue_events, params.events.ISI, params.events.ITIwait, ITIe);
        %replace cue and reward events
        events(events==-99)=4; %cue identity
        %return 2 for reward, 1 for no reward based on probability p
        if pRand < params.p
            e_idx = find(events==0);
            events(events==0)=1; % reward
            events(e_idx+1)=5;
        else
            events(events==0)=1; % no reward
        end
    case 'CSdegrade'
        states = quickConcat(preCue_states, params.states.ISI,params.states.ITIwait);
        events = quickConcat(preCue_events, params.events.ISI, params.events.ITIwait);
        %replace cue and reward events
        events(events==-99)=3; %cue identity
        %return 2 for reward, 1 for no reward based on probability p
        if pRand < params.p
            e_idx = find(events==0);
            events(events==0)=1; % reward
            events(e_idx+1)=2;
        else
            events(events==0)=1; % no reward
        end

        %rewards are possible at every 20s within ITI
        %find index of ITIs and ITIe corresponding to 20s
        %dt is 1s
        deg_ix = 1:20:nITI;
        deg_ix = deg_ix(1:end-1);
        change_state = [];
        

        %for each element in deg_ix, flip a coin and if heads, replace the
        %ITIe with 2
        for i = 1:length(deg_ix)
            if rand < 0.5
                ITIe(deg_ix(i)+1) = 2;
                
                change_state = [change_state,deg_ix(i)+length(states)];
                if i == 1
                    states(end)=ITIfinalState;
                else
                    ITIs(deg_ix(i)-1)=ITIfinalState;
                end
                %replace ITIs with ITIfinalState-1 between deg_ix(i) and deg_ix(i+1)-1
                if i == length(deg_ix)
                    ITIs(deg_ix(i)+1:end-2) = ITIfinalState-1;
                else
                    ITIs(deg_ix(i)+1:deg_ix(i+1)-10) = ITIfinalState-1;
                end
            end
        end
        states = quickConcat(states,ITIs);
        events = quickConcat(events,ITIe);
    case 'CSp2degrade'
        states = quickConcat(preCue_states, params.states.ISI_2,params.states.ITIwait);
        events = quickConcat(preCue_events, params.events.ISI, params.events.ITIwait);
        %replace cue and reward events
        events(events==-99)=4; %cue identity
        %return 2 for reward, 1 for no reward based on probability p
        if pRand < params.p
            e_idx = find(events==0);
            events(events==0)=1; % reward
            events(e_idx+1)=5;
        else
            events(events==0)=1; % no reward
        end

        %rewards are possible at every 20s within ITI
        %find index of ITIs and ITIe corresponding to 20s
        %dt is 1s
        deg_ix = 1:20:nITI;
        deg_ix = deg_ix(1:end-1);

        %for each element in deg_ix, flip a coin and if heads, replace the
        %ITIe with 2
        change_state = [];
        for i = 1:length(deg_ix)
            if rand < 0.5
                ITIe(deg_ix(i)+1) = 2;
                change_state = [change_state,deg_ix(i)+length(states)];
                if i == 1
                    states(end)=ITIfinalState;
                else
                    ITIs(deg_ix(i)-1)=ITIfinalState;
                end
                %replace ITIs with ITIfinalState-1 between deg_ix(i) and deg_ix(i+1)-1
                if i == length(deg_ix)
                    ITIs(deg_ix(i)+1:end-2) = ITIfinalState-1;
                else
                    ITIs(deg_ix(i)+1:deg_ix(i+1)-10) = ITIfinalState-1;
                end
            end
        end
        states = quickConcat(states,ITIs);
        events = quickConcat(events,ITIe);
end

%if params.belief is true
max_isi_state = max(params.states.ISI_2);

if params.belief
    %if there is CSp or CSp2, and rand < params.p, then add max_isi_state to the states
    if any(strcmp(trialType,{'CSp','CSp2','CSdegrade','CSp2degrade'})) && ~(pRand < params.p)

        change_states = find(states<ITIfinalState-1);
        states(change_states) = states(change_states) + max_isi_state-min(states)+1;
        %state maximum value is ITIfinalState
        states(states>ITIfinalState) = ITIfinalState;
    end
    %change states that are greater than max_isi_state but less than ITIfinalState to be ITIfinalState-1
    states(states>max_isi_state & states<ITIfinalState) = ITIfinalState-1;
    if ~isempty(change_state)
        %states(change_state)=20;
    end

end