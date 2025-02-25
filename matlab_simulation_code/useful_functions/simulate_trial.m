function [states, events] = simulate_trial(trialType,params)

%This function generates individual trials, including trailing ITI time
ITIfinalState = max(params.states.ITIwait);
%select a random number from an exponential distribution with mean ITI
%ITIlength = exprnd(params.ITImean);
ITIlength = exprnd(6);
while ITIlength < 4 || ITIlength > 10
    ITIlength = exprnd(6);
end
ITIlength = ITIlength-4;
nITI = ceil(ITIlength/params.dt);
ITIs = repmat(ITIfinalState,1,nITI);
ITIe = ones(1,nITI);

preCue_states = repmat(ITIfinalState,2,1);
preCue_events = ones(1,2);

pRand = rand;

%generate the trial
switch trialType
    case 'CSp'
        states = quickConcat(preCue_states, params.states.ISI,params.states.ITIwait,ITIs);
        events = quickConcat(preCue_events, params.events.ISI, params.events.ITIwait, ITIe);
        %replace cue and reward events
        events(events==-99)=3; %cue identity
        %return 2 for reward, 1 for no reward based on probability p
        if pRand < params.p
            events(events==0)=2; % reward
        else
            events(events==0)=1; % no reward
        end
    case 'CSm'
        states = quickConcat(preCue_states, params.states.ITIwait,ITIs,repmat(ITIfinalState,size(params.states.ISI,1),1));
        events = quickConcat(preCue_events, params.events.ISI, params.events.ITIwait, ITIe);
        %replace cue and reward events
        events(events==-99)=4; %cue identity
        events(events==0)=1; % no reward
    case'blank'
        states = repmat(ITIfinalState,size(quickConcat(preCue_events, params.events.ISI, params.events.ITIwait, ITIe)));
        events = ones(size(quickConcat(preCue_events, params.events.ISI, params.events.ITIwait, ITIe)));
    case'degrade'
        states = quickConcat(preCue_states, repmat(ITIfinalState,size(params.states.ISI)),params.states.ITIwait,ITIs);
        events = quickConcat(preCue_events, params.events.ISI, params.events.ITIwait, ITIe);
        %replace cue and reward events
        events(events==-99)=1; %cue identity is null
        %return 2 for reward, 1 for no reward based on probability p
        if pRand < params.p
            events(events==0)=2; % reward
        else
            events(events==0)=1; % no reward
        end
        %align reward
        events = quickConcat(1,events(1:end-1));
    case 'CSp2'
        states = quickConcat(preCue_states, params.states.ISI,params.states.ITIwait,ITIs);
        events = quickConcat(preCue_events, params.events.ISI, params.events.ITIwait, ITIe);
        %replace cue and reward events
        events(events==-99)=5; %cue identity
        %return 2 for reward, 1 for no reward based on probability p
        if pRand < params.p
            events(events==0)=2; % reward
        else
            events(events==0)=1; % no reward
        end
    case 'CSpExtinct'
        states = quickConcat(preCue_states, params.states.ISI,params.states.ITIwait,ITIs);
        events = quickConcat(preCue_events, params.events.ISI, params.events.ITIwait, ITIe);
        %replace cue and reward events
        events(events==-99)=3; %cue identity
        %no reward
        events(events==0)=1; % no reward
    case 'CSdegrade' %Garr simulation - need to check
        states = quickConcat(preCue_states, params.states.ISI,params.states.ITIwait);
        events = quickConcat(preCue_events, params.events.ISI, params.events.ITIwait);
        %replace cue and reward events
        events(events==-99)=3; %cue identity
        %return 2 for reward, 1 for no reward based on probability p
        if pRand < params.p
            events(events==0)=2; % reward
        else
            events(events==0)=1; % no reward
        end

        %rewards are possible at every 20s within ITI
        %find index of ITIs and ITIe corresponding to 20s
        %dt is 0.2s
        deg_ix = 1:20:nITI;
        deg_ix = deg_ix(1:end-1);

        %for each element in deg_ix, flip a coin and if heads, replace the
        %ITIe with 2
        for i = 1:length(deg_ix)
            if rand < 0.5
                ITIe(deg_ix(i)+1) = 2;
                %replace ITIs with ITIfinalState-1 between deg_ix(i) and deg_ix(i+1)-1
                if i == length(deg_ix)
                    ITIs(deg_ix(i)+1:end-2) = ITIfinalState-1;
                else
                    ITIs(deg_ix(i)+1:deg_ix(i+1)-2) = ITIfinalState-1;
                end
            end
        end
        states = quickConcat(states,ITIs);
        events = quickConcat(events,ITIe);
end

%if params.belief is true
max_isi_state = max(params.states.ISI)-1;

if params.belief
    %if there is CSp or CSp2, and rand < params.p, then add max_isi_state to the states
    if any(strcmp(trialType,{'CSp','CSp2','CSdegrade'})) && ~(pRand < params.p)
        states = states + max_isi_state;
        %state maximum value is ITIfinalState
        states(states>ITIfinalState) = ITIfinalState;
    end
    %change states that are greater than max_isi_state but less than ITIfinalState to be ITIfinalState-1
    states(states>max_isi_state & states<ITIfinalState) = ITIfinalState-1;

end