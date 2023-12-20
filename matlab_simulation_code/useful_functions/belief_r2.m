function [r2,rc,r2_col,event_train] = belief_r2(event_train,wait_state)
persistent cond_t degrade_t cuec_t

event_train.e = event_train.X1*3+event_train.X2*4+event_train.X3*5+event_train.X4*2;
event_train.e(event_train.e==0)=1;

%conditions = ['conditioning','degradation','cue-c'];

cond = event_train.condition(1);
seed = event_train.seed(1);

saveParams = struct();
saveParams.alpha = 0.1;
saveParams.gamma = 0.92;
saveParams.iter = 1;
saveParams.iti = 2;
saveParams.belief = 1;
saveParams.t_switch = 10^9;
nSims = 4000;
rng(seed);

if isempty(cond_t)
    cond_t = simulate_experiment_rnn(nSims,'conditioning');
end
if isempty(degrade_t)
    degrade_t = simulate_experiment_rnn(nSims,'degradation');
end
if isempty(cuec_t)
    cuec_t = simulate_experiment_rnn(nSims,'odor C');
end
max_state = max(cuec_t.states);

T_condition = calculate_transition(cond_t.states,max_state);
O_condition = calculate_observation(cond_t.events,cond_t.states,max_state,5);

T_degrade = calculate_transition(degrade_t.states,max_state);
O_degrade = calculate_observation(degrade_t.events,degrade_t.states,max_state,5);

T_odorc = calculate_transition(cuec_t.states,max_state);
O_odorc = calculate_observation(cuec_t.events,cuec_t.states,max_state,5);


if strcmp(cond,'conditioning')
    T = T_condition;
    O = O_condition;
elseif strcmp(cond,'degradation')
    T = T_degrade;
    O = O_degrade;
elseif strcmp(cond,'cue-c')
    T = T_odorc;
    O = O_odorc;
else
    disp('error')
end

zero_fill = zeros(height(event_train),1);

event_train.b1 = zero_fill;
event_train.b2 = zero_fill;
event_train.b3 = zero_fill;
event_train.b4 = zero_fill;
event_train.b5 = zero_fill;
event_train.b6 = zero_fill;
event_train.b7 = zero_fill;
event_train.b8 = zero_fill;
event_train.b9 = zero_fill;

if wait_state
    
    event_train.b10 = zero_fill;
    ix = event_train.condition==cond&event_train.seed==seed;
    x = event_train.e(ix);
    results = td_belief(x,O,T,[],saveParams,0);
    for i = 1:10
        event_train.(['b',num2str(i)])(ix) = results.b(:,i);
    end
else
    ix = event_train.condition==cond&event_train.rep==rep;
    x = event_train.e(ix);
    results = td_belief(x,O,T,[],saveParams,0);
    for i = 1:7
        event_train.(['b',num2str(i)])(ix) = results.b(:,i);
    end
    event_train.b8=results.b(:,8)+results.b(:,9);
end






fnames = fieldnames(event_train);
%find fnames that begin with b
b_ix = cellfun(@(x) x(1)=='b',fnames);
%find fnames that begin with Z
Z_ix = cellfun(@(x) x(1)=='Z',fnames(1:end-1));

B = table2array(event_train(ix,b_ix));
Z = table2array(event_train(ix,Z_ix));

Z = [Z,ones(length(ix),1)];
% B(:,8) = B(:,8)+B(:,9);
% B = B(:,1:8);

% Z = normalize(Z);
% cv = cvpartition(size(B,1),'HoldOut',0.2);
% idx = cv.test;
% B_train = B(~idx,:);
% B_test = B(idx,:);
% Z_test = Z(idx,:);
% Z = Z(~idx,:);

B_train = B;
Z_train = Z;
Z_test= Z;
B_test = B;

W = (Z'*Z)\Z'*B_train;
rc = rcond(Z'*Z);
W = zeros(size(Z,2),size(B,2));
for i = 1:size(B,2)
    W(:,i) = lsqminnorm(Z,B_train(:,i));
end

numer = B_test-Z_test*W;
denom = B_test;

numer_diff = numer - repmat(mean(numer),size(numer,1),1);
denom_diff = denom - repmat(mean(denom),size(denom,1),1);
var_numer = 0;
var_denom = 0;
for i = 1:size(numer,1)
    var_numer = var_numer + norm(numer_diff(i,:))^2;
    var_denom = var_denom + norm(denom_diff(i,:))^2;
end

r2 = 1 - var_numer/var_denom;
r2_col = r2_matrix(B_test,Z_test*W);