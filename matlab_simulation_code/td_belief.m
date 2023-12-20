function results = td_belief(x,O,T,trueState,params,update_OT)
% TD learning for partially observable MDP.
% Original Author: Dr. Samuel J. Gershman

% initialization
S = size(T,1);      % number of states
b = ones(S,1)/S;    % belief state
w = zeros(S,1);     % weights


results.w = zeros(length(x),S);
results.b = zeros(length(x),S);
nEvents=max(x);

alpha = params.alpha;
gamma = params.gamma;

resets = 0;

if nargin<6
    update_OT = 0;
end

for t = 1:length(x)

    if update_OT
        if t>5000
            if mod(t,20)==0
                T = calculate_transition(trueState(t-2500:t),S);
                O = calculate_observation(x(t-2500:t),trueState(t-2500:t),S,nEvents);
            end
        end
    end

    if t>params.t_switch
        T = params.T_test;
        O = params.O_test;
    end

    b0 = b; % old posterior, used later
    b = b'*(T.*squeeze(O(:,:,x(t))));
    b=b';
    b = b./sum(b);

    if isnan(b)
        %disp('resetting beliefs')
        b(isnan(b))=0;
        b(end)=1;
        resets = resets+1;
    end


    % TD update
    w0 = w;
    r = double(x(t)==2);        % reward
    rpe = r + w'*(gamma*b-b0);
    w = w + alpha*rpe*b0;         % weight update


    results.w(t,:) = w0;
    results.b(t,:) = b0;
    results.rpe(t,1) = rpe;
    results.value(t,1) = w'*(b0); %estimated value



end
disp(num2str(resets ));