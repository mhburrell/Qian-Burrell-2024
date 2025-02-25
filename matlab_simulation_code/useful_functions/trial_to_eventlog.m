function event_log = trial_to_eventlog(data)

data = data(data.events~=1,:);
event_log = table();
event_log.events = data.events-1;
event_log.times = double(data.t)*0.2;
event_log.reward = zeros(size(data.t));
event_log.reward(event_log.events==1)=1;

cue_times = [event_log.times(event_log.events==2);event_log.times(event_log.events==4)];
reward_times = cue_times + 3.4;

for i = 1:length(reward_times)
    v = reward_times(i);
    min_d = min(abs(event_log.times-v));
    if min_d < 0.0001
        % Value exists in the table, set r to 1
%         event_log.r(idx) = 1;
    else
        % Value does not exist, add a new row
        newRow = table();
        newRow.events = 5;
        newRow.times = v;
        newRow.reward = 0;
        event_log = [event_log; newRow];
    end
end



event_log = sortrows(event_log,'times');
% event_log = table2array(event_log);