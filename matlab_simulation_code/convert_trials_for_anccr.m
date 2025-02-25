trial_table = parquetread('simulated_trials.parquet');
anccr_trials = table();
for i = 1:25
    for j = 1:3 
        data = trial_table(trial_table.rep==i&trial_table.testgroup==j,:);
        event_log = trial_to_eventlog(data);
        event_log = addTableVariable(event_log,'IRI',0);
        event_log = addTableVariable(event_log,'ephase',1);
        event_log = addTableVariable(event_log,'omission_id',5);
        event_log = addTableVariable(event_log,'rep',i+25*(j-1));        
        min_t = double(min(data.t(data.phase==2)))*0.2;
        event_log.ephase(event_log.times>min_t) = 2;
        anccr_trials = [anccr_trials;event_log];
    end
end
anccr_trials.t_id = (1:height(anccr_trials))';

for i = 1:75
    for j = 1:2
        current_data = anccr_trials(anccr_trials.rep==i&anccr_trials.ephase==j,:);
        rewards_times = current_data.times(current_data.reward==1);
        IRI = mean(diff(rewards_times));
        anccr_trials.IRI(anccr_trials.rep==i&anccr_trials.ephase==j)=IRI;
    end
end

parquetwrite('simulated_trials_anccr.parquet',anccr_trials);