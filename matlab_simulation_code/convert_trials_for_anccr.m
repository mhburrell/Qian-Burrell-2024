trial_table = parquetread('simulated_trials.parquet');
anccr_trials = table();
for i = 1:25
    for j = 1:3 
        data = trial_table(trial_table.rep==i&trial_table.testgroup==j,:);
        event_log = trial_to_eventlog(data);
        event_log = addTableVariable(event_log,'rep',i);
        event_log = addTableVariable(event_log,'testgroup',j);
        event_log = addTableVariable(event_log,'phase',1);
        min_t = min(data.t(data.phase==2))*0.2;
        event_log.phase(event_log.t>min_t) = 2;
        anccr_trials = [anccr_trials;event_log];
    end
end
anccr_trials.t_id = (1:height(anccr_trials))';
parquetwrite('simulated_trials_anccr.parquet',anccr_trials);