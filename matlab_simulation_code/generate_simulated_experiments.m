%generates the experiment sequences used for simulating TD and ANCCR models
%25 reps of each experiment
%condition - condition
%condition - degradation
%condition - cuedus

trial_table = table();

test_phases = {'conditioning','degradation','odor C'};

for i = 1:1
    %generate common conditioning
    rng(i);
    temp_table_cond = simulate_experiment(8000,'conditioning');
    temp_table_cond = addTableVariable(temp_table_cond,'rep',i);
    temp_table_cond = addTableVariable(temp_table_cond,'phase',1);
    for j = 1:3
        %generate test phase
        rng(i*100+j);
        temp_table_test = simulate_experiment(8000,test_phases{j});
        temp_table_test = addTableVariable(temp_table_test,'rep',i);
        temp_table_test = addTableVariable(temp_table_test,'phase',2);
        %combine and add time index
        add_table = [temp_table_cond;temp_table_test];
        add_table = addTableVariable(add_table,'testgroup',j);
        add_table.t = (1:height(add_table))';
        trial_table = [trial_table;add_table];
    end
end

%parquetwrite('simulated_trials_long.parquet',trial_table);
