%generates the experiment sequences used for simulating TD and ANCCR models
%25 reps of each experiment
%condition - condition
%condition - degradation
%condition - cuedus

test_phases = {'conditioning','degradation','odor C'};

results = cell(25,3);

for i = 1:25
    %generate common conditioning
    rng(i);
    temp_table_cond = simulate_experiment(4000,'conditioning');
    temp_table_cond = addTableVariable(temp_table_cond,'rep',i);
    temp_table_cond = addTableVariable(temp_table_cond,'phase',1);
    for j = 1:3
        %generate test phase
        rng(i*100+j);
        temp_table_test = simulate_experiment(4000,test_phases{j});
        temp_table_test = addTableVariable(temp_table_test,'rep',i);
        temp_table_test = addTableVariable(temp_table_test,'phase',2);
        %combine and add time index
        add_table = [temp_table_cond;temp_table_test];
        add_table = addTableVariable(add_table,'testgroup',j);
        add_table.t = (1:height(add_table))';
        results{i,j} = add_table;
    end

end

trial_table = vertcat(results{:});

%convert to int
trial_table.states = int8(trial_table.states);
trial_table.events = int8(trial_table.events);
trial_table.trialnumber = int16(trial_table.trialnumber);
trial_table.rep = int8(trial_table.rep);
trial_table.phase = int8(trial_table.phase);
trial_table.testgroup = int8(trial_table.testgroup);
trial_table.t = int32(trial_table.t);

parquetwrite('simulated_trials.parquet',trial_table);

%extinction simulation
extinction = trial_table(trial_table.testgroup==1,:);
extinction.events(extinction.events==2&extinction.phase==2)=1;
extinction.testgroup(:)=4;

parquetwrite('simulated_extinction_trials.parquet',extinction);