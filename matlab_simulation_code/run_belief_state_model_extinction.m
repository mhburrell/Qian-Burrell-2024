%run_belief_state_model
discount_factor = 0.925;
parfor i = 1:25
    params = struct();
    params.alpha = 0.1;

    wait_time = rand()*20;
    pause(wait_time)

    trial_table = parquetread('simulated_extinction_trials.parquet');
    trial_table = trial_table(trial_table.rep==i,:);

    for k = 1:length(discount_factor)
        params.gamma = discount_factor(k);
        for j = 4:4
            data = trial_table(trial_table.rep==i&trial_table.testgroup==j,:);
            condition_only = data(data.phase==1,:);
            test_only = data(data.phase==2,:);

            T = calculate_transition(condition_only.states,19);
            O = calculate_observation(condition_only.events,condition_only.states,19,5);

            T_test = T;
            O_test = O;

            params.T_test = T_test;
            params.O_test = O_test;
            params.t_switch = min(data.t(data.phase==2));

            results = td_belief(data.events,O,T,data.states,params);
            save_table = table;

            save_table.rpe = results.rpe;
            save_table.value = results.value;
            save_table.t = (1:height(save_table))';
            save_table = addTableVariable(save_table,'rep',i);
            save_table = addTableVariable(save_table,'testgroup',j);
            save_table = addTableVariable(save_table,'discount_factor',params.gamma);

            file_name = sprintf('td_sim_rep_%d_testgroup_%d_gamma_%.2f_model_switch_%d.parquet',i,j,params.gamma,4);

            parquetwrite(file_name,save_table);

        end
    end
end