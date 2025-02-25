%run_belief_state_model
%before running, generate simulated_trials.parquet using generate_simulated_experiments.m

%trial_table = parquetread('simulated_trials.parquet');

trans_p = 1-10.^(-4:0.1:-0.1);
parfor k = 1:length(trans_p)
    params = struct();
    params.alpha = 0.1;

    wait_time = rand()*20;
    pause(wait_time)
    params.gamma = 0.925;
    for i = 1:25
        for j = 1:3
            file_name = sprintf('td_sim_rep_%d_testgroup_%d_gamma_%.2f_model_switch_%d_trans_p_%.6f.parquet',i,j,params.gamma,4,trans_p(k));
            %if file name exists in subfolder v3, skip
            if isfile(fullfile('v3',file_name))
                continue
            end
            %data = trial_table(trial_table.rep==i&trial_table.testgroup==j,:);
            data = parquetread('simulated_trials.parquet');
            data = data(data.rep==i&data.testgroup==j,:);
            condition_only = data(data.phase==1,:);
            test_only = data(data.phase==2,:);

            T = calculate_transition(condition_only.states,19);
            O = calculate_observation(condition_only.events,condition_only.states,19,5);

            T_test = calculate_transition(test_only.states,19);
            O_test = calculate_observation(test_only.events,test_only.states,19,5);

            T_test(18,18)= trans_p(k);
            T_test(18,19) = 1-trans_p(k);


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
            save_table = addTableVariable(save_table,'trans_p',trans_p(k));

            

            parquetwrite(file_name,save_table);

        end
    end
end