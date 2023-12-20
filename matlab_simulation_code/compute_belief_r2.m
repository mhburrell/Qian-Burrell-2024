pds = parquetDatastore('C:\Users\Mark\Dev\LocalClones\Qian-2023-Contingency\r_code\for_belief_r2','IncludeSubfolders',true);
file_list = pds.Files;
conditions = {'conditioning','degradation','cue-c'};
results_table = table;
for i = 1:length(file_list)
    event_train = parquetread(file_list{i});
    seeds = unique(event_train.seed);
    for j= 1:length(seeds)
        seed = seeds(j);
        for k = 1:3
            condition = conditions(k);
            event_train_selection = event_train(event_train.seed==seed&event_train.condition==condition,:);
            [r2,rc,r2_col,~] = belief_r2(event_train_selection,1);
            new_row = table;
            new_row.seed = seed;
            new_row.condition = condition;
            new_row.r2 = r2;
            new_row.rc = rc;
            for c = 1:10
                new_row.(['r2_',num2str(c)])=r2_col(c);
            end
            results_table = [results_table;new_row];
        end
    end
end
