%run_anccr_model.m
%reads in simulated trials (run convert_trials_for_anccr.m first) and parameter set (generate_anccr_paramspace.m)
%can be run in parallel using array_index as input, including on a cluster (so long as results space is shared) - set chunk size to match number of cores
%v2

function run_anccr_model_paramchunk(param_chunk,input_file,save_directory,rep_range)

trial_table = parquetread(input_file);
trial_table = trial_table(ismember(trial_table.rep,rep_range),:);
%param_table = parquetread('anccr_param_table_newnewk.parquet');
%param_table = parquetread('small_theta_params.parquet');
%param_table = parquetread('reduced_param_table.parquet');
%param_table = parquetread('anccr_param_table_finew.parquet');
%param_table = parquetread('final_param_table.parquet');
param_table = parquetread('anccr_param_table.parquet');
param_table = sortrows(param_table,'p_id');



%common parameters
samplingperiod = 0.2;
minimumrate = 0.001;
maximumjitter = 0.1;
n_unique_events = max(unique(trial_table.events));
beta = zeros(1, n_unique_events);%[0 0 0 0 0 0 0 0 0 0]; %reward is first event, all other events start neutral
for i = 1:n_unique_events
    if sum(trial_table.reward(trial_table.events==i))>0
        beta(i) = 1;
    end
end
rew_id = find(beta);
%temp beta
%beta = [0 1 1 1 1 1 1 1 1 0];
t_cell = 1; %initialize table cell counter
%save results each 10 parameter sets
table_collect = cell(20,1);

% file_list = dir(fullfile(save_directory,'*.parquet'));
% if ~isempty(file_list)
%     file_list = {file_list.name};
%     file_list = cellfun(@(x) strsplit(x,'_'),file_list,'UniformOutput',false);
%     %min p is 4th element, max p is 6th element
%     min_p = cellfun(@(x) str2double(x{4}),file_list);
%     max_p = cellfun(@(x) str2double(x{6}),file_list);
%     %create sequence from min_p to max_p, concatenate into one vector
%     p_list = arrayfun(@(x,y) x:y,min_p,max_p,'UniformOutput',false);
%     p_list = horzcat(p_list{:});
%     param_chunk = param_chunk(~ismember(param_chunk,p_list));
% end

if ~isempty(param_chunk)
    np = length(param_chunk);

    for rr = 1:np
        p = param_chunk(rr);
        disp(num2str(p));
        alpha_anccr = param_table.alpha_anccr(p);
        k = param_table.k(p);
        alpha_r = param_table.alpha_anccr(p);
        w = param_table.w(p);
        threshold = param_table.theta(p);
        Tratio = param_table.Tratio(p);

        %initialize DA as empty
        trial_table.DA = zeros(size(trial_table.events));
        for i = rep_range
            %disp(i);
            event_log = trial_table(trial_table.rep==i,1:3);
            try
                omission_id =  max(trial_table.omission_id(trial_table.rep==i));
            catch
                omission_id = 5;
            end
            omidx = [omission_id,rew_id];
            input_IRI = table2array(trial_table(trial_table.rep==i,4));
            event_log.events = double(event_log.events);
            event_log.reward = double(event_log.reward);
            input_log = table2array(event_log);
            trial_table.DA(trial_table.rep==i) = calculateANCCRperformance2(input_log,input_IRI*Tratio,alpha_anccr,k,samplingperiod,w,threshold,minimumrate,beta,alpha_r,maximumjitter,nan,omidx,0);

        end

        %save results

        t_id = trial_table.t_id;%(1:height(trial_table))';
        DA = trial_table.DA;
        out_table = table(t_id,DA);
        out_table.p(:) = p;

        table_collect{t_cell}=out_table;
        if t_cell == 20
            tic;
            save_table = vertcat(table_collect{:});
            fname= sprintf('rep%danccr_results_p_%d_to_%d_saved.parquet',min(rep_range),min(save_table.p),max(save_table.p));
            robust_parquetwrite(fullfile(save_directory,fname),save_table);
            toc
            t_cell = 1;
            table_collect = cell(20,1);
        else
            t_cell = t_cell+1;
        end

    end

    %if t_cell is not 1, save the remaining tables
    if t_cell ~= 1
        save_table = vertcat(table_collect{:});
        fname= sprintf('rep%danccr_results_p_%d_to_%d_saved.parquet',min(rep_range),min(save_table.p),max(save_table.p));
        robust_parquetwrite(fullfile(save_directory,fname),save_table);
    end

%     sum_results = summarize_anccr_results(save_table,trial_table);
%     sum_fname = fname;%sprintf('anccr_summary_p_%d_to_%d_saved.parquet',min(save_table.p),max(save_table.p));
%     sum_directory = fullfile(save_directory,'summary');
%     if ~exist(sum_directory,'dir')
%         mkdir(sum_directory);
%     end
%     robust_parquetwrite(fullfile(sum_directory,sum_fname),sum_results);

end
end