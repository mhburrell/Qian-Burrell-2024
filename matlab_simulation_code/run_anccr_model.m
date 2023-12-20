function run_anccr_model(array_index)

trial_table = parquetread('simulated_trials_anccr.parquet');
trial_table = trial_table(trial_table.rep<6,:);
param_table = parquetread('anccr_param_table.parquet');
param_table = sortrows(param_table,'p');

n_params = height(param_table);
%split 1:n_params into 100 chunks, choose array_index chunk
param_chunk = arrayfun(@(x,y) x:y,1:ceil(n_params/100):n_params,ceil(n_params/100):ceil(n_params/100):n_params,'UniformOutput',false);
param_chunk = param_chunk{array_index};


save_directory = './anccr_results';


%common parameters
samplingperiod = 0.2;
minimumrate = 0.001;
maximumjitter = 0.1;
beta = [1 0 0 0 0];
t_cell = 1;
table_collect = cell(10,1);

for p = param_chunk
    disp(num2str(p));
    %check if file exists
    file_list = dir(fullfile(save_directory,'*.parquet'));
    if ~isempty(file_list)
        file_list = {file_list.name};
        file_list = cellfun(@(x) strsplit(x,'_'),file_list,'UniformOutput',false);
        %min p is 4th element, max p is 6th element
        min_p = cellfun(@(x) str2double(x{4}),file_list);
        max_p = cellfun(@(x) str2double(x{6}),file_list);
        %create sequence from min_p to max_p, concatenate into one vector
        p_list = arrayfun(@(x,y) x:y,min_p,max_p,'UniformOutput',false);
        p_list = horzcat(p_list{:});
        %check if p is in p_list
        if ismember(p,p_list)
            continue
        end


    end

    alpha_anccr = param_table.alpha_anccr(p);
    k = param_table.k(p);
    alpha_r = param_table.alpha_anccr(p);
    w = param_table.w(p);
    threshold = param_table.theta(p);
    Tratio = param_table.Tratio(p);

    trial_table.DA = zeros(size(trial_table.events));
    for i = 1:5
        for j = 1:3
            event_log = trial_table(trial_table.rep==i&trial_table.testgroup==j,:);
            switch j
                case {1,2}
                    event_log.events(event_log.events==5)=4;
                    omidx = [4,1];
                case {3}
                    omidx = [5,1];
            end

            IRI = diff(event_log.t(event_log.r==1));
            lIRI = length(IRI);
            lEvent_log = length(event_log.events);
            input_IRI = [repmat(mean(IRI(1:floor(lIRI/2))),floor(lEvent_log/2),1); repmat(mean(IRI(floor(lIRI/2)+1:end)),lEvent_log-floor(lEvent_log/2),1)] ;
            input_log = table2array(event_log);
            input_log = input_log(:,1:3);


            trial_table.DA(trial_table.rep==i&trial_table.testgroup==j) = calculateANCCRperformance(input_log,input_IRI*Tratio,alpha_anccr,k,samplingperiod,w,threshold,minimumrate,beta,alpha_r,maximumjitter,nan,omidx,0);

        end
    end

    t_id = (1:height(trial_table))';
    DA = trial_table.DA;
    out_table = table(t_id,DA);
    out_table = addTableVariable(out_table,'p',p);

    table_collect{t_cell}=out_table;
    if t_cell == 10
        save_table = vertcat(table_collect{:});
        fname= sprintf('anccr_results_p_%d_to_%d_saved.parquet',min(save_table.p),max(save_table.p));
        parquetwrite(fullfile(save_directory,fname),save_table);
        t_cell = 1;
        table_collect = cell(10,1);
    else 
        t_cell = t_cell+1;
    end

end

%if t_cell is not 1, save the remaining tables
if t_cell ~= 1
    save_table = vertcat(table_collect{:});
    fname= sprintf('anccr_results_p_%d_to_%d_saved.parquet',min(save_table.p),max(save_table.p));
    parquetwrite(fullfile(save_directory,fname),save_table);
end

end