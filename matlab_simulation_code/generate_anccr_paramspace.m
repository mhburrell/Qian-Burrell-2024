%generate parameter space

Tratio = [0.2,0.6,1,1.2,1.4, 2, 10];
alpha_anccr = [0.01,0.02,0.05,0.1,0.2];
k = [nan, 0.01,0.05,0.2,0.6];
w = [0 0.25 0.4 0.5 0.75 1];
theta = [0.01,0.1,0.3,0.5,0.7];
alpha_r = [0.05,0.1,0.2,0.3];

[Tratio,alpha_anccr,k,w,theta,alpha_r] = ndgrid(Tratio,alpha_anccr,k,w,theta,alpha_r);
vars = {Tratio, alpha_anccr, k, w, theta, alpha_r};
vars = cellfun(@(x) x(:), vars, 'UniformOutput', false);
[Tratio, alpha_anccr, k, w, theta, alpha_r] = vars{:};

param_table = table(Tratio, alpha_anccr, k, w, theta, alpha_r);
param_table.p_id = (1:height(param_table))';

parquetwrite('anccr_param_table.parquet',param_table);