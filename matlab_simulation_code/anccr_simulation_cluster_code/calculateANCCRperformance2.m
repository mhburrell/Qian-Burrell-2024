function [DA,ANCCR,PRC,SRC,NC,Rs,Delta,Mij,Mi,Eij,Ei] =...
    calculateANCCRperformance2(eventlog, T, alpha, k,samplinginterval,w,threshold,...
    minimumrate,beta,alpha_r,maximumjitter,optolog,omidx,exact_mean_or_not,nevent_for_edge)

%original code from https://github.com/namboodirilab/ANCCR
%modifications to improve code runtime (lines 49-56, 217-223)
%allow k = 1/IRI (lines 80-82)
if alpha_r>1
    alpha_r = 1;
end
if nargin<=11 | isnan(optolog)
    optolog = zeros(size(eventlog,1),2);
end
if nargin<=12 | isnan(omidx)
    % First entry is omission index, second entry is corresponding reward index
    omidx = [nan,nan];
end
if nargin<=13
    % if exact_mean_or_not=1, calculate exact mean for Mij instead of using
    % alpha
    exact_mean_or_not = 0;
end

if nargin<=14
    % if nevent_for_edge>0, use averaged NC for last nevent to calculate
    % edge
    nevent_for_edge = 0;
end

% omtrue: whether the omission state will be used or not in the calculaton of ANCCR
%omtrue = true(size(omidx,1),1);
omtrue = false(size(omidx,1),1);
uniquetime = unique(eventlog(:,2));

%% if more than one event happens at the same time, assume random perceptual delay between them
for jt = 1:length(uniquetime)
    if sum(eventlog(:,2)==uniquetime(jt))==1
        continue;
    end
    idx = find(eventlog(:,2)==uniquetime(jt));
    eventlog(idx(2:end),2) = eventlog(idx(2:end),2)+randn(length(idx)-1,1)*maximumjitter;
end
eventlog = sortrows(eventlog,2);
ntime = size(eventlog,1);

%%
nstimuli = max(eventlog(:,1));%length(unique(eventlog(:,1)));
samplingtime = 0:samplinginterval:eventlog(end,2);

%modified code:
[~,~,bin] = histcounts(samplingtime,eventlog(:,2));
ss = cell(length(eventlog(:,2)),1);
for ii = 1:length(bin)
    if bin(ii)
        ss{bin(ii)} = [ss{bin(ii)},ii];
    end
end


% if T is a vector, use T(jt) for the calculation at time jt. otherwise,
% use fixed T
if length(T)==1
    T = repmat(T,size(eventlog,1),1);
end
gamma = exp(-1./T);

% Initialize model values
[Eij,Ei,Mi,Delta] = deal(zeros(nstimuli,ntime));
[Mij,PRC,SRC,NC,ANCCR,Rs] = deal(zeros(nstimuli,nstimuli,ntime));
R = zeros(nstimuli,nstimuli);
numevents = zeros(nstimuli,1);
DA = zeros(ntime,1);

beta = beta(1:max(eventlog(:,1)));%beta(unique(eventlog(:,1)));
Imct = beta(:)>threshold;
nextt = 1;
numsampling = 0;
%tic;
[Delta, Eij, Mij, PRC, SRC, NC, ANCCR, Mi, DA, Rs, Ei] = anccr_loop_newk_mex(ntime, k, T, eventlog, omidx, omtrue, Delta, Eij, Mij, PRC, SRC, NC, numevents, exact_mean_or_not, alpha, gamma, ANCCR, nstimuli, Imct, Mi, minimumrate, R, w, nevent_for_edge, threshold, optolog, DA, beta, Rs, alpha_r, ss, samplingtime, Ei, samplinginterval, nextt, numsampling);
%toc;
end