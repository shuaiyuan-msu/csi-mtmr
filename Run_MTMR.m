%% Load test dataset
clear;clc;
load test_data.mat;

%% Format of test_data:  year, lakeid, lat, lon, reponses vars, predictors; 
data = [data(:,2),data(:,1), data(:,8:9),data(:,[4,5,7]),-data(:,6), data(:,11:end)];
header = [header(1,2),header(1,1), header(1,8:9),header(1,[4,5,7]),header(1,6), header(1,11:end)];
NumVar = 4; NumFeature = 48; 

%% Data Preprocessing

% Normalize data (standardize predictor)
tmp = data(:,5:end);
m = nanmean(tmp);
s = nanstd(tmp);
tmp = (tmp - repmat(m,size(tmp,1),1))./repmat(s,size(tmp,1),1);
data = [data(:,1:4),tmp];

% partition into training and test
trnrate = 2/3;vadrate = 1/2; ROUND = 1;
TrnIdx = cell(1,NumVar);TstIdx = cell(1,NumVar); ResMat = [];

ResMat = [];
for var = 1: NumVar
    data_var = [data(:,1:2), data(:,4+var),data(:,5+NumVar:end)];
    [TrnIdx{var},TstIdx{var},TrnNum{var},TstNum{var}] = GenerateTrnTstIdx(...
       data_var,trnrate,ROUND);%data = Year, lakeid, response var, predictor
    [Xtrn(var,:), Ytrn(var,:), Xtst(var,:), Ytst(var,:)] = SplitTrnTst4(...
        data_var, TrnIdx{var}, TstIdx{var}); % data = year, lake, res, predictor    
    Resvec = data_var(:,3);
    Resvec(TstIdx{var}) = NaN;
    ResMat = [ResMat,Resvec];
end
clear data_var 

% compute spatial autocorrelation(A1) and response var correlation(A2)
Latlon = unique(data(:,2:4),'rows');
A1 = computeRbfKernel(Latlon(:,2:3),sqrt(NumFeature/2)); 
A2 = corrcov(nancov(ResMat));% based on training only
clear tmp ResMat Resvec m s var;

%% Parameter Tuning 
% optimization options
opts.init = 0;      % 0 guess start point from data, 2 zero matrix
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-4;   % tolerance.
opts.maxIter = 500; % maximum iteration number of optimization.
opts.verbose = 0;   % print the iteration.

% Grid search for parameters
Lambda1 = [1e-4,1e-2];
Lambda2 = [1,5];
Lambda3 = [100,200];
Lambda = generateParamset(Lambda1,Lambda2,Lambda3);
tic;
[ best_param, perform_mat,best_idx] = TuneParam_MTMR2...
    ( 'MTMR_Lasso', opts, Lambda, 'eval_rmse', false,data,...
    NumFeature,TrnIdx,TstIdx,vadrate,A1,A2); % data: year,lakeid,latlon,response,predictor
runtime = toc;
%% Results
L1 = computeGraphLaplacian(A1);
L2 = computeGraphLaplacian(A2);
[W, funcVal] = MTMR_Lasso(Xtrn, Ytrn, L1, L2, best_param(1),best_param(2),best_param(3),opts);
rmse = zeros(NumVar,1);
rmseall = cell(NumVar,1);
r2 = cell(NumVar,1);
for v_ii = 1: NumVar
    [rmse(v_ii),rmseall{v_ii},r2{v_ii},r2all{v_ii},Y_pred{v_ii},Y_real{v_ii}] = eval_rmse2(Xtst(v_ii,:),Ytst(v_ii,:),W((v_ii-1)*NumFeature+1:v_ii*NumFeature,:));
end

fname = 'MTMR_result';
save(fname,'rmse','r2','rmseall','r2all','Y_pred','Y_real','best_param','W',...
    'funcVal','runtime','TrnIdx','TstIdx','opts','trnrate','vadrate','perform_mat',...
    'Lambda','NumVar');

