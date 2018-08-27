function [ best_param, perform_mat,best_idx] = TuneParam_MTMR2...
    ( obj_func_str, obj_func_opts, param_range, eval_func_str, higher_better,data,NumFeature,TrnIdx,TstIdx,vadrate,A1,A2)
%% INPUT
% data: year,lakeid,latlon,response,predictor
%   obj_func_str:  1-parameter optimization algorithms
%   param_range:   the range of the parameter. array
%   eval_func_str: evaluation function:
%       signature [performance_measure] = eval_func(Y_test, X_test, W_learnt)
%   higher_better: if the performance is better given
%           higher measurement (e.g., Accuracy, AUC)
%% OUTPUT
%   best_param:  best parameter in the given parameter range
%   perform_mat: the average performance for every parameter in the
%                parameter range.
eval_func = str2func(eval_func_str);
obj_func  = str2func(obj_func_str);
NumVar = size(TrnIdx,2);
% ResMat = [];
for var = 1 : NumVar
    data_var = data(:,[1,2,4+var,5+NumVar:end]);% data:Year,lakeid,response var, 4 to end
    data_var(TstIdx{var},3) = NaN;
 [cv_Xtr(var,:), cv_Ytr(var,:), cv_Xte(var,:), cv_Yte(var,:),~,~,~,~,~] =...
     SplitTrnTst3(data_var, 1-vadrate);  % data:Year,lakeid,response var, 4 to end
%     Resvec = data(:,4+var);
% %     Resvec(teidx) = NaN;
%     Resvec(TstIdx{var}) = NaN;
%     ResMat = [ResMat,Resvec];
end

% Latlon = unique(data(:,2:4),'rows');
% A1 = computeRbfKernel(Latlon(:,2:3),sqrt(NumFeature/2)); 
% tmp = corrcov(nancov(ResMat));tmp(tmp<0)=0;% based on training only
% A2 = tmp;
% A2 = [1,0.68,0,0.37;0.68,1,0,0.30;0,0,1,0;.37,0.29,0,1];
L1 = computeGraphLaplacian(A1);
L2 = computeGraphLaplacian(A2);

perform_mat = zeros(size(param_range,1),1);% performance vector
fprintf('===== tune parameters =======');
parfor p_idx = 1: size(param_range,1)
    performance = zeros(length(NumVar),1);
    W = obj_func(cv_Xtr, cv_Ytr,L1,L2, param_range(p_idx,1),param_range(p_idx,2),param_range(p_idx,3), obj_func_opts);
    for v_ii = 1: NumVar
        performance(v_ii) = eval_func(cv_Xte(v_ii,:),cv_Yte(v_ii,:),W((v_ii-1)*NumFeature+1:v_ii*NumFeature,:));
    end
    perform_mat(p_idx) = mean(performance);
    fprintf('# %i is done\n',p_idx)
end

if(higher_better)
    [~,best_idx] = max(perform_mat);
    best_param = param_range(best_idx,:);
else
    [~,best_idx] = min(perform_mat(:));
    best_param = param_range(best_idx,:);
end
end

