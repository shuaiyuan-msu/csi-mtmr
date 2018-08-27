function [W, funcVal] = MTMR_Trace(X, Y, L1, L2, lambda1,lambda2,lambda3,opts)
%% Input:
% X{j,i} = n_{i,j} by d data matrix for task i variable j; i = 1 ...N, j =
% 1...M.
% Y{j,i} = n_{i,j} by 1 vector for task i variable j;
% L1 = N by N;
% L2 = M by M;
%% Intialization
% runtime = cell(5,3);
if nargin <8
    opts = [];
end
% Initialize options.
opts=init_opts(opts);
if ~isfield(opts, 'verbose')
    opts.verbose = 0;
end
task_num  = size(X,2);
var_num = size(X,1);
dimension = size(X{1,1}, 2);
funcVal = [];

% Initialize a starting point
% 2 = zero matrix
% 1 = user specified in opts.W0;
% 0 = X'Y;
tic;
if opts.init==2
    W0 = zeros(var_num*dimension,task_num);
elseif opts.init == 0
    W0 = intialW0(X,Y);
else
    if isfield(opts,'W0')
        W0=opts.W0;
        if (nnz(size(W0)-[var_num*dimension, task_num]))
            error('\n Check the input .W0');
        end
    else
        W0 = intialW0(X,Y);
    end
end
% runtime{1,1} = 'intialize W';runtime{1,2} = toc;
bFlag=0; % this flag tests whether the gradient step only changes a little
Wz= W0;
Wz_old = W0;

t = 1;
t_old = 0;

iter = 0;
gamma = 1;
gamma_inc = 2;

%% Main iteration
while iter < opts.maxIter
    alpha = (t_old - 1) /t;
    Ws = (1 + alpha) * Wz - alpha * Wz_old;
    
    % Compute function value and gradients of the search point
    tic;
    gWs  = gradVal_eval(Ws);
    
%     runtime{2,1} = 'compute gradient';runtime{2,3} = cat(1,runtime{2,3},toc);
    tic;
    Fs   = funVal_eval(Ws);
%     runtime{3,1} = 'compute function value';runtime{3,3} = cat(1,runtime{3,3},toc);
    
    while true
        tic;
        [Wzp,Wzp_tn] = proxmap_trace(Ws - gWs/gamma, lambda3 / gamma);
%         runtime{4,1} = 'linesearch-svt';runtime{4,3} = cat(1,runtime{4,3},toc);
        tic;
        Fzp = funVal_eval(Wzp);
        
        delta_Wzp = Wzp - Ws;
        r_sum = norm(delta_Wzp, 'fro')^2;
        %Fzp_gamma = Fs + trace(delta_Wzp' * gWs) + gamma/2 * norm(delta_Wzp, 'fro')^2;
        Fzp_gamma = Fs + sum(sum(delta_Wzp .* gWs)) + gamma/2 * norm(delta_Wzp, 'fro')^2;% eq(7)
%         runtime{5,1} = 'linesearch-other';runtime{5,3} = cat(1,runtime{5,3},toc);
        if (r_sum <=1e-20)
            bFlag=1; % this shows that, the gradient step makes little improvement
            break;
        end
        
        if (Fzp <= Fzp_gamma)
            break;
        else
            gamma = gamma * gamma_inc;
        end
        
    end
    
    
    Wz_old = Wz;
    Wz = Wzp;
    
    %funcVal = cat(1, funcVal, Fzp + lambda3 * sum( svd(Wzp, 0) ));
    funcVal = cat(1, funcVal, Fzp + lambda3 * Wzp_tn);
    
    if (bFlag)
        % fprintf('\n The program terminates as the gradient step changes the solution very small.');
        break;
    end
    
    % Check stop condition.
    switch(opts.tFlag)
        case 0 % change of absolute funcVal
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <= opts.tol)
                    break;
                end
            end
        case 1 % change of relative funcVal
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <=...
                        opts.tol* funcVal(end-1))
                    break;
                end
            end
        case 2
            if ( funcVal(end)<= opts.tol)
                break;
            end
        case 3
            if iter>=opts.maxIter
                break;
            end
    end
    
    if(opts.verbose)
    fprintf('Iteration %8i| function value %12.4f \n',iter,funcVal(end));
    end
    iter = iter + 1;
    t_old = t;
    t = 0.5 * (1 + (1+ 4 * t^2)^0.5);
end
W = Wzp;


% runtime{2,2} = sum(runtime{2,3});
% runtime{3,2} = sum(runtime{3,3});
% runtime{4,2} = sum(runtime{4,3});
% runtime{5,2} = sum(runtime{5,3});
% private functions
    function [grad_W] = gradVal_eval(W)
        %         if opts.pFlag
        grad_W1 = zeros(size(W));
        grad_W2 = zeros(size(W));
        grad_W3 = zeros(size(W));
        tmp = zeros(dimension,task_num);
%         gradient of f1
        for v_ii = 1: var_num
            for t_ii = 1:task_num
                tmp(:,t_ii) = 2*X{v_ii,t_ii}'*(X{v_ii,t_ii}*W((v_ii-1)*dimension+1:v_ii*dimension,t_ii) -Y{v_ii,t_ii});
            end
            grad_W1((v_ii-1)*dimension+1:v_ii*dimension,:) = tmp;
        end
        % gradient of f2
        for v_ii = 1: var_num
            grad_W2((v_ii-1)*dimension+1:v_ii*dimension,:) = ...
                2*lambda1*W((v_ii-1)*dimension+1:v_ii*dimension,:)*L1;
        end
        % gradient of f3
        for t_ii = 1: task_num
            Wmatrix = reshape(W(:,t_ii),dimension,var_num);
            WL = Wmatrix*L2;
            grad_W3(:,t_ii) = 2*lambda2*reshape(WL,var_num*dimension,1);
        end
        grad_W = grad_W1+grad_W2+grad_W3;
    end

    function [funcVal] = funVal_eval(W)
        funcVal = 0;
        %         if opts.pFlag
        %             parfor i = 1: task_num
        
        for t_ii = 1: task_num
            for v_ii = 1: var_num
                %f1
                funcVal = funcVal + norm (Y{v_ii,t_ii} - X{v_ii,t_ii} * W((v_ii-1)*dimension+1:v_ii*dimension,t_ii))^2;
            end
            % f3
            Wmatrix = reshape(W(:,t_ii),dimension,var_num);
            funcVal = funcVal+lambda2*sum(sum(Wmatrix*L2.*Wmatrix));
        end
        %f2
        for v_ii = 1: var_num
            funcVal = funcVal+lambda1*sum(sum(W((v_ii-1)*dimension+1:v_ii*dimension,:)*L1.*W((v_ii-1)*dimension+1:v_ii*dimension,:)));
        end
    end

    function [W0_prep] = intialW0(X,Y)
        % precomputation.
        XY = cell(var_num,task_num);
        W0_prep =zeros(var_num*dimension,task_num);
        for t_idx = 1: task_num
            for v_idx = 1:var_num
                XY{v_idx,t_idx} = X{v_idx,t_idx}'*Y{v_idx,t_idx};% d by 1 vector
                W0_prep((v_idx-1)*dimension+1:v_idx*dimension,t_idx) =  XY{v_idx,t_idx};
            end
        end
    end
end

function [L_hat,L_tn] = proxmap_trace(L, alpha)
[d1 d2] = size(L);
if (d1 > d2)
    [U S V] = svd(L, 0);
    thresholded_value = diag(S) - alpha;
    diag_S = thresholded_value .* ( thresholded_value > 0 );
    L_hat = U * diag(diag_S) * V';
    L_tn = sum(diag_S);
else
    new_L = L';
    [U S V] = svd(new_L, 0);
    thresholded_value = diag(S) - alpha;
    diag_S = thresholded_value .* ( thresholded_value > 0 );
    L_hat = U * diag(diag_S) * V';
    L_hat = L_hat';
    L_tn = sum(diag_S);
end
end

