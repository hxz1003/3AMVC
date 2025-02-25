function [UU,A,Z,iter,obj] = algo_qp(X,Y,theta,beta,lambda,target_view)
% m      : the number of anchor. the size of Z is m*n.
% lambda : the hyper-parameter of regularization term.
% X      : n*di

%% initialize
maxIter = 50 ; % the number of iterations
numview = length(X);
numsample = size(Y,1);
k = length(unique(Y));
M = cell(numview,1);

for i = 1:numview
   A{i} = theta{i}';
   X{i} = mapstd(X{i}',0,1); % turn into d*n

   M{i} = (A{i}'*X{i}); 
   for ii=1:numsample
       idx = 1:size(theta{i},1);
       pp(idx,ii) = EProjSimplex_new(M{i} (idx,ii));           
    end
    Zi{i} = pp;
    clear pp;
end

flag = 1;
iter = 0;
%%
while flag
    iter = iter + 1;
    
    %% optimize A_i
    parfor ia = 1:numview
        if size(A{ia},1)<size(A{ia},2)
            A{ia} = X{ia}*Zi{ia}'*pinv(Zi{ia}*Zi{ia}');
        else
            C = X{ia}*Zi{ia}';      
            [U,~,V] = svd(C,'econ');
            A{ia} = U*V';
        end
    end
    
   %% optimize Z-i            

    for a=1:numview
        M{a} = (A{a}'*X{a})/(1+beta); 
        for ii=1:numsample
            idx = 1:size(theta{a},1);
            pp(idx,ii) = EProjSimplex_new(M{a} (idx,ii));           
        end
        Zi{a} = pp;
        clear pp;
    end

    %%
    term1 = 0;
    term2 = 0;
    for iv = 1:numview
        term1 = term1 + norm(X{iv} - A{iv} * Zi{iv},'fro')^2;
        term2 = term2 + norm(Zi{iv},'fro')^2;
    end
    
    obj(iter) = term1+beta*term2;
    
    if (iter>9) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-6 || iter>maxIter || obj(iter) < 1e-10)
        [Z,T] = aligned(Zi,lambda,target_view);
        [UU,~,~]=svd(Z','econ');
        flag = 0;
    end
end
         
         
    
