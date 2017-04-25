function [B S stat] = sparse_coding(AtA,alpha, X_total,Y, num_bases, beta, sparsity_func, epsilon, num_iters, batch_size, fname_save, pars, Binit, resample_size)


if exist('resample_size', 'var') && resample_size
    assert(size(X_total,2) > resample_size);
    X = X_total(:, randsample(size(X_total,2), resample_size));
else
    X = X_total;
end

pars.patch_size = size(X,1);
pars.num_patches = size(X,2);
pars.num_bases = num_bases;
pars.num_trials = num_iters;

if exist('batch_size', 'var') && ~isempty(batch_size)
    pars.batch_size = batch_size;
else
    pars.batch_size = size(X,2)/10;
end

pars.sparsity_func = sparsity_func;
pars.beta = beta;
pars.epsilon = epsilon;

pars.noise_var = 1;
pars.sigma = 1;
pars.VAR_basis = 1;

if ~isfield(pars,'display_images')	pars.display_images = false; end; %true;
if ~isfield(pars,'display_every')	pars.display_every = 1;	end;
if ~isfield(pars,'save_every')	pars.save_every = 1;	end;
if ~isfield(pars,'save_basis_timestamps')	pars.save_basis_timestamps = true;	end;

if exist('fname_save', 'var') && ~isempty(fname_save)
    pars.filename = fname_save;
else
    pars.filename = sprintf('../results/sc_b%d_%s', num_bases, datestr(now, 30));
end;

% Sparsity parameters
if ~isfield(pars,'tol');                 pars.tol = 0.005; end;

% L1 sparsity function
if strcmp(pars.sparsity_func,'epsL1')
    pars.epsilon = epsilon;
    pars.reuse_coeff = false;
else
    pars.epsilon = [];
    pars.reuse_coeff = true;
end;

%pars

% set path

% initialize basis
if ~exist('Binit') || isempty(Binit)
    B = rand(pars.patch_size,pars.num_bases)-0.5;
    B = B - repmat(mean(B,1), size(B,1),1);
%    B = B*diag(1./sqrt(sum(B.*B)));
else
    disp('Using Binit...');
    B = Binit;
end;
[L M]=size(B);
winsize=sqrt(L);

% initialize display
if pars.display_images
    figure(1), display_network_nonsquare2(B);
    % figure(1), colormap(gray);
end

S_all = zeros(M, size(Y,2));

% initialize t only if it does not exist
if ~exist('t')
    t=0;
    
    % statistics variable
    stat= [];
    stat.fobj_avg = [];
    stat.fresidue_avg = [];
    stat.fsparsity_avg = [];
    stat.var_tot = [];
    stat.svar_tot = [];
    stat.elapsed_time=0;
else
    % make sure that everything is continuous
    t= length(stat.fobj_avg)-1;
end
% for i = 1:size(X_total,2)
%     for j = 1:size(Y,2)
%         temp = [double(X_total(:,i)) double(Y(:,j))];
%         dst(i,j) = 1-pdist(temp','cosine');
%     end
% end
dst = X'*Y;
% optimization loop

while t < pars.num_trials
    t=t+1;
    start_time= cputime;
    
    stat.fobj_total=0;
    stat.fresidue_total=0;
    stat.fsparsity_total=0;
    stat.var_tot=0;
    stat.svar_tot=0;
    
    if exist('resample_size', 'var') && resample_size
        fprintf('resample X (%d out of %d)...\n', resample_size, size(X_total,2));
        X = X_total(:, randsample(size(X_total,2), resample_size));
    end
    
    % Take a random permutation of the samples
    indperm = randperm(size(X,2));
    
    for batch=1:(size(X,2)/pars.batch_size),
        % Show progress in epoch
        if 1, end%fprintf('.'); end
        if (mod(batch,20)==0) end %fprintf('\n'); end
        
        % This is data to use for this step
        batch_idx = indperm((1:pars.batch_size)+pars.batch_size*(batch-1));
        Xb = X(:,batch_idx);
        %%
        
        %%
        
        % learn coefficients (conjugate gradient)
        if t==1 || ~pars.reuse_coeff
            if strcmp(pars.sparsity_func, 'L1') || strcmp(pars.sparsity_func, 'LARS') || strcmp(pars.sparsity_func, 'FS')
                S= l1ls_featuresign(AtA,dst, alpha, X_total, Y, pars.beta/pars.sigma*pars.noise_var);
            else
                S= cgf_fitS_sc2(B, Xb(:,1), pars.sparsity_func, pars.noise_var, pars.beta, pars.epsilon, pars.sigma, pars.tol, false, false, false);
            end
            S(find(isnan(S)))=0;
            S_all= S;
        else
            if strcmp(pars.sparsity_func, 'L1') || strcmp(pars.sparsity_func, 'LARS') || strcmp(pars.sparsity_func, 'FS')
                tic
                S= l1ls_featuresign(AtA,dst, alpha, X_total, Y, pars.beta/pars.sigma*pars.noise_var, S_all);
                %FS_time = toc
            else
                S= cgf_fitS_sc2(B, Xb, pars.sparsity_func, pars.noise_var, pars.beta, pars.epsilon, pars.sigma, pars.tol, false, false, false, S_all);
            end
            S(find(isnan(S)))=0;
            S_all= S;
        end
        
        if strcmp(pars.sparsity_func, 'L1') || strcmp(pars.sparsity_func, 'LARS') || strcmp(pars.sparsity_func, 'FS')
            sparsity_S = sum(S(:)~=0)/length(S(:));
            %fprintf('sparsity_S = %g\n', sparsity_S);
        end
    end
end

return

%%

function retval = assert(expr)
retval = true;
if ~expr
    error('Assertion failed');
    retval = false;
end
return
