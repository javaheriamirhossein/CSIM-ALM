function [s,rel_err,time_iter] = IMAT(D,y,Opt)

if ~isfield(Opt,'alpha')
    Opt.alpha = 0.2;
end
if ~isfield(Opt,'beta')
    Opt.beta = 50;
end
if ~isfield(Opt,'lambda')
    Opt.lambda = 0.7;
end

[~,K] = size(D);
s = zeros(K,1);
Dt = conj(D');
Niter= Opt.maxIter;

rel_err = NaN(1,Niter);
time_iter = NaN(1,Niter);

if Opt.report
    x0 = Opt.x0;
    D0 = Opt.D0;
    mean_x = Opt.mean_x ;
    scale_x = Opt.scale_x ;
    norm_x0 = norm(x0);
end

alpha = Opt.alpha;
lambda = Opt.lambda;
beta = Opt.beta;
Thresh_min = 1e-3;


t0 = tic;
for j = 1:Niter
    Thresh = max(beta*exp(-alpha*(j-1)),Thresh_min);    
    s = s+lambda*Dt*(y-D*s);
    s = Threshold(s,Thresh);
    if Opt.report
        time_iter(j) = toc(t0);
        xhat = scale_x *D0*s + mean_x;
        rel_err(j) = norm(x0 - xhat)/norm_x0;
    end
end

end



function x = Threshold(s,thr)
    x = s;
    x(abs(x)<thr) = 0;
end

