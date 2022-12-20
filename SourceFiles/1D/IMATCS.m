function [sol_IMATCS,rel_err,time_it] = IMATCS(y,A,T_Max,T_Min,Opt) %p is L tmax = 1, tmin = 0.001,

itermax = Opt.maxIter;
K = size(A,2);


time_it = NaN(1,itermax);
rel_err = NaN(1,itermax);
%--------------------------------------
if Opt.report
    x0 = Opt.x0;
    D0 = Opt.D0;
    mean_x = Opt.mean_x ;
    scale_x = Opt.max_x ;
    norm_x0 = norm(x0);    
end
%--------------------------------------


maxEIG_A = norm(A,2)^2;
lambda = 2/maxEIG_A;
alpha = log(T_Max/T_Min)/itermax;

t0 = tic;
x = pinv(A)*y;
for i = 1:itermax
    
    xp = x;    
    Trsh = T_Max*exp(-alpha *(i-1));
    temp = lambda*(A'*y)+ xp-lambda*(A'*A)*xp;    
    x = Threshold(temp,Trsh);
        
    % -------------------------------------
    if Opt.report
        xhat = scale_x *D0*x + mean_x;
        rel_err(i) = norm(x0 - xhat)/norm_x0;
        time_it(i) = toc(t0);
    end
    %--------------------------------------
end

toc(t0)
sol_IMATCS = x;

end

function x = Threshold(s,thr)

Supp=(abs(s)>=thr);
% shat = s.* Supp;
x = sign(s).*(abs(s)-thr).*Supp;
end
