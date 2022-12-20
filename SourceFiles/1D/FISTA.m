function [shat,rel_err,time_iter] =FISTA(D,y,Opt)

% b - m x 1 vector of observations/data (required input)
% A - m x n measurement matrix (required input)
%
% tol - tolerance for stopping criterion.
%     - DEFAULT 1e-7 if omitted or -1.
% maxIter - maxilambdam number of iterations
%         - DEFAULT 10000, if omitted or -1.
% lineSearchFlag - 1 if line search is to be done every iteration
%                - DEFAULT 0, if omitted or -1.
% continuationFlag - 1 if a continuation is to be done on the parameter lambda
%                  - DEFAULT 1, if omitted or -1.
% eta - line search parameter, should be in (0,1)
%     - ignored if lineSearchFlag is 0.
%     - DEFAULT 0.9, if omitted or -1.
% lambda - relaxation parameter
%    - ignored if continuationFlag is 1.
%    - DEFAULT 1e-3, if omitted or -1.
% outputFileName - Details of each iteration are dumped here, if provided.
%
% x_hat - estimate of coeeficient vector


% Initializing optimization variables
if ~isfield(Opt,'D_pinv')
    Opt.D_pinv = pinv(D);
end

D_pinv = Opt.D_pinv;
maxIter = Opt.maxIter ;

mu = 1 ;
mu_old = 1 ;
L0 = 1 ;
DtD = D'*D ;
nIter = 0 ;
c = D'*y ;
lambda0 = 0.2*L0*norm(c,inf) ;
eta = 0.9 ;
lambda_min = 1e-6;
s = D_pinv*y ;
lambda = lambda0 ;
L = L0 ;
beta = 1.5;

% ---------------------------------------
if Opt.errorshow
    x0 = Opt.x0;
    D0 = Opt.D0;
    mean_x = Opt.mean_x ;
    scale_x = Opt.max_x ;
    norm_x0 = norm(x0);
    rel_err = zeros(1,maxIter);
    time_iter = zeros(1,maxIter);

end
% ---------------------------------------

s_old = s;
t0=tic;

while (nIter < maxIter)
    nIter = nIter + 1 ;
    
    s = s + ((mu_old-1)/mu)*(s-s_old) ;
    
    stop_backtrack = 0 ;
    
    temp = DtD*s - c ; % gradient of f at y
    
    count = 0;
    while ~stop_backtrack && count<3
        
        gk = s - (1/L)*temp ;
        thr = lambda/L;
        %         thr = sqrt(2*lambda/L);
        sp = Threshold(gk,thr) ;
        
        temp1 = 0.5*norm(y-D*sp)^2 ;
        temp2 = 0.5*norm(y-D*s)^2 + (sp-s)'*temp + (L/2)*norm(sp-s)^2 ;
        
        if temp1 <= temp2
            stop_backtrack = 1 ;
        else
            L = L*beta ;
        end
        count = count+1;
    end
    
    lambda = max(eta*lambda,lambda_min) ;
    mu_old = mu ;
    mu = 0.5*(1+sqrt(1+4*mu*mu)) ;
    s_old = s ;
    s = sp ;
    
    if Opt.errorshow
        time_iter(nIter) = toc(t0);
        xhat = scale_x *D0*s + mean_x;
        rel_err(nIter) = norm(x0 - xhat)/norm_x0;   
    end
end

shat = s ;
toc(t0)
end


function [shat,Supp] = Threshold(s,thr)
Supp=(abs(s)>=thr);
shat = sign(s).*(abs(s)-thr).*Supp;
end