function [s,u,rel_err,time_iter] = csim_alm( D,y,Opt )

%-----------------------------------------------------------------------------------------------------------
% CSIM-ALM algorithm to solve:
%       minimize_{x,s,z} CSIM(z) + alpha*||s||_1+gamma||z||^2,
%       s.t. {x=Ds, z=Mx-y}
%
%   where D is the dictionary matrix, y is the observed signal and s is
%   the sparse coefficient vector
%
%-----------------------------------------------------------------------------------------------------------
%
% INPUT
%   D           : mxK  dictionary matrix
%   y           : mx1 vector of observed values
%   Opt         : struct variable of options
%   Opt.maxIter : max iteration limit of the algorithm
%   Opt.mask    : sampling mask of the signal (M)
%   Opt.report  : if set to 1, the algorithm computes the time and relative error of reconstruction in each
%                 iteration
%   Opt.x0      : the  value of true signal
%   Opt.D0      : basis dictionary of the true signal representation,
%                 (x0 = D0s0)
%   Opt.mean_x  : mean shift value of the observed signals before missing
%   Opt.scale_x : scale value of the observed signals before missing

% OUTPUT
%   s           : Kx1 vector of sparse coefficients
%   u           : auxiliary variable (u is an estimae of true signal)
%   rel_err     : vector values of relative error in each iteration
%   time_iter   : vector values of time taken in each iteration
%
% USAGE EXAMPLES
%   [d] = CMN_ADM(D,y);
%   [x,rel_err,time_iter] = CMN_ADM(D,y,Opt);
%
%-------------------------------------------------------------------------------------------------------------

m = length(y);
u = zeros(m,1);
[~,K]= size(D);
Dt = D';

if ~isfield(Opt,'mask')
    Opt.mask = ones(m,1);
end
if ~isfield(Opt,'maxIter')
    Opt.maxIter = 100;
end

M = Opt.mask;   % M is the mask signal
M = M(:);
sr = 2*sum(M)/m;  % sampling rate

DtD = Dt*D;
Ds = D(M,:);
ys = y(M);
x0_est = Ds'*ys;


Niter = Opt.maxIter ;
rel_err = NaN(1,Niter);
time_iter = NaN(1,Niter);

% ---------------------------------------------------------------------
if Opt.report
    x0 = Opt.x0;
    D0 = Opt.D0;
    mean_x = Opt.mean_x ;
    scale_x = Opt.scale_x ;
    norm_x0 = norm(x0);
end

% --------------------- Algorithm parameters ---------------------------
sigma1 = 0.2*sr;
sigma2 = 2*sr;
ratio = 0.1;
gamma = 1;

sigma_tilde = sigma1/ (sigma2);
sigma_tilde_inv = 1/(1+sigma_tilde);
sigma1_inv = 1/sigma1;

L0 = norm(Ds,2)^2;
L = L0 ;
lambda = ratio*L0*norm(x0_est,inf) ;


% ---------------------- CSIM index parameters -------------------------
k2 = (m-1);
k1 = 0.86*k2;

theta1 = k2/(m-1);
theta2 = k1/m^2-k2/(m*(m-1));

theta1 = 2*theta1+sigma2+2*gamma;
theta2 = 2*theta2;
in_K_A = 1/(theta1) *( eye(m)- theta2/(theta1+m*theta2) *(ones(m)) );


% ------------------ Initialization of the variables --------------------
s = zeros(K,1) ;
z = -y;
mu1 = zeros(m,1);
mu2 = zeros(m,1);


sigma_tot =  sigma1_inv*sigma_tilde_inv;
sigma1_D = sigma1*D;
sigma_tot_mask = sigma_tot*M;
sigma2_mask = sigma2*M;
sigma1_p = 1.9*sigma1;

% -----------------------------------------------------------------------
t0 = tic;
iter = 0;
while iter<Niter
    
    b = sigma1_D*s-mu1 + sigma2_mask.*(z+y)+mu2.*M;
    u = sigma1_inv*b - sigma_tot_mask.*b ;
    
        
    temp = DtD*s - Dt*(u + sigma1_inv*mu1) ;    
    gk = s - (1/L)*temp ;
    thr = lambda/L*sigma1_inv;
    sp = Threshold(gk,thr) ;       
    s = sp ;
    
    iter = iter+1;
    
    % ---------------- Error calculation ---------------    
    if Opt.report
        time_iter(iter) = toc(t0);
        xhat = scale_x *D0*s + mean_x;
        rel_err(iter) = norm(x0 - xhat)/norm_x0;
    end
    % --------------------------------------------------
    
    
    c = sigma2_mask.*(u-y)- mu2;
    z = in_K_A*c;
    
    
    mu1 = mu1+ sigma1_p *(u-D*s);
    mu2 = mu2+ sigma2*z-(u-y).*sigma2_mask;
    
    %     if norm(u-D*s)<1e-6 && norm(z-u.*mask+y)<1e-6
    %         break;
    %     end
    
    
end

% u(mask) = ys;
toc(t0)

end


function shat = Threshold(s,thr)
    Supp=(abs(s)>=thr);
    shat = sign(s).*(abs(s)-thr).*Supp;
end
