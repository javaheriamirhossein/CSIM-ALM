function [img_rec,PSNR_vec,time_iter] = csim_alm_inpaint_2D( img,mask,Opt )
%-----------------------------------------------------------------------------------------------------------
% 2D CSIM-ALM algorithm for recovery of image missing samples 
%-----------------------------------------------------------------------------------------------------------
%
% INPUT
%   img         : n1xn2 observed image (with missing samples)
%   mask        : n1xn2 binary sampling mask
%   Opt         : struct variable of options 
%   Opt.maxIter : max iteration limit of the algorithm
%   Opt.report  : if set to 1, the algorithm computes the time and PSNR of the reconstructed image in each
%                 iteration
%   Opt.img_org : n1xn2 original image signal
%   Opt.DicType : Applied 2D sparsifying dictionary (transform)

% OUTPUT
%   img_rec     : n1xn2 reconstructed image
%   PSNR_vec    : vector values of PSNR in each iteration
%   time_iter   : vector values of time taken after each iteration
%
% USAGE EXAMPLES
%   [img_rec] = csim_alm_inpaint_2D(img,mask);
%   [img_rec,PSNR_vec,time_iter] = csim_alm_inpaint_2D(img,mask,Opt);
%
%-------------------------------------------------------------------------------------------------------------

[n1 , n2] = size(img);
M = n1*n2;
img_rec = zeros(n1,n2);


if ~isfield(Opt,'maxIter')
    Opt.maxIter = 50;
end
if ~isfield(Opt,'report')
    Opt.report = 0;
end
if ~isfield(Opt,'DicType')
    Opt.DicType = 'DCT';
end


Niter = Opt.maxIter ;

%-------------------- Error report -------------------
if Opt.report
    img_org = Opt.img_org;
    PSNR_vec = zeros(1,Niter);
    time_iter = zeros(1,Niter);
end


%------------ 2D sparsifying transform ---------------

DicType = Opt.DicType;
switch DicType
    case 'DCT'        
        D = @(x) idct2(x);
        Dt = @(x) dct2(x);
    case 'FFT'
        D = @(x) ifft2(x);
        Dt = @(x) fft2(x);
end

%---------------- Algorithm parameters ---------------
M_s = sum(sum(mask));
sr = M_s/M;

sigma1 = 2*sr;
sigma2 = 20*sr;
ratio = 0.1;
gamma = 1;
eta = 0.9;

sigma_tilde = sigma1/ (sigma2);
sigma_tilde_inv = 1/(1+sigma_tilde);
sigma1_inv = 1/sigma1;
sigma_tot =  sigma1_inv*sigma_tilde_inv;
sigma_tot_mask = sigma_tot*mask;
sigma2_mask = sigma2*mask;
sigma1p = 1.9*sigma1;

L0 = 1;
lambda_min = 1e-6;
L = L0 ;
beta = 1.1;

%--------------- CSIM parameters ----------------------
k2 = (M-1);
k1 = 0.86*k2;

theta1 = k2/(M-1);
theta2 = k1/M^2-k2/(M*(M-1));

theta1 = 2*theta1+sigma2+2*gamma;
theta2 = 2*theta2;
in_K_A = @(x)  1/(theta1) *(x- theta2/(theta1+M*theta2)*sum(sum(x))*ones(n1,n2) );

%------------------- Initialization --------------------
mu1=  zeros(n1,n2);
mu2=  zeros(n1,n2);
Z =  zeros(n1,n2);
coeff = zeros(n1,n2) ;


% mag0 = Dt(my_mov_avrg(img,mask));
% img =Interpolation_Initial(img,~mask);
mag = Dt(img) ;
lambda = ratio*L0*max(max(abs(mag))) ;


% ------------------------------------------------------
t0 = tic;
for iter=1:Niter
    
    
    parts = D(coeff);    
    B = sigma1*parts-mu1 + sigma2_mask.*(Z+img)+mu2.*mask;
    img_rec = sigma1_inv*B - sigma_tot_mask.*B ;
    
        
    resid = my_mov_avrg( parts - (img_rec + sigma1_inv*mu1),mask)  ;
    
    stop_backtrack = 0 ;
    count=0;
    while ~stop_backtrack && count<3        
        count = count+1;
        coeff_est = Dt(parts - (1/L)*resid) ;
        thr = lambda/L*sigma1_inv;
        coeff_th = Threshold(coeff_est,thr) ;
                
        temp1 = norm(D(mag-coeff_th),'fro');
        temp2 = sqrt(L)*norm(mag-coeff_th,'fro');
                
        if temp1 <= temp2
            stop_backtrack = 1 ;
        else
            L = L*beta;
        end        
    end
    
    lambda = max(eta*lambda,lambda_min) ;
    coeff = coeff_th ;
    
    
    % ---------------- Error calculation --------------- 
    if Opt.report
        time_iter(iter) = toc(t0);
        PSNR_vec(iter) = PSNR(img_org,img_rec);
    end
    % --------------------------------------------------
    
    c = sigma2_mask.*(img_rec-img)- mu2;
    Z = in_K_A(c);
    
    
    mu1 = mu1+ sigma1p *(img_rec-D(coeff));
    mu2 = mu2+ sigma2*Z-(img_rec-img).*sigma2_mask;
    
end

img_rec(img_rec<0) = 0;
img_rec(mask) = img(mask);

toc(t0)
end


function shat = Threshold(s,thr)
    [n1,n2] = size(s);
    s = s(:);
    Supp=(abs(s)>=thr);
    shat = sign(s).*(abs(s)-thr).*Supp;
    shat = reshape(shat,n1,n2);
end


