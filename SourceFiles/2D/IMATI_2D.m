% img_path='Lena.bmp';
% itermax=100;
% eps=1e-4;
% lambda=1.8;
function [img_rec]=IMATI_2D(img,Mask,itermax,lambda,eps)
% Iterative Method with Adaptive Thresholding and Interpolation (IMATI)for random sampling recovery
% Inputs
% img_path              the path of the input image.
% itermax               the maximum number of iterations.
% srate                 the sampling rate.
% eps                   the maximum residual error ratio, used for stopping criterion
% lambda                the relaxation parameter where 0<lambda<2 .
% outputs
% img_rec               the recovered image using IMATI.

img_samp=Mask.*img;
% adjust the thresholding parameters
% In this part, the thresholding parameters are adjusted automatically according to the sampled image.
Y=mov_avrg(img_samp,Mask);                                    % Y is a rough recovery of the image
% mov_avrg is the function of a simple moving average interpolator.
tmpdct=dct2(Y);
T0=0.5*(max(max(abs(tmpdct)))+0.1);                      % initial threshold
alpha=abs(mean(mean(diff(sort(tmpdct,'descend')))));   % exponential threshold reduction factor
% thresholding operator
itr=1:itermax;
T(itr)=T0*exp(-alpha*(itr-1));
% initialization
X_old=0;
[as1,as2]=size(img);
X=zeros(as1,as2);
% IMATI recovery algorithm

for itr=1:itermax
    Xk=lambda*Y+X-lambda*mov_avrg(X,Mask); % iterative updation
    X_DCT=dct2(Xk);                        % transferring into sparsity domain
    X_Th=X_DCT.*(abs(X_DCT)>T(itr));       % thresholding in the sparsity domain
    X=idct2(X_Th);                         % transferring back to the image domain
    if norm(X_old-X)/norm(Y)<eps           % checking the stopping criterion
        % display the number of iterations
        %       disp(['Num of Itr for Sampling Rate(',num2str(srate),') : ',num2str(itr)]);
        break;
    end
    X_old=X;
end
img_rec=X;                                 % the recovered image using IMATI



