% img_path='Lena.bmp';
% itermax=100;
% eps=1e-4;
% lambda=1.8;
function [img_rec]=IMAT_2D(img,Mask,itermax,lambda,eps)
% Iterative Method with Adaptive Thresholding (IMAT)for random sampling recovery
%% Inputs
% itermax               the maximum number of iterations.
% eps                   the maximum residual error ratio, used for stopping criterion
% lambda                the relaxation parameter where 0<lambda<2 .
%% outputs
% img_rec                 the recovered image using IMAT.
%% Description
% This code implements the IMAT random sampling recovery algorithm. 
% An image is random sampled with the aid of a binary random mask. The
% sparsity of the image in DCT domain is exploited to recover the missing
% samples.An interpolating operator is used at the first iteration to
% adjust the algorithm.
% The IMAT algorithm has been presented in the following papers:
% Farokh Marvasti, et al., "Sparse signal processing using iterative method with adaptive thresholding (IMAT)," 19th International Conference on Telecommunications (ICT 2012), pp. 1-6, April 2012.
% Farokh Marvasti,et al., "A Unified Approach to Sparse Signal Processing," EURASIP Journal on Advances in Signal Processing, 2012.
% M.Azghani and F.Marvasti, "sparse Signal Processing," chapter 8 in the book entitled New Perspectives on Approximation and Sampling Theory, Springer 2014.
% written by: Masoumeh Azghani (azghani@ee.sharif.edu)
% supervisor: Farokh Marvasti
% Version: 1.1
% Last modified: 13 December 2014.
% Affiliation:
% Advanced Communications Research Institute (ACRI)
% Electrical Engineering Department, Sharif University of Technology
% Tehran, Iran
% For any problems, contact me at azghani@ee.sharif.edu
%% PSNR function definition
%% import the image
                        % the input image
[as1,as2]=size(img);
%% sample data
img_samp=Mask.*img;                                    % randomly sampled image
%% adjust the thresholding parameters
% In this part, the thresholding parameters are adjusted automatically according to the sampled image.
Y=mov_avrg(img,Mask);                                  % Y is a rough recovery of the image
% mov_avrg is the function of a simple moving average interpolator.
tmpdct=dct2(Y);
T0=0.5*abs(max(max(tmpdct))+0.1);                      % initial threshold
alpha=abs(mean(mean(diff(sort(tmpdct,'descend'))))); 
% alpha = 0.5;% exponential threshold reduction factor
%% thresholding operator
itr=1:itermax;
T(itr)=T0*exp(-alpha*(itr-1));
%% initialization
X_old=0;
X=zeros(as1,as2);
%% IMAT random sampling recovery algorithm
for itr=1:itermax
    Xk=lambda*img_samp+X-lambda*(X.*Mask);   % iterative updation
    X_DCT=dct2(Xk);                        % transferring into sparsity domain
    X_Th=X_DCT.*(abs(X_DCT)>T(itr));       % thresholding in the sparsity domain
    X=idct2(X_Th);                         % transferring back to the image domain
    if norm(X_old-X)/norm(Y)<eps           % checking the stopping criterion
        %% display the number of iterations
        %       disp(['Num of Itr for Sampling Rate(',num2str(srate),') : ',num2str(itr)]);
        break;
    end
    X_old=X;
end
img_rec=X;                                 % the recovered image using IMAT

