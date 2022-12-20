function [ z ] = CSIM( x,y )
x= x(:);
y= y(:);

mu1=mean(x);
mu2=mean(y);
cov_mat=cov(x,y);

h1 = 2*mu1*mu2;
h2 = mu1^2+mu2^2;
h3 = 2*cov_mat(1,2);
h4 = cov_mat(1,1)+cov_mat(2,2);


k1=1e-1;
k2=4*k1;
% rho = k2/k1;
z= (k1*(h2-h1)+k2*(h4-h3));

end

