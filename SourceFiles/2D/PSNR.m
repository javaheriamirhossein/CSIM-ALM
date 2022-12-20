function [ y ] = PSNR( img1,img2 )

img_vec1=double(img1(:));
img_vec2=double(img2(:));
% maxI=max(max(img_vec1));
maxI = 255;
MSE=mean((img_vec1-img_vec2).^2);

y=10*log10(maxI^2/MSE);

end

