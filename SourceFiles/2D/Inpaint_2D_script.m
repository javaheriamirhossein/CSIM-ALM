clear
close all

ImgName = {'lena256.png'};
MaskName = {'Mask_0_3_256.png','Mask_0_5_256.png'};  % Sampling masks for sr = 0.3, 0.5
% The user may add more images and mask

nI = length(ImgName);
nM = length(MaskName);


% RecImages = cell(nI,nM,3);
Times = zeros(nI,nM,3);
PSNRs = zeros(nI,nM,3);
SSIMs = zeros(nI,nM,3);
CSIMs = zeros(nI,nM,3); 

for i = 1:nI
    
    
    img = imread(ImgName{i});
    [N,M,dim]=size(img);
    if dim>1
        img_Ycbcr = rgb2ycbcr(img);
        img = img_Ycbcr(:,:,1);
    end
    img = double(img);
    
    for j = 1:nM
        
        mask = imread(MaskName{j});
        if size(mask,3)>1
            mask = mask(:,:,1);
        end
        img_masked = img.*mask;
        
        % -------------- IMDAT-2D --------------------------------
        tic;
        [Img_rec] = IMAT_2D(img,mask,100,1.8,1e-4);
        figure;
        imshow(uint8(Img_rec));
        Times(i,j,1) = toc;
        % RecImages{i,j,1} = Img_rec(1:N,1:M,:);
        PSNRs(i,j,1) = PSNR(img,Img_rec);
        SSIMs(i,j,1) = ssim_index(img,Img_rec);
        CSIMs(i,j,1) = CSIM(img,Img_rec);
        imwrite(uint8(Img_rec),['Rec_Img_IMAT_Im',num2str(i),'_sr_',num2str(j),'.png']);
        
        
        % -------------- IMATI-2D --------------------------------
        tic
        [Img_rec] = IMATI_2D(img,mask,100,1.8,1e-4);
        figure;
        imshow(uint8(Img_rec));
        Times(i,j,2) = toc;
        % RecImages{i,j,2} = Img_rec(1:N,1:M,:);
        PSNRs(i,j,2) = PSNR(img,Img_rec);
        SSIMs(i,j,2) = ssim_index(img,Img_rec);
        CSIMs(i,j,2) = CSIM(img,Img_rec);
        imwrite(uint8(Img_rec),['Rec_Img_IMATI_Im',num2str(i),'_sr_',num2str(j),'.png']);
        

        
        % -------------- CSIM-ALM-2D -----------------------------
        Opt.img_org = img;
        t0 = cputime;        
        [Img_rec] = csim_alm_inpaint_2D( img_masked,mask,Opt );
        figure;
        imshow(uint8(Img_rec));
        Times(i,j,3) = cputime-t0;
        % RecImages{i,j,3} = img_rec(1:N,1:M,:);
        PSNRs(i,j,3) = PSNR(img,Img_rec);
        SSIMs(i,j,3) = ssim_index(img,Img_rec);
        CSIMs(i,j,3) = CSIM(img,Img_rec);
        imwrite(uint8(Img_rec),['Rec_Img_CSIM_ALM_Im',num2str(i),'_sr_',num2str(j),'.png']);
    end
end

save('CSIM_IMAT_IMATI_2D_results.mat')
