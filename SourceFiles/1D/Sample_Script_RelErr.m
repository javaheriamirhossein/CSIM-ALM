clear
close all

DictType = 'WP';

K = 64; 
BlSize = 8;
m = BlSize^2;


% Data_type:
% --------------------------------------------
%   'k_sparse':      strictly sparse signal generated at random
%   'compressible':  compressible signal generated at random

Data_type = 'k_sparse';

%% ---------- Dictionary --------------------

switch DictType
    

    case 'WP'
        D = wmpdictionary(m,'lstcpt',{{'wpsym4',6}});
        D = D(1:m,1:K);
        D = full(D);
    case 'DCT'
        Pn = ceil(sqrt(K));
        D = zeros(BlSize,Pn);
        for k = 0:1:Pn-1
            V = cos([0:1:BlSize-1]'*k*pi/Pn);
            if k>0, V = V-mean(V); end;
            D(:,k+1) = V/norm(V);
        end
        D = kron(D,D);
        D = D(1:m,1:K);
end

%--------------------------
D = normc(D);
I = displayDictionaryElementsAsImage(D);
imwrite((I),'Dictionary.png');
Dinv = pinv(D);


%% ------- Parameters of algorithms ------------------
T = ceil(0.1*K);
Opt.L = T;
Opt.T = T;

Opt.Dpinv = Dinv;
Opt.sigma_min = 1e-6;
Opt.sigma_decrease_factor = 0.5;


Opt.alpha = 0.2;
Opt.beta = 50;
Opt.lambda = 0.7;


maxIteration = 100;
Opt.maxIter = maxIteration;


Opt.maxTime = 8;
Opt.STOPPING_TIME = -2;


maxTime = 20;
tolerance = 1e-3;
STOPPING_TIME = -2;
STOPPING_GROUND_TRUTH = -1;
STOPPING_DUALITY_GAP = 1;
STOPPING_SPARSE_SUPPORT = 2;
STOPPING_OBJECTIVE_VALUE = 3;
STOPPING_SUBGRADIENT = 4;


opts.mu = 2^8;
opts.beta = 2^5;
opts.tol = 1e-3;
opts.maxit = 100;
opts.TVnorm = 1;
opts.nonneg = true;


Niteration = maxIteration;
Opt.Niteration = Niteration;
Opt.errorshow = 1;
Opt.report = 1;


%-----------------------------------------------
Ntrial = 200;    % Number of random trials

%-----------------------------------------------
% List the name of algorithms and choose them with Ind1 - Ind4

methods1 = {'FISTA'; 'SL0'};
methods2 = {'csim_alm_alpha'; 'csim_alm'};
methods3 = {'SolveDALM_ST'};
methods4 = {'IMATCS'};

N1 = length(methods1);    
N2 = length(methods2);
N3 = length(methods3);
N4 = length(methods4);

Ind1 = [1,2];
Ind2 = [1,2];
Ind3 = [1];
Ind4 = [1];

Nal = N1+N2+N3+N4;

% ----------------------------------------------
SamplingRatio_vec = [0.2,0.4];
N_SR = length(SamplingRatio_vec);

% ----------------------------------------------

times = nan(Nal,Ntrial);
Times = nan(Nal,N_SR);

err_vec = nan(Nal,Ntrial,Niteration);
ERROR_mean = nan(N_SR,Nal,Niteration);

time_vec = zeros(Nal,Ntrial,Niteration);
Times_mean = zeros(N_SR,Nal,Niteration);



% ------- Sparse Signal Generation -------------
S = zeros(K,Ntrial);
X = zeros(m,Ntrial);

switch Data_type
    case 'compressible'
        for i = 1:Ntrial
            S(:,i) = randcs(K, 1, 1, 'gam');
            s_scale = norm(S(:,i));
            S(:,i) = S(:,i)/s_scale;
            X(:,i) = D*S(:,i);
        end
    case 'k_sparse'
        for i = 1:Ntrial
            S(:,i) = GenSparseVec(K,T);
            s_scale = norm(S(:,i));
            S(:,i) = S(:,i)/s_scale;
            X(:,i) = D*S(:,i);
        end

end


% -----------------------------------------------------------------------
Opt.D0 = eye(K);
scale_x = 1;
mean_x = 0;

for j = 1:N_SR
    
    
    SampleRatio = SamplingRatio_vec(j);
    m_s = ceil(SampleRatio*m);
    
    for i = 1:Ntrial

        % --- Random sampling mask ----------------
        mask = false(m,1);
        mask(randperm(m,m_s))= true;
    
        x = X(:,i);
        s = S(:,i);
        
                
        Opt.U0 = reshape(x,BlSize,[]);
        Opt.x0 = s;
        Opt.mean_x = mean_x;
        Opt.max_x = scale_x;
        Opt.scale_x = scale_x;
        Opt.mask = mask;
        Ds = D(mask,:);
        A = eye(length(mask));
        A = A(mask,:);
        
        % -----------------------------------------
        y = mask.*(x - mean_x);
        % std_x = std(x);
        % n = 0.1*std_x*randn(size(x));
        % y = y+n;
        y = y/scale_x;
        ys = y(mask);
        
        
        % --------------- Recovery ----------------
        
        for k = Ind1
            methodHandle = str2func(methods1{k}) ;
            t0 = cputime;
            [~,rel_err,time_it] = methodHandle(Ds,ys,Opt);
            times(k,i) = cputime - t0;
            err_vec(k,i,:) = rel_err(1:Niteration);
            time_vec(k,i,:) = time_it(1:Niteration);
        end
        
        for k = Ind2
            methodHandle = str2func(methods2{k}) ;
            t0 = cputime;
            [~,~,rel_err,time_it] = methodHandle(D,y,Opt);
            times(N1+k,i) = cputime - t0;
            err_vec(N1+k,i,:) = rel_err(1:Niteration);
            time_vec(N1+k,i,:) = time_it(1:Niteration);
        end
        
        for k = Ind3
            methodHandle = str2func(methods3{k}) ;           
            t0 = cputime;
            [~,rel_err,time_it]= methodHandle(Ds,ys,Opt, 'maxiteration', maxIteration,'tolerance',tolerance,'maxtime',maxTime);
            times(N1+N2+k,i)=cputime - t0;
            err_vec(N1+N2+k,i,:)=rel_err(1:Niteration);
            time_vec(N1+N2+k,i,:)=time_it(1:Niteration);
        end
        
        for k = Ind4
            
            methodHandle = str2func(methods4{k}) ;
            t0 = cputime;
            
            [~,rel_err,time_it] = methodHandle(ys,Ds,1,1e-6,Opt);                

            times(N1+N2+N3+k,i)=cputime - t0;           
            err_vec(N1+N2+N3+k,i,:) = rel_err(1:Niteration);
            time_vec(N1+N2+N3+k,i,:) = time_it(1:Niteration);
            
        end
        
        
    end

    ERROR_mean(j,:,:) = nanmean(err_vec,2);
    Times_mean(j,:,:) = nanmean(time_vec,2);
    Times(:,j) = nanmean(times,2);
end


%% ----------------------------------------------------------------------

KeepInd = [Ind1 Ind2+N1 Ind3+N1+N2 Ind4+N1+N2+N3];


legendStr = {'FISTA'; 'SL0'; ...
    'CSIM-ALM-$\alpha$'; 'CSIM-ALM';...
    'DALM';...
    'IMATCS'};

legendStr = legendStr(KeepInd);

ERROR_mean = 100*ERROR_mean(:,KeepInd,:);
Times_mean = Times_mean(:,KeepInd,:);
% Times=Times(KeepInd,:);

% ----------------------------------------------------------------
for nfig = 1:N_SR
    
    titleStr=['Sparse Recovery From ',num2str(SamplingRatio_vec(nfig)),' ', 'Random Sampling For ', num2str(m), 'x',num2str(K),' ', DictType,' Dictionary'];
    figname = [DictType,'_',num2str(K),'_SampleRatio',num2str(nfig),'_DataType_',num2str(Data_type)];

    figure
    set(gcf, 'Position', [200 100 750 500])
    Error_plot = squeeze(ERROR_mean(nfig,:,:));
    loglog(1:Niteration,Error_plot','-s','LineWidth',1.5)
    ylabel('Relative error (\%)','interpreter','latex','FontSize',13)
    legend(legendStr,'location','southeastoutside','interpreter','latex')
    axis([-inf inf -inf inf])
    title(titleStr)
    grid on
%     savefig(gcf,['RelErr_',figname,'.fig']);
%     saveas(gcf,['RelErr_',figname,'.eps']);
%     close(gcf)
    
    
    figure
    set(gcf, 'Position', [200 100 750 500])
    Time_plot = squeeze(Times_mean(nfig,:,:));
    loglog(1:Niteration,Time_plot','-s','LineWidth',1.5)
    xlabel('Iteration','interpreter','latex','FontSize',13)
    ylabel('Time(s)','interpreter','latex','FontSize',13)
    legend(legendStr,'location','southeastoutside','interpreter','latex')
    axis([-inf inf -inf inf])
    title(titleStr)
    grid on
%     savefig(gcf,['Time_',figname,'.fig']);
%     saveas(gcf,['Time_',figname,'.eps']);
%     close(gcf)
    
end


variables_list = who;
save(['Missing_RelErr_DataType_',num2str(Data_type),'Dic_',DictType,'_K',num2str(K),'.mat'],variables_list{:})
