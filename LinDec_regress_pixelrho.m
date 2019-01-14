%%
% Linear decoder to decode frames. This script regresses the performance over 
% each pixel against local properties of the receptive fields and various 
% measures of that pixel intensity distribution over frames

clearvars
clearvars -global
close all

%%
% define settings
SAVE = 0;
DIARY = 0;
%SAVE_F= 0;
%SAVE_ONLY = 0;
NORM = 1;
SHUFFLE = 1;
FIXEDPERM = 1;

basew = 5;
baseh = 5;
diaryname = 'LinDec_real_regress_pixelrho\\Output_linear_regression_pixelrho.txt';
if exist(diaryname,'file')
    delete(diaryname)
end

%%
% add strflab folder if it's the first run
FIRST_RUN = 0;
if FIRST_RUN    
    cd('C:\Users\Stef\Box Sync\PhD-Imperial-LafiteII\Data-Code\strflab_v1.45')
    strflabDir = get_function_dir('replicate_hierarchical');
    if isempty(strflabDir)
        error('Cannot find strflab directory!');
    end
    addpath(genpath(strflabDir))
    cd('C:\Users\Stef\Box Sync\FromWork\VRWM')
    
    % also add the subfolder
    addpath('C:\Users\Stef\Box Sync\FromWork\VRWM\UtilsVRWM')

end


%%
% Load the data
my_dir= 'C:\Users\Stef\Documents\PhD-Imperial-LafiteII\Data-Code\strflab_v1.45'; %'D:\Data-Code\Thomas_hierarchical_data\';
datafolder= 'Data\';
subfolders= {'region1\','region2\','region3\'};
% apply it to one region at a time for now
iregion = 1;

%%
lam_orders = 1:-1:-4;
nLam = numel(lam_orders);

    
%% ----------------------------------------------------------------- %%
% LINEAR DECODER FOR REAL DATA 
% ----------------------------------------------------------------- %%

% load the experimental data
tmp = load([datafolder,subfolders{iregion},'full_data.mat']);%,'val_inputs','val_set','training_inputs','training_set');
tmp.val_inputs = tmp.val_inputs/2.55e-4;
tmp.training_inputs = tmp.training_inputs/2.55e-4;

% no need for multiple iterations
Strain0 = tmp.training_inputs';
Stest0 = tmp.val_inputs';
R0 = tmp.training_set';
Rtest0 = tmp.val_set';
Rtest_raw0 = permute(tmp.raw_val_set,[1,3,2]); %10x103x50

% % normalise each image to 0 mean and 1 std
% for ii=1:size(Strain0,2)
%     Strain0(:,ii) = (Strain0(:,ii) - min(Strain0(:,ii)))./(max(Strain0(:,ii)) - min(Strain0(:,ii)));
% end
% for ii=1:size(Stest0,2)
%     Stest0(:,ii) = (Stest0(:,ii) - min(Stest0(:,ii)))./(max(Stest0(:,ii)) - min(Stest0(:,ii)));
% end

% reshape Strain0 and Stest0
Strain = Strain0;
Stest = Stest0;
Strain0 = reshape(Strain0,31,31,[]);
Stest0 = reshape(Stest0,31,31,[]);

% take the central patch
%Strain0 = Strain0(8:23,8:23,:);
%Stest0 = Stest0(8:23,8:23,:);
N = size(Strain0,1);
N2 = N*N;

clearvars tmp

%% First do the usual linear reconstruction

% Find lambda using cross validation
all_lambdas= 10.^(floor(log10(norm(diag(R0*R0'))))+lam_orders);
[lambda_real, idxlambdaopt] = trainOLE_5foldcv_lambda(R0,Strain,all_lambdas);

% now retrain using the optimal lambda
[R,mu_real,sigma_real,Rtest] = preprocess_normOLE(R0,1,Rtest0);

% compute optimal linear filter
K= Strain*R'*pinv(R*R' + 0*lambda_real*eye(size(R,1)));

Strain_hat = K*R;
Sbaseline = mean(Strain,2);

% reconstruct test data
Stest_hat = K*Rtest;

% compute correlation
Ctrain = diag(corr(Strain,Strain_hat));
Ctest = diag(corr(Stest,Stest_hat));
Ctest_pixels = diag(corr(Stest',Stest_hat'));
Ctest_base = (corr(Stest,Sbaseline));

%% ------------------------------------------------------------------- %%
% Now, extract relevant features against which to regress the performance
% --------------------------------------------------------------------- %

%% Extract relevant PIXEL-WISE image statistics
[Npixels, n_neurons] = size(Stest);
all_feats_pixels = {'meanPixels','stdPixels','skewnessPixels','kurtosisPixels','localstdPixels','localcorrPixels'};
names2plotP = {'mean','standard deviation','skewness','kurtosis','local std','local correlation'};
featsP.meanPixels = mean(Stest,2);
featsP.stdPixels = std(Stest,[],2);
%all_feat_pixels.range_pixels = max(Stest) - min(Stest);
featsP.skewnessPixels = skewness(Stest,[],2);
featsP.kurtosisPixels = kurtosis(Stest,[],2);
% now, the average standard deviation in a neighbour and the average
%% correlation in a neighbourhood
featsP.localstdPixels = zeros(Npixels,1);
featsP.localcorrPixels = zeros(Npixels,1);
sNpixels=sqrt(Npixels);
d3 = size(Stest0,3);
for ip1 = 1:sNpixels
    for ip2 = 1:sNpixels
        ip = sub2ind([sNpixels,sNpixels],ip1,ip2);
        tmp = Stest0(max(1,ip1-1):min(sNpixels,ip1+1),max(1,ip2-1):min(sNpixels,ip2+1),:);
        tmp = reshape(tmp,[],d3);
        featsP.localstdPixels(ip) = mean(std(tmp));
        tmp2 = squeeze(Stest0(ip1,ip2,:));
        Z = corr(tmp2,tmp');
        Z = Z(Z~=1);
        featsP.localcorrPixels(ip) = mean(Z);
    end
end

%% Now, RF statistics. First, load RF coverage
load('C:\Users\Stef\Box Sync\FromWork\VRWM\CoverageAnalysis_vs_linear_full\basicvariables.mat');

% compute the number of RFs at each pixel, while saving the binary mask
coverage = zeros(31);
[X,Y] = meshgrid(0:30,0:30);
offset = 1;
all_masks = zeros(Npixels,n_neurons);
for ii=1:103
    if ~isempty(rfradius{ii})
        xx = (offset+X - rfcent{ii}(1))*cos(rftheta{ii}) + (offset+Y - rfcent{ii}(2))*sin(rftheta{ii});
        yy = -(offset+X - rfcent{ii}(1))*sin(rftheta{ii}) + (offset+Y - rfcent{ii}(2))*cos(rftheta{ii});
        a = rfradius{ii}(1);
        b = rfradius{ii}(2);
        %coverage = coverage + double(((X - rfcent{ii}(1)).^2 + (Y - rfcent{ii}(2)).^2)<=rfradius{ii}^2);
        mask = double(((xx/a).^2 + (yy/b).^2)<=1);
        coverage = coverage + mask;
        all_masks(:,ii) = reshape(mask,[],1);
        %figure,imagesc(coverage),hold on,plot_ellipse(rfcent{ii}(1),rfcent{ii}(2),rfradius{ii}(1),rfradius{ii}(2),rftheta{ii},struct('color','k','linewidth',1.1));
    end
end

%
all_feats_rfs = {'numberRFs','meanThetas','varThetas','spreadThetas','rangeThetas'};
names2plotR = {'RF coverage', 'mean (\theta)', 'variance (\theta)', 'spread (\theta)', 'range (\theta)'};
featsRF = struct();
featsRF.numberRFs = reshape(coverage,[],1)';
% For each pixel, build a distribution of the thetas associated with RFs that include that pixel
all_thetas = cell(n_neurons,1);
for ii = 15:Npixels
    % get which RFs cover it
    idxrf = all_masks(ii,:);
    A = rftheta(logical(idxrf));
    if numel(A)>0
        all_thetas{ii} = zeros(1,numel(A));
        for it = 1:numel(A)
            all_thetas{ii}(it) = double(A{it});
        end
    else
        all_thetas{ii} = NaN;
    end
    featsRF.meanThetas(ii) = cos((pi+circularmean(all_thetas{ii}))/2);
    featsRF.varThetas(ii) = circularvariance(all_thetas{ii});
    featsRF.rangeThetas(ii) = max(all_thetas{ii})-min(all_thetas{ii});
    featsRF.spreadThetas(ii) = circular_spread(all_thetas{ii});
end
% eliminate NaNs by making them equal to the mean value
featsRF.meanThetas(isnan(featsRF.meanThetas))= mean(featsRF.meanThetas(~isnan(featsRF.meanThetas)));
featsRF.varThetas(isnan(featsRF.varThetas))= mean(featsRF.varThetas(~isnan(featsRF.varThetas)));
featsRF.rangeThetas(isnan(featsRF.rangeThetas))= mean(featsRF.rangeThetas(~isnan(featsRF.rangeThetas)));
featsRF.spreadThetas(isnan(featsRF.spreadThetas))= mean(featsRF.spreadThetas(~isnan(featsRF.spreadThetas)));

%% collect all the features together
feats_pixels2use = {'meanPixels','stdPixels','skewnessPixels','kurtosisPixels'};
names2plotP = {'mean','standard deviation','skewness','kurtosis'};
feats_rfs2use = {'numberRFs','meanThetas','spreadThetas'};
names2plotR = {'RF coverage', 'mean (\theta)', 'spread (\theta)'};
feats_rfs = zeros(Npixels,numel(feats_rfs2use));
for ii = 1:numel(feats_rfs2use)
    feats_rfs(:,ii) = featsRF.(feats_rfs2use{ii});
end
feats_pixels = zeros(Npixels,numel(feats_pixels2use));
for ii = 1:numel(feats_pixels2use)
    feats_pixels(:,ii) = featsP.(feats_pixels2use{ii});
end
features = [feats_rfs, feats_pixels];
%names = cat(2,feats_rfs2use,feats_pixels2use);
names = cat(2,names2plotR,names2plotP);
all_names = names;
all_names{end+1} = 'y';
% standardise them and the dependent variable
std_features = zscore(features);
Y = zscore(Ctest_pixels);

%%
ifig = 1;
hf(ifig) = figure('Name','correlation_matrix_pixelrho');
prepare_figure_v2(hf(ifig),struct('width',2*basew,'height',1.5*baseh,'axesfontsize',8,'textfontsize',8));
ifig=ifig+1;
imagesc(corr([std_features,Y]))
axis square
colormap(greyl())
colorbar()
set(gca,'YTick',1:numel(all_names),'YTickLabels',all_names)
set(gca,'XTick',1:numel(all_names),'XTickLabels',all_names,'XTickLabelRotation',45)
hold on,plot([0,numel(all_names)+.5],.5+[numel(feats_rfs2use),numel(feats_rfs2use)],'r','LineWidth',1.3)
hold on,plot(.5+[numel(feats_rfs2use),numel(feats_rfs2use)],[0,numel(all_names)+.5],'r','LineWidth',1.3)
%figure,plot(corr(std_features,Y)),set(gca,'XtickLabels',names,'XTickLabelRotation',45)


%% regress reconstruction performance against features
if DIARY
    diary(diaryname)
end
% create cv partition object with 10 folds
c = cvpartition(Npixels,'kfold',10); %50,'LeaveOut');
% TODO: should I 
% vanilla linear regression
mylm = fitlm(std_features,Y,'linear','VarNames',all_names);
% robust linear regression
mylm_rob = fitlm(std_features,Y,'linear','RobustOpt','on','VarNames',all_names);
% improve the robust regression by adding and removing coefficients. DO
% NOT include interaction terms
mylm1 = step(mylm,'NStep',40,'Criterion','BIC','Upper','linear');
% same as above but with interaction and quadrtic terms
mylm2 = step(mylm,'NStep',40,'Criterion','BIC');
ci = coefCI(mylm);
mylmtable = table(ci(2:end,1),ci(2:end,2),(ci(2:end,2)-ci(2:end,1))/2,...
    'RowNames',names','VariableNames',{'lowCI','highCI','pm'});
ci = coefCI(mylm1);
mylm1table = table(ci(2:end,1),ci(2:end,2),(ci(2:end,2)-ci(2:end,1))/2,...
    'RowNames',names([1,3,4,5])','VariableNames',{'lowCI','highCI','pm'});
display(' ---------- VANILLA LM ---------------------------------')
mylm
mylmtable
display(' ---------- ROBUST LM ---------------------------------')
mylm_rob
display(' ---------- MINIMAL LM ---------------------------------')
mylm1
mylm1table
display(' ---------- INTERACTION LM ---------------------------------')
mylm2

%% use bootstrap to check the SE of the vanilla lm
Nreps=2000;
mylm_coeffs = zeros(numel(names)+1,Nreps);
mylm_res = zeros(2,Nreps);
for irep = 1:Nreps
    [features2,idx] = datasample(features,Npixels);
    Ctest_pixels2 = Ctest_pixels(idx);
    std_features2 = zscore(features2);
    Y2 = zscore(Ctest_pixels2);
    mylm_tmp = fitlm(std_features2,Y2,'linear','VarNames',all_names);
    mylm_coeffs(:,irep) = table2array(mylm_tmp.Coefficients(:,1));
    mylm_res(1,irep) = mylm_tmp.Rsquared.Ordinary;
    mylm_res(2,irep) = mylm_tmp.Rsquared.Adjusted;
end

%%
% MORE things I can get: coefCI(mylm)
% Test the null hypothesis that all predictor variable coefficients are equal 
% to zero versus the alternate hypothesis that at least one of them is different from zero.
% [p,F,d] = coefTest(mylm);
% Now, perform a hypothesis test on the coefficients of the first and second predictor variables.
% H = [0 1 0 0; 0 0 1 0]; % first column is the intercept
% [p,F,d] = coefTest(lm,H)
% % SVM regression with gaussian kernel with cross validation
% mysvm = fitrsvm(std_features,Y,'CrossVal','on','CVPartition',c,'KernelFunction','gaussian');
% mytree = fitrtree(std_features,Y,'CrossVal','on','CVPartition',c);
% L1=kfoldLoss(mysvm,'mode','individual');L2=kfoldLoss(mytree,'mode','individual');
% display('---------------------------------')
% mylm
% figure,plot(predict(mylm,std_features),Y,'o')
% We used a linear regression model and leave-one-out cross-validation to estimate the weights of these 
% features
% diary off

% TODO: adjust p-values with bonferroni correction for multiple comparisons

%% Relative importance of regressors. TODO: check this
% We show the total contribution toward explaining the average difference 
% in decoding performance—that is, the product between the estimated 
% coefficient and the average of the factor difference biAxi—normalized to a total 
% effect size of 1. Error bars shown are the standard errors of the coefficients in the 
% linear regression model scaled by the mean difference ?x.
coeffvals = table2array(mylm.Coefficients(:,1));
pratt = coeffvals(1:end).*[1;corr(std_features,Y)];
divider = sum(pratt);
pratt= pratt./divider;
ci_pratt = table2array(mylm.Coefficients(:,2));
ci_pratt = ci_pratt.*[1;corr(std_features,Y)];
ci_pratt = ci_pratt/divider;
pratt2 = coeffvals(2:end).*mean(Y); %mean(std_features)';
pratt2= pratt2./sum(pratt2);
%inv_pvals= 1./table2array(mylm.Coefficients(2:end,4));
hf(ifig)=figure('Name','Vanilla_lm_pratt_indices_pixelrho');
prepare_figure_v2(hf(ifig),struct('width',2*basew,'height',baseh,'axesfontsize',8,'textfontsize',8));
ifig=ifig+1;
bar(pratt,'LineWidth',1.2,'FaceColor',[0,0,0],'FaceAlpha',.6,'EdgeColor',.35*ones(1,3))
hold on
plot(repmat((1:numel(names)+1)',1,2)',[pratt-ci_pratt,pratt+ci_pratt]','k','LineWidth',1.2)
%hold on,plot(pratt2,'LineWidth',1.2)
%,plot(inv_pvals./max(inv_pvals))
%legend({'pratt measure 1'},'Location','southeast')
set(gca,'XTickLabels',cat(2,{'Intercept'},names),'XTickLabelRotation',45)
%ylim([2*min(pratt),1.2*max(pratt)])
ylabel('Contribution')
box off

%% Now an alternative: use permutation importance
out = permutation_importance(mylm,std_features,Y,1000);
%%
% compute percentange of change
out.change_adjrsq = (mylm.Rsquared.Adjusted - out.adjrsq)/mylm.Rsquared.Adjusted*100;
out.change_mse = (out.mse - mylm.MSE)/mylm.MSE*100;
% show the relative importance
hf(ifig)=figure('Name','Vanilla_lm_perm_importance_pixelrho');
prepare_figure_v2(hf(ifig),struct('width',2*basew,'height',baseh,'axesfontsize',8,'textfontsize',8));
ifig=ifig+1;
mu = mean(out.change_adjrsq);
sig = std(out.change_adjrsq);
sig = sig./sum(mu);
mu = mu./sum(mu);
bar(mu,'LineWidth',1.2,'FaceColor',[0,0,0],'FaceAlpha',.6,'EdgeColor',.35*ones(1,3))
hold on
plot(repmat((1:numel(names))',1,2)',[mu'-sig',mu'+sig']','k','LineWidth',1.2)
%hold on,plot(pratt2,'LineWidth',1.2)
%,plot(inv_pvals./max(inv_pvals))
%legend({'pratt measure 1'},'Location','southeast')
set(gca,'XTickLabels',names,'XTickLabelRotation',45)
%ylim([2*min(pratt),1.2*max(pratt)])
ylim([-.1,1.1])
ylabel('Contribution')
box off

%% Now use models with regularisation and inbuilt feature selection
% kridge = 0:1e-1:100;
% myridge = ridge(Y,std_features,kridge);
% figure,plot(kridge,myridge)
% % plot predictions
% figure,plot(std_features*myridge(:,500),Y,'o')
% ELASTIC NET WITH ALPHA = 0.95 (basically LASSO)
[lassomodel,lassoFit] = lasso(std_features,Y,'CV',c,'Alpha',0.95,'PredictorNames',...
    names,'MCReps',10,'LambdaRatio',1e-3);
% % ELASTIC NET WITH ALPHA = 0.05 (basically RIDGE)
% [ridgemodel,ridgeFit] = lasso(std_features,Y,'CV',c,'Alpha',0.05,'PredictorNames',names,'MCReps',5);

%% TODO: Use bootstrap to get confidence intervals on regressors at the optimal lambda value (See NOTES.txt)
% This will give me error bars
lambda_lasso = lassoFit.Lambda1SE;
betas_lasso2 = lassomodel(:,lassoFit.Index1SE);
Yhat2 = std_features*betas_lasso2 + lassoFit.Intercept(lassoFit.Index1SE);
Nreps=2000;
betasB_lasso2 = zeros(numel(betas_lasso2),Nreps);
for irep = 1:Nreps
    [std_features2,idx] = datasample(std_features,Npixels);
    Y2 = Y(idx);
    betasB_lasso2(:,irep) = lasso(std_features2,Y2,'Lambda',lambda_lasso,'Alpha',0.95,'PredictorNames',names);
end
[betasB_lasso2,IDX] = sort(betasB_lasso2,2);
%%
lowerci = 2.5*Nreps/100;
upperci = 97.5*Nreps/100;
betasCI_lasso = betasB_lasso2(:,[lowerci,upperci]);
% TODO: should I also do a permutation test(See NOTES.txt)? It's not really recommended
% % TODO: check how I get the p-values
% for ii = 1:size(betasB_lasso,1)
%     if betasB_lasso(ii)>0
%         pval(ii) = sum(betasB_lasso(ii,:)>betasB_lasso(ii))/Nreps;
%     else
%         pval(ii) = sum(betasB_lasso(ii,:)<betasB_lasso(ii))/Nreps;
%     end
% end

%% TODO: print out regressors with confidence intervals from
% bootstrap (maybe p-values if you've done the perm test)
lasso_table = table(betas_lasso2,betasCI_lasso(:,1),betasCI_lasso(:,2),...
    (betasCI_lasso(:,2) - betasCI_lasso(:,1))/2,...
    'RowNames',names','VariableNames',{'beta','lowCI','highCI','pm'});
lasso_table
% get rsquared
[rsq, adj_rsq] = get_rsquared(Y,Yhat2,numel(names)+1);
sprintf('LASSO. R-squared: %2.3f. Adjusted R-squared: %2.3f\n',rsq,adj_rsq)

%% LASSO PLOTS
%figure,plot(log(1./lassoFit.Lambda),lassomodel')
%legend(names)
% MSE versus lambda
ax1_lasso = lassoPlot(lassomodel,lassoFit,'PlotType','CV');
set(gcf,'Name','lasso1_cvplot_pixelrho');
hf(ifig) = gcf;
prepare_figure_v2(hf(ifig),struct('width',2*basew,'height',baseh,'axesfontsize',8,'textfontsize',8));
ifig=ifig+1;
box off
% Regressors path
ax2_lasso = lassoPlot(lassomodel,lassoFit,'PlotType','L1','PredictorNames',names);
xl = xlim();
xlim([xl(1),xl(2)+1.2]);
set(findobj('Type','line'),'LineWidth',1.2)
set(gcf,'Position',[360.3333  249.0000  773.3333  368.6667],'Name','lasso1_l1plot_pixelrho');
hf(ifig) = gcf;
prepare_figure_v2(hf(ifig),struct('width',2*basew,'height',baseh,'axesfontsize',8,'textfontsize',8));
ifig=ifig+1;
hl=legend('show','Location','best');
set(hl,'box','off','Position',[0.6253    0.3127    0.2196    0.4390]);
box off
% % copy figures two new axes (not good)
% oldfig = gcf;
% h3 = figure('Position',[60  60, 560  500]); %create new figure
% s1 = subplot(2,1,1); %create and get handle to the subplot axes
% s2 = subplot(2,1,2);
% fig1 = get(ax_lasso1,'children'); %get handle to all the children in the figure
% fig2 = get(ax_lasso2,'children');
% copyobj(fig1,s1); %copy children to new parent axes i.e. the subplot axes
% copyobj(fig2,s2);
% xl = xlim();
% xlim([xl(1),xl(2)+1]);
% xl = ylim();
% %ylim([xl(1),xl(2)+1]);
% hl2=legend(get(hl,'String'),'box','off','Location','eastoutside'); %'Position',[0.6608    0.1184    0.2327    0.3853]);

%
%lassoPlot(lassomodel,lassoFit,'PlotType','Lambda')
%figure,plot(std_features*lassomodel(:,lassoFit.IndexMinMSE),Y,'o')
%hold on,plot(std_features*lassomodel(:,lassoFit.Index1SE),Y,'o')
%title(num2str([corr(std_features*lassomodel(:,lassoFit.IndexMinMSE),Y)])) %,corr(std_features*lassomodel(:,lassoFit.IndexMinMSE),Y)]))

%% Now permutation importance with Lasso
outL = permutation_importance(struct('coeffs',betas_lasso2,'bias',lassoFit.Intercept(lassoFit.Index1SE)),...
    std_features,Y,1000);

%% TODO: repeat LASSO with interaction terms
% Then, bootstrap again
LASSO2 = 0;
if LASSO2
    num_regr = size(std_features,2);
    std_features_int = zeros(Npixels,num_regr+(num_regr*(num_regr-1))/2);
    std_features_int(:,1:num_regr)=std_features;
    iter=num_regr+1;
    names2 = names;
    for ii=1:num_regr
        for jj=ii+1:num_regr
            std_features_int(:,iter) = std_features(:,ii).*std_features(:,jj);
            names2{iter} = [names{ii},':',names{jj}];
            iter = iter+1;
        end
    end
    std_features_int = zscore(std_features_int);

    %% Now use models with regularisation and inbuilt feature selection
    % ELASTIC NET WITH ALPHA = 0.95 (basically LASSO)
    [lassomodel2,lassoFit2] = lasso(std_features_int,Y,'CV',c,'Alpha',0.95,...
        'PredictorNames',names2,'MCReps',10,'LambdaRatio',1e-3);

    %% TODO: Use bootstrap to get confidence intervals on regressors at the optimal lambda value (See NOTES.txt)
    % This will give me error bars
    lambda_lasso2 = lassoFit2.Lambda1SE;
    betas_lasso2 = lassomodel2(:,lassoFit2.Index1SE);
    Nreps=2000;
    betasB_lasso2 = zeros(numel(betas_lasso2),Nreps);
    for irep = 1:Nreps
        [std_features2,idx] = datasample(std_features_int,Npixels);
        Y2 = Y(idx);
        betasB_lasso2(:,irep) = lasso(std_features2,Y2,'Lambda',lambda_lasso2,...
            'Alpha',0.95,'PredictorNames',names2);
    end
    [betasB_lasso2,IDX] = sort(betasB_lasso2,2);
    %%
    lowerci = 2.5*Nreps/100;
    upperci = 97.5*Nreps/100;
    betasCI_lasso2 = betasB_lasso2(:,[lowerci,upperci]);
    % TODO: should I also do a permutation test(See NOTES.txt)? It's not really recommended

    %% TODO: print out regressors with confidence intervals from
    % bootstrap (maybe p-values if you've done the perm test)
    lasso_table2 = table(betas_lasso2,betasCI_lasso2(:,1),betasCI_lasso2(:,2),...
        'RowNames',names2,'VariableNames',{'beta','lowCI','highCI'});
    lasso_table2

    %% LASSO PLOTS.
    %figure,plot(log(1./lassoFit.Lambda),lassomodel')
    %legend(names)
    % MSE versus lambda
    ax1_lasso2 = lassoPlot(lassomodel2,lassoFit2,'PlotType','CV');
    set(gcf,'Name','lasso2_cvplot_pixelrho')
    hf(ifig) = gcf;
    ifig=ifig+1;
    % Regressors path
    ax2_lasso2 = lassoPlot(lassomodel2,lassoFit2,'PlotType','L1','PredictorNames',names2);
    xl = xlim();
    xlim([xl(1),xl(2)+2]);
    oldlines = findobj(gcf,'Type','line');
    oldlines = oldlines(end:-1:1);
    lines2plot = abs(betas_lasso2)>0;
    if exist('all_lines','var')
        clearvars all_lines
    end
    all_lines = oldlines(1:2);
    iter= 3;
    for ii=3:numel(oldlines)
        if lines2plot(ii-2)
            all_lines(iter) = oldlines(ii);
            iter=iter+1;
        else
            delete(oldlines(ii));
        end
    end
    Nlines = numel(all_lines);
    linescmap = lines(7);
    for ii=3:9
        set(all_lines(ii),'LineStyle','-','Color',linescmap(ii-2,:));
    end
    for ii=10:min(Nlines,16)
        set(all_lines(ii),'LineStyle','--','Color',linescmap(ii-9,:));
    end
    for ii=17:min(Nlines,23)
        set(all_lines(ii),'LineStyle','-.','Color',linescmap(ii-16,:));
    end
    set(all_lines,'LineWidth',1.2)
    set(gcf,'Position',[360.3333  249.0000  773.3333  368.6667],'Name','lasso2_l1plot_pixelrho');
    hl=legend('show','Location','best');
    set(hl,'box','off','Position',[0.6253    0.3127    0.2196    0.4390]);
    box off
    hf(ifig) = gcf;
    ifig=ifig+1;
end

diary off

%% save figs
SAVEF = 1;
plotdir = 'LinDec_real_regress_pixelrho\\figTNR\\';
if SAVEF
    for ii=1:numel(hf)
        savefig(hf(ii),[plotdir,get(hf(ii),'Name'),'.fig'])
        export_fig(hf(ii),[plotdir,get(hf(ii),'Name')],...
            '-png','-pdf','-transparent','-painters')
        figure(hf(ii))
        print([plotdir,get(hf(ii),'Name')],'-depsc','-painters');
        print([plotdir,get(hf(ii),'Name')],'-dsvg','-painters');
    end
end
%%
if SHUFFLE
    if FIXEDPERM
        if NORM
            savestr = '_fixshuffle_norm_';
        else
            savestr = '_fixshuffle_';
        end
    else
        if NORM
            savestr = '_shuffle_norm_';
        else
            savestr = '_shuffle_';
        end
    end
else
    if NORM
        savestr = '_norm_';
    else
        savestr = '_';
    end
end
if SAVE
    save();
end

% keep this in case you switch to cross-validation tests
% %test_idx_start(irun)= itest_start;
% [m_tmp, i_tmp] = max(reshape(muC',1,[]));
% best_idx(irun) = i_tmp;
% best_corr(irun)= m_tmp;
% keep_C_all{irun}= C{i_tmp};
% 
% save('Batch_seven_avg_measures_CVtestsubset_movie1.mat','test_idx_start','best_idx','best_corr','keep_C_all');



%% NOTE: https://www.sciencedirect.com/science/article/pii/S0893608018301722