%%
% Linear decoder to decode frames. This script regresses the performance over
% each pixel against local properties of the receptive fields and various 
% measures of that pixel intensity distribution over frames
% Results have been published here:
% Garasto, S., et al. (in press) ``Neural sampling strategies for visual stimulus 
% reconstruction from two-photon imaging of mouse primary visual cortex''. 
% 9th International IEEE EMBS Conference on Neural Engineering.

clearvars
clearvars -global
close all

%%
% define settings
SAVE = 0;
DIARY = 0;
NORM = 1;
SHUFFLE = 1;
FIXEDPERM = 1;

% figures settings
basew = 5;
baseh = 5;
diaryname = 'LinDec_real_regress_pixelrho\\Output_linear_regression_pixelrho.txt';
if exist(diaryname,'file')
    delete(diaryname)
end

%%
% add strflab folder (where data is located) if it's the first run
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

% assign training and test data
Strain0 = tmp.training_inputs';
Stest0 = tmp.val_inputs';
R0 = tmp.training_set';
Rtest0 = tmp.val_set';
Rtest_raw0 = permute(tmp.raw_val_set,[1,3,2]); %10x103x50

% reshape Strain0 and Stest0
Strain = Strain0;
Stest = Stest0;
Strain0 = reshape(Strain0,31,31,[]);
Stest0 = reshape(Stest0,31,31,[]);

% get stimulus width/height in pixels and total number of pixels
N = size(Strain0,1);
N2 = N*N;

clearvars tmp

%% First do the usual linear reconstruction of the visual stimulus from neural responses

% Find lambda using cross validation (L2 regularisation)
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
for ii = 1:Npixels
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

% vanilla linear regression
mylm = fitlm(std_features,Y,'linear','VarNames',all_names);
% robust linear regression
mylm_rob = fitlm(std_features,Y,'linear','RobustOpt','on','VarNames',all_names);
% improve the robust regression by adding and removing coefficients. DO
% NOT include interaction terms
mylm1 = step(mylm,'NStep',40,'Criterion','BIC','Upper','linear');
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

%% Relative importance of regressors.
% We show the total contribution toward explaining the linear decoder performance
% using permutation importance
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
set(gca,'XTickLabels',names,'XTickLabelRotation',45)
ylim([-.1,1.1])
ylabel('Contribution')
box off

%% Now use models with regularisation and inbuilt feature selection
% ELASTIC NET WITH ALPHA = 0.95 (basically LASSO)
[lassomodel,lassoFit] = lasso(std_features,Y,'CV',c,'Alpha',0.95,'PredictorNames',...
    names,'MCReps',10,'LambdaRatio',1e-3);

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

%% print out regressors with confidence intervals from
% bootstrap (maybe p-values if you've done the perm test)
lasso_table = table(betas_lasso2,betasCI_lasso(:,1),betasCI_lasso(:,2),...
    (betasCI_lasso(:,2) - betasCI_lasso(:,1))/2,...
    'RowNames',names','VariableNames',{'beta','lowCI','highCI','pm'});
lasso_table
% get rsquared
[rsq, adj_rsq] = get_rsquared(Y,Yhat2,numel(names)+1);
sprintf('LASSO. R-squared: %2.3f. Adjusted R-squared: %2.3f\n',rsq,adj_rsq)

%% LASSO PLOTS
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

%% Now permutation importance with Lasso
outL = permutation_importance(struct('coeffs',betas_lasso2,'bias',lassoFit.Intercept(lassoFit.Index1SE)),...
    std_features,Y,1000);

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

