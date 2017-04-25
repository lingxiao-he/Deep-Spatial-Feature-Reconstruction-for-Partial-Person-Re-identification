%% test Partial REID
% We provide the test codes for the Partial REID
% Lingxiao He et al, Deep Spatial Feature Reconstruction for
%Partial Person Re-identification: Alignment-free Approach, CVPR, 2018
% Plese kindly cite this paper if you compare DSR in your research.
%%
clear all
clc
if ~exist('opt_choice', 'var')
    opt_choice = 1;
end
num_bases = 128;
beta = 0.4;
alpha = 1.2;
num_iters = 5;
rank = 10;
if opt_choice==1
    sparsity_func= 'L1';
    epsilon = [];
elseif opt_choice==2
    sparsity_func= 'epsL1';
    epsilon = 0.01;
end
fname_save = sprintf('../results/sc_%s_b%d_beta%g_%s', sparsity_func, num_bases, beta, datestr(now, 30));
Binit = [];
N = 1; %if N=1,it is single-shot setting, if N>1, it is multi-shot setting
load('feature.mat') % Feature Extracted by ResNet50
for i = 1:length(Whole) % The number of Subject
    for j=1:length(Whole{i})
        GallerySpatial{i}{j} = Whole{i}{j};
        ProbeSpatial{i}{j} = Partial{i}{j};
    end
end

Dictionary = [];
for i = 1:length(GallerySpatial)
    subDictionary = [];
    for j = 1:N
        GallerySpatialFeature = multiscale(GallerySpatial{i}{j});
        subDictionary = [subDictionary;GallerySpatialFeature];
    end
    numEachGalImg(i) =  size(subDictionary,1);
    Dictionary = [Dictionary;subDictionary];
end
batch_size = size(Dictionary,1);

totalGalKeys = sum(numEachGalImg);
cumNumEachGalImg = [0; cumsum(numEachGalImg')];
AtA = Dictionary*Dictionary';


accuracy = zeros(1, rank);
for kk = 1:length(Partial{1})
    fprintf('probe %d testing\n',kk);
    for i = 1:length(ProbeSpatial)
        ProbeSpatialFeature = SF_Extraction(ProbeSpatial{i}{kk});
        [B S stat] = sparse_coding(AtA,alpha, Dictionary', double(ProbeSpatialFeature'), num_bases, beta, sparsity_func, epsilon, num_iters, batch_size, fname_save, Binit);
        for m = 1:length(numEachGalImg)
            recovery = S(cumNumEachGalImg(m)+1:cumNumEachGalImg(m+1),:)'*Dictionary(cumNumEachGalImg(m)+1:cumNumEachGalImg(m+1),:);
            single_score(m) = mean(sum((recovery-double(ProbeSpatialFeature)).^2,2));
        end
        score(:,i) = single_score;
    end
    rankscore = calrank(score,1:10,'ascend');
    fprintf('rank-1: %2.2f%%\n', rankscore(1)*100);
    accuracy = accuracy + calrank(score,1:10,'ascend')/length(Partial{1});
end
fprintf('Average rank-1: %2.2f%%\n',accuracy(1)*100);


