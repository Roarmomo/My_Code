clc
clear
rng('shuffle')
gpuDevice(1)

imds = imageDatastore('C:\Users\roar沫沫\Desktop\党宁的论文\特征图片\MAS\训练', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); %文件夹'MerchData'中的所有子文件夹的内容，每个文件夹即为一类 
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8,'randomized');

imds2 = imageDatastore('C:\Users\roar沫沫\Desktop\党宁的论文\特征图片\MAS\测试', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); %文件夹'MerchData'中的所有子文件夹的内容，每个文件夹即为一类 
[imdsTest,~] = splitEachLabel(imds2,0.9,'randomized');
net = alexnet;
% analyzeNetwork(net)
inputSize = net.Layers(1).InputSize;
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
% [learnableLayer,classLayer] 
numClasses = numel(categories(imdsTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);
layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:22) = freezeWeights(layers(1:22));
lgraph = createLgraphUsingConnections(layers,connections);
inputSize = net.Layers(1).InputSize;
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);
miniBatchSize = 256;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',30, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');
 gpuDevice(1)

        convnet = trainNetwork(augimdsTrain,layers,options);


        YPred_Train= classify(convnet,augimdsTrain,'MiniBatchSize',100);
        YTest_Train = imdsTrain.Labels;
        accuracy_T = sum(YPred_Train==YTest_Train)/numel(YTest_Train);
        YPred_V = classify(convnet,augimdsValidation,'MiniBatchSize',100);
        YTest_V = imdsValidation.Labels;
        accuracy_V = sum(YPred_V==YTest_V)/numel(YTest_V);
        YTest_V = dummyvar(double(YTest_V))'; % dummyvar requires Statistics and Machine Learning Toolbox
        YPred_V = dummyvar(double(YPred_V))';
%         if size(YTest_V,1)==size(YPred_V,1)
%             plotconfusion(YTest_V,YPred_V);
%             saveas(gcf,['D:\My_experience\特征图片删掉异常\结果\' ...
%                         '验证集_accuracy=' num2str(accuracy_V)  ...
%                         '.png']);
%         end
        
        
        YPred_T = classify(convnet,augimdsTest,'MiniBatchSize',100);
        YTest_T = imdsTest.Labels;
        test_accuracy = sum(YPred_T==YTest_T)/numel(YTest_T);
        YTest_T = dummyvar(double(YTest_T))'; % dummyvar requires Statistics and Machine Learning Toolbox
        YPred_T = dummyvar(double(YPred_T))';
       
        if size(YTest_T,1)==size(YPred_T,1)
            plotconfusion(YTest_T,YPred_T);
%             saveas(gcf,['C:\Users\roar沫沫\Desktop\党宁的论文\实验结果\MAS' ...
%                 'test_accuracy=' num2str(test_accuracy) ...
%                 'InitialLR=' num2str(grid_InitialLearnRate) 'DropFactor=' num2str(grid_LearnRateDropFactor) '_DropPeriod=' num2str(grid_LearnRateDropPeriod(p)) '.png']);
        end        
        

