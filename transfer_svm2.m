clc
clear
rng('shuffle')
gpuDevice(1)
fprintf('开始提取特征');
imds = imageDatastore('C:\Users\roar沫沫\Desktop\党宁的论文\特征图片\MAS\训练', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); %文件夹'MerchData'中的所有子文件夹的内容，每个文件夹即为一类 
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8,'randomized');

imds2 = imageDatastore('C:\Users\roar沫沫\Desktop\党宁的论文\特征图片\MAS\测试', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); %文件夹'MerchData'中的所有子文件夹的内容，每个文件夹即为一类 
[imdsTest,~] = splitEachLabel(imds2,0.9,'randomized');

net=alexnet();
inputSize = net.Layers(1).InputSize;
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);
layersTransfer = net.Layers(1:end-3);
Layers = [layersTransfer
        fullyConnectedLayer(8,'Name','fc3')
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
layer = 'fc8';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');
YTrain = imdsTrain.Labels;
fprintf('开始分类');
classifier = fitcecoc(featuresTrain,YTrain);
YPred = predict(classifier,featuresTest);
YTest = imdsTest.Labels;

YPred_T = classify(netTransfer,augimdsTest,'MiniBatchSize',100);
YTest_T = imdsTest.Labels;
test_accuracy = sum(YPred_T==YTest_T)/numel(YTest_T);
Acc=sum(YPred==YTest_T)/numel(YTest_T);
