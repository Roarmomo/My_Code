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

net=vgg19();
inputSize = net.Layers(1).InputSize;
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

layersTransfer = net.Layers(1:end-3);
Layers = [layersTransfer
        fullyConnectedLayer(8,'Name','fc3')
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];

  grid_InitialLearnRate=0.0001;
  grid_LearnRateDropFactor=0.9 ;
  grid_LearnRateDropPeriod= 3;


        miniBatchSize =256;
        validationFrequency = floor(numel(imdsTrain.Labels)/miniBatchSize);

        options = trainingOptions('sgdm',...
              'LearnRateSchedule','piecewise',...
              'InitialLearnRate',grid_InitialLearnRate,...
              'LearnRateDropFactor',grid_LearnRateDropFactor,... 
              'LearnRateDropPeriod',grid_LearnRateDropPeriod,... 
              'MaxEpochs',30,...
              'MiniBatchSize',miniBatchSize,...
              'Plots','training-progress');
            gpuDevice(1)

        convnet = trainNetwork(augimdsTrain,Layers,options);


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
        


