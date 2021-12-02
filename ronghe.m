clear
clc
load C:\Users\roar沫沫\Desktop\党宁的论文\训练好的网络\alexnet
load C:\Users\roar沫沫\Desktop\党宁的论文\训练好的网络\alexnet_svm
net_alexnet=netTransfer;
alexnet_svm=classifier;
load C:\Users\roar沫沫\Desktop\党宁的论文\训练好的网络\googlenet
load C:\Users\roar沫沫\Desktop\党宁的论文\训练好的网络\googlenet_svm
net_googlenet=netTransfer;
googlenet_svm=classifier;
load C:\Users\roar沫沫\Desktop\党宁的论文\训练好的网络\inceptionv3
load C:\Users\roar沫沫\Desktop\党宁的论文\训练好的网络\inceptionv3_svm
net_inceptionv3=netTransfer;
inceptionv3_svm=classifier;
load C:\Users\roar沫沫\Desktop\党宁的论文\训练好的网络\resnet18
load C:\Users\roar沫沫\Desktop\党宁的论文\训练好的网络\resnet18_svm
net_resnet18=netTransfer;
resnet18_svm=classifier;
load C:\Users\roar沫沫\Desktop\党宁的论文\训练好的网络\vgg16
load C:\Users\roar沫沫\Desktop\党宁的论文\训练好的网络\vgg16_svm
net_vgg16=netTransfer;
vgg16_svm=classifier;
load C:\Users\roar沫沫\Desktop\党宁的论文\训练好的网络\vgg19
load C:\Users\roar沫沫\Desktop\党宁的论文\训练好的网络\vgg19_svm
net_vgg19=netTransfer;
vgg19_svm=classifier;
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
YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;
train_f=[];
test_f=[];
%% alexnet
net=alexnet();
inputSize = net.Layers(1).InputSize;
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);
layer = 'fc_new';
featuresTrain = activations(net_alexnet,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net_alexnet,augimdsTest,layer,'OutputAs','rows');
train_f=[train_f featuresTrain];
test_f=[test_f featuresTest];
[YPred,score_alexnet]= predict(alexnet_svm,featuresTest);
Acc_svm_alexnet=sum(YPred==YTest)/numel(YTest);
YPred_T = classify(net_alexnet,augimdsTest,'MiniBatchSize',100);
YTest_T = imdsTest.Labels;
Acc_net_alexnet = sum(YPred_T==YTest_T)/numel(YTest_T);
%% googlenet
net=googlenet();
inputSize = net.Layers(1).InputSize;
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);
layer = 'fc_new';
featuresTrain = activations(net_googlenet,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net_googlenet,augimdsTest,layer,'OutputAs','rows');
train_f=[train_f featuresTrain];
test_f=[test_f featuresTest];

[YPred,score_googlenet] = predict(googlenet_svm,featuresTest);
Acc_svm_googlenet=sum(YPred==YTest)/numel(YTest);
YPred_T = classify(net_googlenet,augimdsTest,'MiniBatchSize',100);
YTest_T = imdsTest.Labels;
Acc_net_googlenet = sum(YPred_T==YTest_T)/numel(YTest_T);
%% inceptionv3
% net=inceptionv3();
% inputSize = net.Layers(1).InputSize;
% augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
% augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
% augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);
% layer = 'fc_new';
% featuresTest = activations(net_inceptionv3,augimdsTest,layer,'OutputAs','rows');
% [YPred,score_inceptionv3] = predict(inceptionv3_svm,featuresTest);
% Acc_svm_inceptionv3=sum(YPred==YTest)/numel(YTest);
% YPred_T = classify(net_inceptionv3,augimdsTest,'MiniBatchSize',100);
% YTest_T = imdsTest.Labels;
% Acc_net_inceptionv3 = sum(YPred_T==YTest_T)/numel(YTest_T);
%% resnet18
net=resnet18();
inputSize = net.Layers(1).InputSize;
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);
layer = 'fc_new';
featuresTrain = activations(net_resnet18,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net_resnet18,augimdsTest,layer,'OutputAs','rows');
train_f=[train_f featuresTrain];
test_f=[test_f featuresTest];
[YPred,score_resnet18] = predict(resnet18_svm,featuresTest);
Acc_svm_resnet18=sum(YPred==YTest)/numel(YTest);
YPred_T = classify(net_resnet18,augimdsTest,'MiniBatchSize',100);
YTest_T = imdsTest.Labels;
Acc_net_resnet18 = sum(YPred_T==YTest_T)/numel(YTest_T);
%% vgg16
net=vgg16();
inputSize = net.Layers(1).InputSize;
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);
layer = 'fc_new';
featuresTrain = activations(net_vgg16,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net_vgg16,augimdsTest,layer,'OutputAs','rows');
train_f=[train_f featuresTrain];
test_f=[test_f featuresTest];
[YPred,score_vgg16] = predict(vgg16_svm,featuresTest);
Acc_svm_vgg16=sum(YPred==YTest)/numel(YTest);
YPred_T = classify(net_vgg16,augimdsTest,'MiniBatchSize',100);
YTest_T = imdsTest.Labels;
Acc_net_vgg16 = sum(YPred_T==YTest_T)/numel(YTest_T);
%% vgg19
net=vgg19();
inputSize = net.Layers(1).InputSize;
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);
layer = 'fc_new';
featuresTrain = activations(net_vgg19,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net_vgg19,augimdsTest,layer,'OutputAs','rows');
train_f=[train_f featuresTrain];
test_f=[test_f featuresTest];
[YPred,score_vgg19] = predict(vgg19_svm,featuresTest);
Acc_svm_vgg19=sum(YPred==YTest)/numel(YTest);
YPred_T = classify(net_vgg19,augimdsTest,'MiniBatchSize',100);
YTest_T = imdsTest.Labels;
Acc_net_vgg19 = sum(YPred_T==YTest_T)/numel(YTest_T);
%% ronghe gailv
score=[score_alexnet+score_googlenet+score_resnet18+score_vgg16+score_vgg19];
final_label=[];
for i=1:size(score,1)
    [m,n]=max(score(i,:));
    switch n
        case 1
           final_label=[final_label;'ABSZ'];
        case 2
           final_label=[final_label; 'CPSZ'];
        case 3
           final_label=[final_label; 'FNSZ'];
        case 4
           final_label=[final_label; 'GNSZ'];
        case 5
           final_label=[final_label; 'MYSZ'];
        case 6
           final_label=[final_label; 'SPSZ'];
        case 7
           final_label=[final_label; 'TCSZ'];
        case 8
           final_label=[final_label; 'TNSZ'];
        
    end
end
final_label=categorical(cellstr(final_label));
Acc_final_score=sum(final_label==YTest)/numel(YTest);

%% ronghe tezheng
classifier = fitcecoc(train_f,YTrain);
YPred_f = predict(classifier,test_f);
Acc_final_feature=sum(YPred_f==YTest)/numel(YTest);