clear;
clc;
imdsTrain = imageDatastore('C:\Users\roar沫沫\Desktop\党宁的论文\特征图片\MAS\训练\FNSZ\', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); %文件夹'MerchData'中的所有子文件夹的内容，每个文件夹即为一类 

% imdsTest= imageDatastore('D:\首期尝试\4、归一化后\003\nm-02\', ...
%     'IncludeSubfolders',true, ...
%     'LabelSource','foldernames'); %文件夹'MerchData'中的所有子文件夹的内容，每个文件夹即为一类 

net=googlenet;

% net=load('D:\保存训练好的模型\aaaaaa_googlenet.mat');

inputSize = net.Layers(1).InputSize;            %输入层为224*224*3


augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);       %要在不执行进一步数据增强的情况下自动调整验证图像的大小，请使用增强的图像数据存储，而不指定任何其他预处理操作
% augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);



layer = 'pool5-7x7_s1';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
% featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');

% idx = [242 245 250 255];
% figure
% for i = 1:numel(idx)
%     subplot(2,2,i)
%     I = readimage(layer,idx(i));
%     label = YPred(idx(i));
%     imshow(I)
%     title(char(label))
% end
% m=featuresTrain(1,1:49);
% matrix=reshape(featuresTrain,64,64);
% imwrite(uint8(matrix), 'D:\jiujintian\matrix.tif');
FNSZ=featuresTrain;
save('C:\Users\roar沫沫\Desktop\党宁的论文\迁移学习提取的特征\googlenet\FNSZ.mat','FNSZ');