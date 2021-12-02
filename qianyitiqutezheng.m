clear;
clc;
imdsTrain = imageDatastore('C:\Users\roarĭĭ\Desktop\����������\����ͼƬ\MAS\ѵ��\FNSZ\', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); %�ļ���'MerchData'�е��������ļ��е����ݣ�ÿ���ļ��м�Ϊһ�� 

% imdsTest= imageDatastore('D:\���ڳ���\4����һ����\003\nm-02\', ...
%     'IncludeSubfolders',true, ...
%     'LabelSource','foldernames'); %�ļ���'MerchData'�е��������ļ��е����ݣ�ÿ���ļ��м�Ϊһ�� 

net=googlenet;

% net=load('D:\����ѵ���õ�ģ��\aaaaaa_googlenet.mat');

inputSize = net.Layers(1).InputSize;            %�����Ϊ224*224*3


augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);       %Ҫ�ڲ�ִ�н�һ��������ǿ��������Զ�������֤ͼ��Ĵ�С����ʹ����ǿ��ͼ�����ݴ洢������ָ���κ�����Ԥ�������
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
save('C:\Users\roarĭĭ\Desktop\����������\Ǩ��ѧϰ��ȡ������\googlenet\FNSZ.mat','FNSZ');