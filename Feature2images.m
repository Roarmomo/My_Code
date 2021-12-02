clc
clear
%%load BECT_NORMAL训练
tic
file='D:\My_experience\去除伪迹的特征\小波熵\BECT_Normal训练';
list=dir(file);
list=list(3:end);
BECT_Normal_train=[];
fprintf('loading BECT_Normal_train\n')
for i=1:length(list)
    file2=[list(i).folder,'\',list(i).name];
    load(file2);
    BECT_Normal_train=[BECT_Normal_train Data];
   
end
fprintf('complete loading BECT_Normal_train\n')
toc
%%load BECT_NORMAL测试
tic
file='D:\My_experience\提取的特征\new\小波熵\BECT_Normal测试';
list=dir(file);
list=list(3:end);
BECT_Normal_test=[];
fprintf('loading BECT_Normal_test\n')
for i=1:length(list)
    file2=[list(i).folder,'\',list(i).name];
    load(file2);
    BECT_Normal_test=[BECT_Normal_test Data];
   
end
fprintf('complete loading BECT_Normal_test\n')
toc
%%load BECT训练
tic
file='D:\My_experience\去除伪迹的特征\小波熵\BECT训练';
list=dir(file);
list=list(3:end);
BECT_train=[];
fprintf('loading BECT_train\n')
for i=1:length(list)
    file2=[list(i).folder,'\',list(i).name];
    load(file2);
    BECT_train=[BECT_train Data];
   
end
fprintf('complete loading BECT_train\n')
toc
%%load BECT测试
tic
file='D:\My_experience\提取的特征\new\小波熵\BECT测试';
list=dir(file);
list=list(3:end);
BECT_test=[];
fprintf('loading BECT_test\n')
for i=1:length(list)
    file2=[list(i).folder,'\',list(i).name];
    load(file2);
    BECT_test=[BECT_test Data];
   
end
fprintf('complete loading BECT_test\n')
toc
%%load WEST_Normal训练
tic
file='D:\My_experience\去除伪迹的特征\小波熵\WEST_Normal训练';
list=dir(file);
list=list(3:end);
WEST_Normal_train=[];
fprintf('loading WEST_Normal_train\n')
for i=1:length(list)
    file2=[list(i).folder,'\',list(i).name];
    load(file2);
    WEST_Normal_train=[WEST_Normal_train Data];
   
end
fprintf('complete loading WEST_Normal_train\n')
toc
%%load WEST_Normal测试
tic
file='D:\My_experience\提取的特征\new\小波熵\WEST_Normal测试';
list=dir(file);
list=list(3:end);
WEST_Normal_test=[];
fprintf('loading WEST_Normal_test\n')
for i=1:length(list)
    file2=[list(i).folder,'\',list(i).name];
    load(file2);
    WEST_Normal_test=[WEST_Normal_test Data];
   
end
fprintf('complete loading WEST_Normal_test\n')
toc
%%load WEST训练
tic
file='D:\My_experience\去除伪迹的特征\小波熵\WEST训练';
list=dir(file);
list=list(3:end);
WEST_train=[];
fprintf('loading WEST_train\n')
for i=1:length(list)
    file2=[list(i).folder,'\',list(i).name];
    load(file2);
    WEST_train=[WEST_train Data];
   
end
fprintf('complete loading WEST_train\n')
toc
%%load WEST测试
tic
file='D:\My_experience\提取的特征\new\小波熵\WEST测试';
list=dir(file);
list=list(3:end);
WEST_test=[];
fprintf('loading WEST_test\n')
for i=1:length(list)
    file2=[list(i).folder,'\',list(i).name];
    load(file2);
    WEST_test=[WEST_test Data];
   
end
fprintf('complete loading WEST_test\n')
toc
clearvars -except BECT_Normal_train BECT_Normal_test BECT_train BECT_test WEST_Normal_train WEST_Normal_test WEST_train WEST_test;
train1=BECT_Normal_train;
train2=BECT_train;
train3=WEST_Normal_train;
train4=WEST_train;
test1=BECT_Normal_test;
test2=BECT_test;
test3=WEST_Normal_test;
test4=WEST_test;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
totalDD1=cat(2,train1,train2,train3,train4);
totalDD2=cat(2,test1,test2,test3,test4);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Bect_normal=train1;
lenBect_normal=size(Bect_normal,2);
Bect=train2;
lenBect=size(Bect,2);
West_normal=train3;
lenWest_normal=size(West_normal,2);
West=train4;
lenWest=size(West,2);
lentotal=lenBect_normal+lenBect+lenWest_normal+lenWest;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[NORMALDATA_train,PS]=mapminmax(totalDD1, 0,255);
NORMALDATA_test=mapminmax('apply',totalDD2, PS);
Bect_normal=NORMALDATA_train(:,1:lenBect_normal);
Bect=NORMALDATA_train(:,lenBect_normal+1:lenBect_normal+lenBect);
West_normal=NORMALDATA_train(:,lenBect_normal+lenBect+1:lenBect_normal+lenBect+lenWest_normal);
West=NORMALDATA_train(:,lenBect_normal+lenBect+lenWest_normal+1:lenBect_normal+lenBect+lenWest_normal+lenWest);
Bect_normal_test=NORMALDATA_test(:,1:size(test1,2));
Bect_test=NORMALDATA_test(:,size(test1,2)+1:size(test1,2)+size(test2,2));
West_normal_test=NORMALDATA_test(:,size(test1,2)+size(test2,2)+1:size(test1,2)+size(test2,2)+size(test3,2));
West_test=NORMALDATA_test(:,size(test1,2)+size(test2,2)+size(test3,2)+1:size(test1,2)+size(test2,2)+size(test3,2)+size(test4,2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Bect_normal=reshape(Bect_normal,1,21,1,size(Bect_normal,2));
BECT_normal={};
for i=1:size(Bect_normal,4)
  BECT_normal{i,1}=Bect_normal(:,:,:,i);
end
%%%%%%
Bect=reshape(Bect,1,21,1,size(Bect,2));
BECT={};
for i=1:size(Bect,4)
  BECT{i,1}=Bect(:,:,:,i);
end
%%%%%%
West_normal=reshape(West_normal,1,21,1,size(West_normal,2));
WEST_normal={};
for i=1:size(West_normal,4)
  WEST_normal{i,1}=West_normal(:,:,:,i);
end
%%%%%%
West=reshape(West,1,21,1,size(West,2));
WEST={};
for i=1:size(West,4)
  WEST{i,1}=West(:,:,:,i);
end
%%%%%%
Bect_normal_test=reshape(Bect_normal_test,1,21,1,size(Bect_normal_test,2));
BECT_normal_test={};
for i=1:size(Bect_normal_test,4)
  BECT_normal_test{i,1}=Bect_normal_test(:,:,:,i);
end
%%%%%%
Bect_test=reshape(Bect_test,1,21,1,size(Bect_test,2));
BECT_test={};
for i=1:size(Bect_test,4)
  BECT_test{i,1}=Bect_test(:,:,:,i);
end
%%%%%%
West_normal_test=reshape(West_normal_test,1,21,1,size(West_normal_test,2));
WEST_normal_test={};
for i=1:size(West_normal_test,4)
  WEST_normal_test{i,1}=West_normal_test(:,:,:,i);
end
%%%%%%
West_test=reshape(West_test,1,21,1,size(West_test,2));
WEST_test={};
for i=1:size(West_test,4)
  WEST_test{i,1}=West_test(:,:,:,i);
end
% save('D:\My_experience\归一化特征\BECT_normal.mat','BECT_normal');
% save('D:\My_experience\归一化特征\BECT.mat','BECT');
% save('D:\My_experience\归一化特征\WEST_normal.mat','WEST_normal');
% save('D:\My_experience\归一化特征\WEST.mat','WEST');
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% save('D:\My_experience\归一化特征\BECT_normal_test.mat','BECT_normal_test');
% save('D:\My_experience\归一化特征\BECT_test.mat','BECT_test');
% save('D:\My_experience\归一化特征\WEST_normal_test.mat','WEST_normal_test');
% save('D:\My_experience\归一化特征\WEST_test.mat','WEST_test');
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k=0;
for j=1:length(WEST_test)
        AA=WEST_test{j};
        imagesc(AA)
        axis off;
        s = getframe(gcf);
        xd = s.cdata(32:374, 74:507,:);
        k=k+1
        imwrite(xd,strcat(strcat('D:\My_experience\新特征图片\小波熵\测试（含伪迹）\WEST\WEST_test_',num2str(k)),'.png')); %保存 未归一化(提前归一化了)后的特征图片
           
end