clear;
clc;
load C:\Users\roar沫沫\Desktop\党宁的论文\数据集\montage数据八类\ABSZ
load C:\Users\roar沫沫\Desktop\党宁的论文\数据集\montage数据八类\CPSZ
load C:\Users\roar沫沫\Desktop\党宁的论文\数据集\montage数据八类\FNSZ
load C:\Users\roar沫沫\Desktop\党宁的论文\数据集\montage数据八类\GNSZ
load C:\Users\roar沫沫\Desktop\党宁的论文\数据集\montage数据八类\MYSZ
load C:\Users\roar沫沫\Desktop\党宁的论文\数据集\montage数据八类\SPSZ
load C:\Users\roar沫沫\Desktop\党宁的论文\数据集\montage数据八类\TCSZ
load C:\Users\roar沫沫\Desktop\党宁的论文\数据集\montage数据八类\TNSZ
save_path_train='C:\Users\roar沫沫\Desktop\党宁的论文\特征图片\MASMPSD\训练\';
save_path_test='C:\Users\roar沫沫\Desktop\党宁的论文\特征图片\MASMPSD\测试\';
 x=ABSZ;
[MI_train,MI_test]=splitData(x);
num1 = size(MI_train,2);
x=CPSZ;
[MII_train,MII_test]=splitData(x);
num2 = size(MII_train,2);
x=FNSZ;
[MIII_train,MIII_test]=splitData(x);
num3 = size(MIII_train,2);
x=GNSZ;
[MIV_train,MIV_test]=splitData(x);
num4 = size(MIV_train,2);
x=MYSZ;
[MV_train,MV_test]=splitData(x);
num5 = size(MV_train,2);
x=SPSZ;
[MVI_train,MVI_test]=splitData(x);
num6 = size(MVI_train,2);
x=TCSZ;
[MVII_train,MVII_test]=splitData(x);
num7 = size(MVII_train,2);
x=TNSZ;
[MVIII_train,MVIII_test]=splitData(x);
num8 = size(MVIII_train,2);
train=[MI_train MII_train MIII_train MIV_train MV_train MVI_train MVII_train MVIII_train];
test=[MI_test MII_test MIII_test MIV_test MV_test MVI_test MVII_test MVIII_test];
Data=[train test];
[~,PS]=mapminmax(Data);
MI_train= mapminmax('apply',MI_train,PS);
MII_train= mapminmax('apply',MII_train,PS);
MIII_train= mapminmax('apply',MIII_train,PS);
MIV_train= mapminmax('apply',MIV_train,PS);
MV_train= mapminmax('apply',MV_train,PS);
MVI_train= mapminmax('apply',MVI_train,PS);
MVII_train= mapminmax('apply',MVII_train,PS);
MVIII_train= mapminmax('apply',MVIII_train,PS);
MI_test= mapminmax('apply',MI_test,PS);
MII_test= mapminmax('apply',MII_test,PS);
MIII_test= mapminmax('apply',MIII_test,PS);
MIV_test= mapminmax('apply',MIV_test,PS);
MV_test= mapminmax('apply',MV_test,PS);
MVI_test= mapminmax('apply',MVI_test,PS);
MVII_test= mapminmax('apply',MVII_test,PS);
MVIII_test= mapminmax('apply',MVIII_test,PS);

for i = 1 : num1
  A = MI_train(:,i);
  A = reshape(A,20,38);
  sc=strcat(save_path_train,'\ABSZ\',sprintf('%d',i));  %命名保存
  Sc=strcat(sc,'.jpg');
  imagesc(A)
        axis off;
        s = getframe(gcf);
        xd = s.cdata(32:374, 74:507,:);
  imwrite(xd,Sc);
end
num1 = size(MI_test,2);
for i = 1 : num1
  A = MI_test(:,i);
  A = reshape(A,20,38);
  sc=strcat(save_path_test,'\ABSZ\',sprintf('%d',i));  %命名保存
   Sc=strcat(sc,'.jpg');
  imagesc(A)
        axis off;
        s = getframe(gcf);
        xd = s.cdata(32:374, 74:507,:);
  imwrite(xd,Sc);
end




for i = 1 : num2
  B = MII_train(:,i);
  B = reshape(B,20,38);
  sc=strcat(save_path_train,'\CPSZ\',sprintf('%d',i));  %命名保存
  Sc=strcat(sc,'.jpg');
  imagesc(B)
        axis off;
        s = getframe(gcf);
        xd = s.cdata(32:374, 74:507,:);
  imwrite(xd,Sc);
end
num2 = size(MII_test,2);
for i = 1 : num2
  B = MII_test(:,i);
  B = reshape(B,20,38);
  sc=strcat(save_path_test,'\CPSZ\',sprintf('%d',i));  %命名保存
  Sc=strcat(sc,'.jpg');
  imagesc(B)
        axis off;
        s = getframe(gcf);
        xd = s.cdata(32:374, 74:507,:);
  imwrite(xd,Sc);
end


for i = 1 : num3
  C = MIII_train(:,i);
  C = reshape(C,20,38);
  sc=strcat(save_path_train,'\FNSZ\',sprintf('%d',i));  %命名保存
  Sc=strcat(sc,'.jpg');
  imagesc(C)
        axis off;
        s = getframe(gcf);
        xd = s.cdata(32:374, 74:507,:);
  imwrite(xd,Sc);
end
num3 = size(MIII_test,2);
for i = 1 : num3
  C = MIII_test(:,i);
  C = reshape(C,20,38);
  sc=strcat(save_path_test,'\FNSZ\',sprintf('%d',i));  %命名保存
  Sc=strcat(sc,'.jpg');
   imagesc(C)
        axis off;
        s = getframe(gcf);
        xd = s.cdata(32:374, 74:507,:);
  imwrite(xd,Sc);
end


for i = 1 : num4
  D = MIV_train(:,i);
  D = reshape(D,20,38);
  sc=strcat(save_path_train,'\GNSZ\',sprintf('%d',i));  %命名保存
  Sc=strcat(sc,'.jpg');
 imagesc(D)
        axis off;
        s = getframe(gcf);
        xd = s.cdata(32:374, 74:507,:);
  imwrite(xd,Sc);
end
num4 = size(MIV_test,2);
for i = 1 : num4
  D = MIV_test(:,i);
  D = reshape(D,20,38);
  sc=strcat(save_path_test,'\GNSZ\',sprintf('%d',i));  %命名保存
  Sc=strcat(sc,'.jpg');
  imagesc(D)
        axis off;
        s = getframe(gcf);
        xd = s.cdata(32:374, 74:507,:);
  imwrite(xd,Sc);
end



for i = 1 : num5
  E = MV_train(:,i);
  E = reshape(E,20,38);
  sc=strcat(save_path_train,'\MYSZ\',sprintf('%d',i));  %命名保存
  Sc=strcat(sc,'.jpg');
   imagesc(E)
        axis off;
        s = getframe(gcf);
        xd = s.cdata(32:374, 74:507,:);
  imwrite(xd,Sc);
end
num5 = size(MV_test,2);
for i = 1 : num5
  E = MV_test(:,i);
  E = reshape(E,20,38);
  sc=strcat(save_path_test,'\MYSZ\',sprintf('%d',i));  %命名保存
  Sc=strcat(sc,'.jpg');
   imagesc(E)
        axis off;
        s = getframe(gcf);
        xd = s.cdata(32:374, 74:507,:);
  imwrite(xd,Sc);
end


for i = 1 : num6
  F = MVI_train(:,i);
  F = reshape(F,20,38);
  sc=strcat(save_path_train,'\SPSZ\',sprintf('%d',i));  %命名保存
  Sc=strcat(sc,'.jpg');
   imagesc(F)
        axis off;
        s = getframe(gcf);
        xd = s.cdata(32:374, 74:507,:);
  imwrite(xd,Sc);
end
num6 = size(MVI_test,2);
for i = 1 : num6
  F = MVI_test(:,i);
  F = reshape(F,20,38);
  sc=strcat(save_path_test,'\SPSZ\',sprintf('%d',i));  %命名保存
  Sc=strcat(sc,'.jpg');
   imagesc(F)
        axis off;
        s = getframe(gcf);
        xd = s.cdata(32:374, 74:507,:);
  imwrite(xd,Sc);
end

for i = 1 : num7
  G = MVII_train(:,i);
  G = reshape(G,20,38);
  sc=strcat(save_path_train,'\TCSZ\',sprintf('%d',i));  %命名保存
  Sc=strcat(sc,'.jpg');
   imagesc(G)
        axis off;
        s = getframe(gcf);
        xd = s.cdata(32:374, 74:507,:);
  imwrite(xd,Sc);
end
num7 = size(MVII_test,2);
for i = 1 : num7
  G = MVII_test(:,i);
  G = reshape(G,20,38);
  sc=strcat(save_path_test,'\TCSZ\',sprintf('%d',i));  %命名保存
  Sc=strcat(sc,'.jpg');
  imagesc(G)
        axis off;
        s = getframe(gcf);
        xd = s.cdata(32:374, 74:507,:);
  imwrite(xd,Sc);
end



for i = 1 : num8
  H = MVIII_train(:,i);
  H = reshape(H,20,38);
  sc=strcat(save_path_train,'\TNSZ\',sprintf('%d',i));  %命名保存
  Sc=strcat(sc,'.jpg');
  imagesc(H)
        axis off;
        s = getframe(gcf);
        xd = s.cdata(32:374, 74:507,:);
  imwrite(xd,Sc);
end
num8 = size(MVIII_test,2);
for i = 1 : num8
  H = MVIII_test(:,i);
  H = reshape(H,20,38);
  sc=strcat(save_path_test,'\TNSZ\',sprintf('%d',i));  %命名保存
  Sc=strcat(sc,'.jpg');
 imagesc(H)
        axis off;
        s = getframe(gcf);
        xd = s.cdata(32:374, 74:507,:);
  imwrite(xd,Sc);
end

