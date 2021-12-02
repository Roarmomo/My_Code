function [trainData,testData]=splitData(x)
%[trainData,testData]
l=length(x);  
fs=250; % 采样频率
o=2; m=fix(l/fs); % m为该段信号的总得可取的样本个数

PI=[];Psingular=[]; % PI存储正常样本，Psingular存储奇异值的样本
%% 设定阈值，当特征小于这个阈值时，该样本作为奇异值的样本
Threshold=-500;
 for j=1:m-1        
%    2 second a 
     b=x(:,o*fs*(j-1)/2+1:o*fs*(j+1)/2);
     k=size(b,1);
       for ii=1:k   %% 该循环获取到23*38的矩阵
         x1=b(ii,:);
         T=1/fs;
         L=length(x1);
        %%  P=abs(fft(x1)); for calculate mean
         P=abs(fft(x1));
           
        
         P1 = mean(P(2:4));P2=mean(P(4:6));P3=mean(P(6:9));P4=mean(P(9:11));P5=mean(P(11:14));P6=mean(P(14:17));P7=mean(P(17:38));P8=mean(P(38:22));P9=mean(P(22:25));P10=mean(P(25:32));
          P11=mean(P(32:39));P12=mean(P(39:46));P13=mean(P(46:53));P14=mean(P(53:61));P15=mean(P(61:75));P16=mean(P(75:89));P17=mean(P(89:103));P18=mean(P(103:115));P38=mean(P(127:141));  
          px1=[P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12,P13,P14,P15,P16,P17,P18,P38];
           % 70-128HZ
%           P1 = mean(P(141:146));P2=mean(P(146:151));P3=mean(P(151:156));P4=mean(P(156:161));P5=mean(P(161:166));P6=mean(P(166:171));P7=mean(P(171:176));P8=mean(P(176:181));P9=mean(P(181:186));P10=mean(P(186:381));
%           P11=mean(P(381:386));P12=mean(P(386:201));P13=mean(P(201:206));P14=mean(P(206:211));P15=mean(P(211:216));P16=mean(P(216:221));P17=mean(P(221:226));P18=mean(P(226:235));P38=mean(P(247:257));
           px1=[P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12,P13,P14,P15,P16,P17,P18,P38];
           pk1(ii,:)=px1;
        %% P=10*log10(abs(fft(x1)).^2/L); Calculate the PSD of the signal
           P=10*log10(abs(fft(x1)).^2/L);
            % 0-70HZ
        
         P1 = mean(P(2:4));P2=mean(P(4:6));P3=mean(P(6:9));P4=mean(P(9:11));P5=mean(P(11:14));P6=mean(P(14:17));P7=mean(P(17:38));P8=mean(P(38:22));P9=mean(P(22:25));P10=mean(P(25:32));
          P11=mean(P(32:39));P12=mean(P(39:46));P13=mean(P(46:53));P14=mean(P(53:61));P15=mean(P(61:75));P16=mean(P(75:89));P17=mean(P(89:103));P18=mean(P(103:115));P38=mean(P(127:141));  px2=[P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12,P13,P14,P15,P16,P17,P18,P38];
          % 70-128HZ
%           P1 = mean(P(141:146));P2=mean(P(146:151));P3=mean(P(151:156));P4=mean(P(156:161));P5=mean(P(161:166));P6=mean(P(166:171));P7=mean(P(171:176));P8=mean(P(176:181));P9=mean(P(181:186));P10=mean(P(186:381));
%           P11=mean(P(381:386));P12=mean(P(386:201));P13=mean(P(201:206));P14=mean(P(206:211));P15=mean(P(211:216));P16=mean(P(216:221));P17=mean(P(221:226));P18=mean(P(226:235));P38=mean(P(247:257));
           px2=[P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12,P13,P14,P15,P16,P17,P18,P38];
          pk2(ii,:)=px2;
       end
            %pk=pk2;
        pk=[pk1];%,pk2];
        pk=reshape(pk',20*19,1);
        if find(pk<Threshold)
           Psingular=[Psingular,pk];
        else 
            PI=[PI,pk];
        end
 end   
 
%     PI = mapminmax( PI, 0, 1);
    ll=size( PI,2); % 样本的总个数
    ratio=fix(0.9*ll);%训练样本的个数
    Index1=randperm(ll,ratio);
    z=1:ll; % 总得样本集的下标
    % 取随机选取的补集作为测试集的下标
    Index2=setdiff(z,Index1);
    % Index1为一个行向量，其值为ll个样本中的随机0.8*ll个样本，Index2为测试集的下标
    len1=length(Index1);
    len2=length(Index2);
    PtrainI=[];
    PtestI=[];
    % 得到随机选取后的训练样本数
    for index=1:len1
        PtrainI(:,index)=PI(:,Index1(index));
    end
    % 得到随机选取后的测试样本数
    for index=1:len2
        PtestI(:,index)=PI(:,Index2(index));
    end
trainData=PtrainI;
testData=PtestI;