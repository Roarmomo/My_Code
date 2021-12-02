function  Shang=xiaoboshang(sig)
E=[];
fen=wpt251(sig);
    m2=size(fen,1);    
      for j=1:m2
          E(j)=sum(abs(fen(j,:).^2));% 能量求和
      end 
      E1 = sum(E);
      dim = length(E);
      for j= 1:dim
      p(j)= E(j)/E1;
      end
      Shang= -sum(p.*log(p)); % 熵的定义式
      
   
end
function sign=wpt251(sig)
%%%%%小波包分解返回15个频段的系数
    wpt=wpdec(sig,7,'db4','shannon');
    sign(1,:)=wprcoef(wpt,127);
    sign(2,:)=wprcoef(wpt,128);
    sign(3,:)=wprcoef(wpt,130);
    sign(4,:)=wprcoef(wpt,129);
    sign(5,:)=wprcoef(wpt,133);
    sign(6,:)=wprcoef(wpt,134);
    sign(7,:)=wprcoef(wpt,132);
    sign(8,:)=wprcoef(wpt,131);
    sign(9,:)=wprcoef(wpt,139);
    sign(10,:)=wprcoef(wpt,140);
    sign(11,:)=wprcoef(wpt,142);
    sign(12,:)=wprcoef(wpt,141);
    sign(13,:)=wprcoef(wpt,137);
    sign(14,:)=wprcoef(wpt,138);
    sign(15,:)=wprcoef(wpt,136);
    sign(16,:)=wprcoef(wpt,135);
    sign(17,:)=wprcoef(wpt,151);
    sign(18,:)=wprcoef(wpt,152);
    sign(19,:)=wprcoef(wpt,154);
    sign(20,:)=wprcoef(wpt,153);
end