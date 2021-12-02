function [MAS,MPSD]=MASMPSD(x1)
L=length(x1);
%%  P=abs(fft(x1)); for calculate mean
P=abs(fft(x1));
P1 = mean(P(2:4));P2=mean(P(4:6));P3=mean(P(6:9));P4=mean(P(9:11));P5=mean(P(11:14));P6=mean(P(14:17));P7=mean(P(17:19));P8=mean(P(19:22));P9=mean(P(22:25));P10=mean(P(25:32));
P11=mean(P(32:39));P12=mean(P(39:46));P13=mean(P(46:53));P14=mean(P(53:61));P15=mean(P(61:75));P16=mean(P(75:89));P17=mean(P(89:97));P18=mean(P(103:115));P19=mean(P(127:141));  
px1=[P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12,P13,P14,P15,P16,P17,P18,P19];
% 70-128HZ
%P1 = mean(P(141:146));P2=mean(P(146:151));P3=mean(P(151:156));P4=mean(P(156:161));P5=mean(P(161:166));P6=mean(P(166:171));P7=mean(P(171:176));P8=mean(P(176:181));P9=mean(P(181:186));P10=mean(P(186:191));
%P11=mean(P(191:196));P12=mean(P(196:201));P13=mean(P(201:206));P14=mean(P(206:211));P15=mean(P(211:216));P16=mean(P(216:221));P17=mean(P(221:226));P18=mean(P(226:235));P19=mean(P(247:257));
%px1=[P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12,P13,P14,P15,P16,P17,P18,P19];
MAS=px1;
%% P=10*log10(abs(fft(x1)).^2/L); Calculate the PSD of the signal
P=10*log10(abs(fft(x1)).^2/L);
% 0-70HZ
P1 = mean(P(2:4));P2=mean(P(4:6));P3=mean(P(6:9));P4=mean(P(9:11));P5=mean(P(11:14));P6=mean(P(14:17));P7=mean(P(17:19));P8=mean(P(19:22));P9=mean(P(22:25));P10=mean(P(25:32));
P11=mean(P(32:39));P12=mean(P(39:46));P13=mean(P(46:53));P14=mean(P(53:61));P15=mean(P(61:75));P16=mean(P(75:89));P17=mean(P(89:103));P18=mean(P(103:115));P19=mean(P(127:141));  px2=[P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12,P13,P14,P15,P16,P17,P18,P19];
%70-128HZ
%P1 = mean(P(141:146));P2=mean(P(146:151));P3=mean(P(151:156));P4=mean(P(156:161));P5=mean(P(161:166));P6=mean(P(166:171));P7=mean(P(171:176));P8=mean(P(176:181));P9=mean(P(181:186));P10=mean(P(186:191));
%P11=mean(P(191:196));P12=mean(P(196:201));P13=mean(P(201:206));P14=mean(P(206:211));P15=mean(P(211:216));P16=mean(P(216:221));P17=mean(P(221:226));P18=mean(P(226:235));P19=mean(P(247:257));
px2=[P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12,P13,P14,P15,P16,P17,P18,P19];
MPSD=px2;
end