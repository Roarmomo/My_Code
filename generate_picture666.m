index=1;
for i=1:64:length(I)
    for ii=1:64:length(I)
        index=index+1;
        II=I(i:i+63,ii:ii+63,:);
        II=gather(II);
        II=imresize(II,[343 434 ]);
        str='C:\Users\roar沫沫\Desktop\党宁的论文\生成图片\MAS\TNSZ';
        path=[str,'\',num2str(index),'.png'];
        imwrite(II,path);
    end
end