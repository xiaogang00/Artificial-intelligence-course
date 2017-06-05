function featurevec = extractHOGFeatures(img)
img=double(img);
step=8;      
%step*step个像素作为一个cell
[m1, n1]=size(img);
img=imresize(img,[floor(m1/step)*step,floor(n1/step)*step],'nearest');
[m, n]=size(img);

% 2、%伽马校正
img=sqrt(img);

% 3、求梯度和方向
fy=[-1 0 1];       
fx=fy';        
Iy=imfilter(img,fy,'replicate');    %竖直梯度
Ix=imfilter(img,fx,'replicate');    %水平梯度
Ied=sqrt(Ix.^2+Iy.^2);              %梯度值
Iphase=Iy./Ix;             
the=atan(Iphase)*180/3.14159; 
%求梯度角度

for i=1:m
    for j=1:n
        if(Ix(i,j)>=0 && Iy(i,j)>=0) %第一象限
            the(i,j)=the(i,j);
        elseif(Ix(i,j)<=0 && Iy(i,j)>=0) %第二象限
            the(i,j)=the(i,j)+180;
        elseif(Ix(i,j)<=0 && Iy(i,j)<=0) %第三象限
            the(i,j)=the(i,j)+180;
        elseif(Ix(i,j)>=0 && Iy(i,j)<=0) %第四象限
            the(i,j)=the(i,j)+360;
        end

        if isnan(the(i,j))==1  
            the(i,j)=0;
        end

    end
end
the=the+0.000001; 

% 4、划分cell，求cell的直方图( 1 cell = 8*8 pixel )
clear i j;

step=8;            
orient=9;               %方向直方图的方向个数
jiao=360/orient;        %每个方向包含的角度数
Cell=cell(1,1);             
ii=1;
jj=1;

for i=1:step:m
    ii=1;
    for j=1:step:n
        Hist1(1:orient)=0;
        for p=1:step
            for q=1:step
                %梯度方向直方图
                Hist1(ceil(the(i+p-1,j+q-1)/jiao))=Hist1(ceil(the(i+p-1,j+q-1)/jiao))+Ied(i+p-1,j+q-1);
            end
        end
        Cell{ii,jj}=Hist1;    
        ii=ii+1;
    end
    jj=jj+1;
end

% 5、划分block，求block的特征值,使用重叠方式( 1 block = 2*2 cell )
clear m n i j;
[m, n]=size(Cell);
feature=cell(1,(m-1)*(n-1));
for i=1:m-1
    for j=1:n-1
        block=[Cell{i,j}(:)' Cell{i,j+1}(:)' Cell{i+1,j}(:)' Cell{i+1,j+1}(:)'];
        block=block./sum(block); 
        feature{(i-1)*(n-1)+j}=block;
    end
end

% 6、图像的HOG特征值
[~, n]=size(feature);
l=2*2*orient;
featurevec=zeros(1,n*l);
for i=1:n
    featurevec((i-1)*l+1:i*l)=feature{i}(:);
end

[m,n]=find(isnan(featurevec)==1);
featurevec(m,n)=0;
featurevec( find(featurevec==0) )=[];
end