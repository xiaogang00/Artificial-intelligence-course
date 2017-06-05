function feature = feature2(images)
[m ,n] = size(images);
feature = zeros(n, 256);
%在这里使用lbp进行特征的计算，lbp的计算输出特征向量的长度为256
for i = 1:n
    length = 28;
    temp_image = zeros(length, length);
    count = 1;
    for number1 = 1:length
        for number2 = 1:length
            temp_image(number2, number1) = images(count, i);
            count = count + 1;
        end
    end
    %imshow(temp_image),查看是否是我们需要的图片
    temp_feature = lbp(temp_image);
    for number3 = 1:256
        feature(i, number3) = temp_feature(number3);
    end
end