function  binary_feature = feature1(images)
[m, n] = size(images);
binary_feature = zeros(n, m);
%����������������ͼ�����0,1�Ķ�ֵ����
for i = 1:n
    for j = 1:m
        if images(j, i) > 0
            binary_feature(i, j) = 1;
        end
    end
end