function K = kernel(X,Y,type)
%这个函数是在svm中我们需要用到的核函数，他可以用来进行最后的预测
%在这里定义了线性核函数和径向基核函数两种
switch type
    case 'linear'
        K = X*Y';
    case 'rbf'
        gamma = 5;
        gamma = gamma*gamma;
        XX = sum(X.*X,2);
        YY = sum(Y.*Y,2);
        XY = X*Y';
        K = abs(repmat(XX,[1 size(YY,1)]) + repmat(YY',[size(XX,1) 1]) - 2*XY);
        K = exp(-K./gamma);
end
end