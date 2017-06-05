% main_svm
clear;
addpath('../data/');
addpath('../src/libsvm-3.21/');
%在这里载入训练数据，并且用lbp计算特征
images = loadMNISTImages('../data/train-images-idx3-ubyte');
label = loadMNISTLabels('../data/train-labels-idx1-ubyte');
train_feature = feature2(images);

%在这里载入测试数据，并且用lbp计算特征
test_images = loadMNISTImages('../data/t10k-images-idx3-ubyte');
test_label = loadMNISTLabels('../data/t10k-labels-idx1-ubyte');
test_feature = feature2(test_images);


%而下面这段注释的代码是用libsvm的方法训练，准确率大概在75%左右
% cd('../src/libsvm-3.21/matlab/');
% %在这里由于训练数据太多，程序跑得慢，每次先放入10000个进行训练
% %统计6组这样的10000个样本训练的结果的准确率
% for echo = 1 : 6
% %进行训练
% svmStruct=svmtrain(label(1+(echo-1)*10000:(echo)*10000,:),...
% train_feature(1+(echo-1)*10000:(echo)*10000,:), '-s 0 -t 2');
% %a是记录预测的label,acc是记录每次预测的准确率
% acc = [];
% %在这里测试数据都是10000个，都分10组来完成
% correct = 0;
% wrong = 0;
% for i = 1 : 10
%     [predict_label, ~, dec_values] = svmpredict(test_label(1+(i-1)*1000: i*1000,:),...
%          test_feature(1+(i-1)*1000: i*1000,:), svmStruct); 
%     temp_label = test_label(1+(i-1)*1000: i*1000,:);
%     for number = 1:length(predict_label)
%         if predict_label(number) == temp_label(number)
%             correct = correct + 1;
%         else
%             wrong = wrong + 1;
%         end
%     end
% end
% accuracy = correct/(correct +wrong);
% acc= [acc, accuracy];
% end
% %最后的准确率是每次10000条训练的准确率的平均
% final_acc = mean(acc);


%下面的是自己编写的svm的程序，但是由于运行时间很长，
%时间复杂度太大，每次的训练数据只能有几百的数据量,而且内积矩阵的规模会变得很大，
%在这里我们处理多分类的问题是1vs1的方法，通过最后的vote来获得分类结果
%在这里train_len是我们放入的训练数据的数目，一般在1000的时候，有40%的准确率
%nClass是我们的分类数目
nClass = 10;
train_len = 1000;
%采取1vs1的分类方式
for ii=1:(nClass-1)
    for jj=(ii+1):(nClass)
        clear X;
        clear Y;
        temp1 =find(label==(ii-1));
        if length(temp1) > train_len 
            x_positive = train_feature(temp1(1:train_len ),:);
            y_positive = ones(1,train_len);
        else
            x_positive = train_feature(temp1,:);
            y_positive = ones(1,length(x_positive));
        end
        
        temp2 =find(label == (jj-1));
        if length(temp2) > train_len 
            x_negative = train_feature(temp2(1:train_len ),:);
            y_negative = -ones(1,train_len);
        else
            x_negative = train_feature(temp2,:);
            y_negative = -ones(1,length(x_negative));
        end
        
        x = [x_positive ;x_negative];
        % 设定两两分类时的类标签
        y = [y_positive, y_negative]';
        % 第ii个类别和第jj个类别两两分类时的分类器结构信息，存储为struct
        model = svm_train(x, y','rbf');
        Struct{ii, jj} = model;
       
        x_test = test_feature;
        y_test = test_label;
        %根据我们之前的模型，计算出svm优化方程里面的参数，包括a,epsilon
        a = model.a;
        epsilon=1e-8;
        i_sv=find(abs(a)>epsilon);
        %选取支持向量
        b = model.b;
        tmp=(a'.*y(i_sv,:)')*kernel(x(i_sv,:),x_test,'rbf');
        %在这里计算我们预测的输出
        yd=sign(tmp+b);
        for i = 1:length(yd);
            if yd(i) == 1
                yd(i) = ii-1;
            else
                yd(i) = jj-1;
            end
        end
        predict{ii, jj} = yd;
     end
end

%将每个分类器的输出放在一个向量里面
result = predict{1, 2}';
for ii=1:(nClass-1)
    for jj=(ii+1):(nClass)
        if (ii ~= 1) || (jj ~= 2)
            result = cat(2, result, predict{ii, jj}');
        end
    end
end

final = zeros(length(result),1);
correct = 0;
wrong = 0;
for i = 1:length(result)
    %计算最后判断的输出，是其中的众数
    final(i) = mode(result(i,:));
    if  final(i) == test_label(i)
        correct = correct + 1;
    else
        wrong = wrong + 1;
    end
end
%计算准确率
acc = correct / (correct + wrong);
        
        




