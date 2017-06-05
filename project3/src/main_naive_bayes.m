% main_naive_bayes
clear;
addpath('../data/');
images = loadMNISTImages('../data/train-images-idx3-ubyte');
label = loadMNISTLabels('../data/train-labels-idx1-ubyte');
%首先载入训练数据，并且根据其二值图本身作为特征
binary_feature = feature1(images);

%载入测试的数据，并且按照相同的方法计算其特征
test_images = loadMNISTImages('../data/t10k-images-idx3-ubyte');
test_label = loadMNISTLabels('../data/t10k-labels-idx1-ubyte');
test_feature = feature1(test_images);
acc= [];


%在这里由于训练数据太多，程序跑得慢，每次先放入1000个进行训练
%统计60组这样的1000个样本训练的结果的准确率
for train_echo = 1 : 60
    fprintf('\n training echo is : %d \n', train_echo); 
    correct = 0;
    wrong = 0;
    for test_echo = 1 : 10000
        %对每一组测试的数据调用朴素贝叶斯的方法进行训练
            class = naive_Bayes(test_feature(test_echo, :),...
                binary_feature(1+(train_echo-1)*1000:train_echo*1000,:),...
                label(1+(train_echo-1)*1000:train_echo*1000,:));
            
            if class == test_label(test_echo)
                correct = correct + 1;
            else
                wrong = wrong + 1;
            end
    end
    %计算准确率
    accuracy = correct/(correct +wrong);
    acc = [acc, accuracy];
end
final_acc = mean(acc);
fprintf('\n The Rate of correct class in training data : %2.2f \n', final_acc); 
