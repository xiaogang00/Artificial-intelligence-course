% main_naive_bayes
clear;
addpath('../data/');
images = loadMNISTImages('../data/train-images-idx3-ubyte');
label = loadMNISTLabels('../data/train-labels-idx1-ubyte');
%��������ѵ�����ݣ����Ҹ������ֵͼ������Ϊ����
binary_feature = feature1(images);

%������Ե����ݣ����Ұ�����ͬ�ķ�������������
test_images = loadMNISTImages('../data/t10k-images-idx3-ubyte');
test_label = loadMNISTLabels('../data/t10k-labels-idx1-ubyte');
test_feature = feature1(test_images);
acc= [];


%����������ѵ������̫�࣬�����ܵ�����ÿ���ȷ���1000������ѵ��
%ͳ��60��������1000������ѵ���Ľ����׼ȷ��
for train_echo = 1 : 60
    fprintf('\n training echo is : %d \n', train_echo); 
    correct = 0;
    wrong = 0;
    for test_echo = 1 : 10000
        %��ÿһ����Ե����ݵ������ر�Ҷ˹�ķ�������ѵ��
            class = naive_Bayes(test_feature(test_echo, :),...
                binary_feature(1+(train_echo-1)*1000:train_echo*1000,:),...
                label(1+(train_echo-1)*1000:train_echo*1000,:));
            
            if class == test_label(test_echo)
                correct = correct + 1;
            else
                wrong = wrong + 1;
            end
    end
    %����׼ȷ��
    accuracy = correct/(correct +wrong);
    acc = [acc, accuracy];
end
final_acc = mean(acc);
fprintf('\n The Rate of correct class in training data : %2.2f \n', final_acc); 
