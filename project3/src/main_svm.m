% main_svm
clear;
addpath('../data/');
addpath('../src/libsvm-3.21/');
%����������ѵ�����ݣ�������lbp��������
images = loadMNISTImages('../data/train-images-idx3-ubyte');
label = loadMNISTLabels('../data/train-labels-idx1-ubyte');
train_feature = feature2(images);

%����������������ݣ�������lbp��������
test_images = loadMNISTImages('../data/t10k-images-idx3-ubyte');
test_label = loadMNISTLabels('../data/t10k-labels-idx1-ubyte');
test_feature = feature2(test_images);


%���������ע�͵Ĵ�������libsvm�ķ���ѵ����׼ȷ�ʴ����75%����
% cd('../src/libsvm-3.21/matlab/');
% %����������ѵ������̫�࣬�����ܵ�����ÿ���ȷ���10000������ѵ��
% %ͳ��6��������10000������ѵ���Ľ����׼ȷ��
% for echo = 1 : 6
% %����ѵ��
% svmStruct=svmtrain(label(1+(echo-1)*10000:(echo)*10000,:),...
% train_feature(1+(echo-1)*10000:(echo)*10000,:), '-s 0 -t 2');
% %a�Ǽ�¼Ԥ���label,acc�Ǽ�¼ÿ��Ԥ���׼ȷ��
% acc = [];
% %������������ݶ���10000��������10�������
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
% %����׼ȷ����ÿ��10000��ѵ����׼ȷ�ʵ�ƽ��
% final_acc = mean(acc);


%��������Լ���д��svm�ĳ��򣬵�����������ʱ��ܳ���
%ʱ�临�Ӷ�̫��ÿ�ε�ѵ������ֻ���м��ٵ�������,�����ڻ�����Ĺ�ģ���úܴ�
%���������Ǵ��������������1vs1�ķ�����ͨ������vote����÷�����
%������train_len�����Ƿ����ѵ�����ݵ���Ŀ��һ����1000��ʱ����40%��׼ȷ��
%nClass�����ǵķ�����Ŀ
nClass = 10;
train_len = 1000;
%��ȡ1vs1�ķ��෽ʽ
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
        % �趨��������ʱ�����ǩ
        y = [y_positive, y_negative]';
        % ��ii�����͵�jj�������������ʱ�ķ������ṹ��Ϣ���洢Ϊstruct
        model = svm_train(x, y','rbf');
        Struct{ii, jj} = model;
       
        x_test = test_feature;
        y_test = test_label;
        %��������֮ǰ��ģ�ͣ������svm�Ż���������Ĳ���������a,epsilon
        a = model.a;
        epsilon=1e-8;
        i_sv=find(abs(a)>epsilon);
        %ѡȡ֧������
        b = model.b;
        tmp=(a'.*y(i_sv,:)')*kernel(x(i_sv,:),x_test,'rbf');
        %�������������Ԥ������
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

%��ÿ�����������������һ����������
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
    %��������жϵ�����������е�����
    final(i) = mode(result(i,:));
    if  final(i) == test_label(i)
        correct = correct + 1;
    else
        wrong = wrong + 1;
    end
end
%����׼ȷ��
acc = correct / (correct + wrong);
        
        




