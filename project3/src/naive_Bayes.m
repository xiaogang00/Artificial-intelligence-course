function class=naive_Bayes(test_object,feature,label)
%�����ر�Ҷ˹��˼�룬������Ҫ���ȼ�����������ʣ�
%Ҳ���Ǵ�ѵ�������ҵ�����ĵ���Ŀs

total_class=label;
number=unique(total_class);
result=[];
%������Ҫ���Ǽ����������
for i=1:length(number)
    total_each{i}=find(total_class==number(i));
    result=[result, length(total_each{i})];
end

%֮��������Ҫ�����������ʣ�p(xi|c)��c��class
t_sum=[];
for i=1:length(number)
    for k=1:length(test_object)
        s1=0;
        for j=1:length(total_each{i})
            if test_object(k)==feature(total_each{i}(j), k)
                s1=s1+1;
            end
        end
        t_sum(k,i)=s1;
    end
end

%�����Ҫ����ǰ��ļ�����������
for i=1:length(number)
    post_probability(:,i)=t_sum(:,i)./result(i);
end

%���ڶ����Լ���
p=prod(post_probability);
p=p.*result./size(feature,1);
[~, j]=max(p);
%�ҳ��������������Ϊ����Ԥ����
class=number(j);
