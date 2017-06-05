function class=naive_Bayes(test_object,feature,label)
%由朴素贝叶斯的思想，我们需要首先计算其先验概率，
%也就是从训练集中找到各类的的数目s

total_class=label;
number=unique(total_class);
result=[];
%首先需要我们计算先验概率
for i=1:length(number)
    total_each{i}=find(total_class==number(i));
    result=[result, length(total_each{i})];
end

%之后我们需要计算条件概率：p(xi|c)，c是class
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

%最后需要依照前面的计算出后验概率
for i=1:length(number)
    post_probability(:,i)=t_sum(:,i)./result(i);
end

%基于独立性假设
p=prod(post_probability);
p=p.*result./size(feature,1);
[~, j]=max(p);
%找出后验概率最大的作为最后的预测结果
class=number(j);
