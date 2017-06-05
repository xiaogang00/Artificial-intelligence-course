function model = svm_train(x, y, kerType)
[n, dim] = size(x);
C = 10;
%在这里需要输入的便签是二分类的，也就是需要1和-1
plable = find(y==1);
nlable = find(y==-1);
%统计其不同类别的数目
plen = length(plable);
nlen = length(nlable);

%下面将调用matlab自带的quadprog函数来解二次优化问题，
%要求我们输入的优化问题是1/2xHx^T + fx的形式
%而下面的变量Options是用来控制算法的选项参数的向量
options = optimset;   
options.LargeScale = 'off';
options.Display = 'off';
%H在这里是二次项的矩阵，
H = (y'*y).*kernel(x,x,kerType);
%f = cat(1,zeros(plen,1),-ones(nlen,1));
f=-ones(n,1);
A = [];
b = [];
%Aeq，beq是等式约束
Aeq = y;
beq = 0;
lb = zeros(n,1);
ub = C*ones(n,1);
% a0是解的初始近似值
a0 = zeros(n,1);  

%在这里解svm的优化问题
[a,fval,eXitflag,output,lambda]  = quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
%在这里求出支持向量，满足大于epsilon的则认为x为支持向量
epsilon = 1e-8;
sv_label = find(abs(a)>epsilon);  
a = a(sv_label);
Xsv = x(sv_label,:);
Ysv = y(sv_label);
svnum = length(sv_label);
%设置模型的参数，参数是我们求出来的优化的结果
model.a = a;
model.Xsv = Xsv;
model.Ysv = Ysv;
model.svnum = svnum;
num = length(Ysv);

%在这里求出权重
W = zeros(1,dim);
for i = 1:num
    W = W+ a(i,1)*Ysv(i)*Xsv(i,:);
end
model.W = W;

%在这里求出偏差b的值，方法也是按照推到出来的公式
py_label = find(Ysv==1);
pa = a(py_label);
pXsv = Xsv(py_label,:);
pYsv = Ysv(py_label);
pnum = length(py_label);
tmp = pYsv - a'.*Ysv*kernel(Xsv,pXsv,kerType);
b = mean(tmp);
model.b = b;
end


