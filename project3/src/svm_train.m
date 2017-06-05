function model = svm_train(x, y, kerType)
[n, dim] = size(x);
C = 10;
%��������Ҫ����ı�ǩ�Ƕ�����ģ�Ҳ������Ҫ1��-1
plable = find(y==1);
nlable = find(y==-1);
%ͳ���䲻ͬ������Ŀ
plen = length(plable);
nlen = length(nlable);

%���潫����matlab�Դ���quadprog������������Ż����⣬
%Ҫ������������Ż�������1/2xHx^T + fx����ʽ
%������ı���Options�����������㷨��ѡ�����������
options = optimset;   
options.LargeScale = 'off';
options.Display = 'off';
%H�������Ƕ�����ľ���
H = (y'*y).*kernel(x,x,kerType);
%f = cat(1,zeros(plen,1),-ones(nlen,1));
f=-ones(n,1);
A = [];
b = [];
%Aeq��beq�ǵ�ʽԼ��
Aeq = y;
beq = 0;
lb = zeros(n,1);
ub = C*ones(n,1);
% a0�ǽ�ĳ�ʼ����ֵ
a0 = zeros(n,1);  

%�������svm���Ż�����
[a,fval,eXitflag,output,lambda]  = quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
%���������֧���������������epsilon������ΪxΪ֧������
epsilon = 1e-8;
sv_label = find(abs(a)>epsilon);  
a = a(sv_label);
Xsv = x(sv_label,:);
Ysv = y(sv_label);
svnum = length(sv_label);
%����ģ�͵Ĳ�����������������������Ż��Ľ��
model.a = a;
model.Xsv = Xsv;
model.Ysv = Ysv;
model.svnum = svnum;
num = length(Ysv);

%���������Ȩ��
W = zeros(1,dim);
for i = 1:num
    W = W+ a(i,1)*Ysv(i)*Xsv(i,:);
end
model.W = W;

%���������ƫ��b��ֵ������Ҳ�ǰ����Ƶ������Ĺ�ʽ
py_label = find(Ysv==1);
pa = a(py_label);
pXsv = Xsv(py_label,:);
pYsv = Ysv(py_label);
pnum = length(py_label);
tmp = pYsv - a'.*Ysv*kernel(Xsv,pXsv,kerType);
b = mean(tmp);
model.b = b;
end


