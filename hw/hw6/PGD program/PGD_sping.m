clc
clear;
load('data.mat');

[M,N] = size(data);
mn =  mean(data,1);
data = data - repmat(mn,M,1);
ec=0.1;

[ Model, e, eB] = HOPGD( data,ec );
A = Model.F1*Model.F2';
R = norm(data-A)/norm(data);
plot(data(:,1),data(:,2),"r.",'MarkerSize',10);

hold on
plot(A(:,1),A(:,2),'k','linewidth',2)
xlabel('xa(t)')
ylabel('ya(t)')
legend('Mean-subtracted data','PGD data','location','northwest')
backcolor=[1,1,1];
set(gca, 'color', backcolor)
axis([-50 50 -100 100]);

