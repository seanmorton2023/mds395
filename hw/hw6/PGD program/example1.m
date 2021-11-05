clc;
clear;
x=0:0.1:10;
y=0:0.1:10;
y1=sin(x);
y2=y;
X=[y1'*y2];
ec=0.1;
[ Model, e, eB ] = ALT_FIX_2D( X,ec );
figure(1)
F1=Model.F1;
plot(x,F1(:,1)','linewidth',2);
figure(2)
F2=Model.F2;
plot(x,F2(:,1)','linewidth',2);
xlabel('','FontSize',30);
ylabel('','FontSize',30);
