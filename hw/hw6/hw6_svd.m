%{
MECH ENG 395
Sean Morton
Homework 6
SVD and PCA
%}
clear;
clc;

%load in the 6-dimensional data for the spring-mass-damper system
multidata_table = readtable('multidata2.csv');
multidata = table2array(multidata_table);

% disp(multidata);
B = multidata(:, 2:7);
time_data = multidata(:, 1);

% disp(posn_data);
% disp(time_data);

%perform svd on data and see what the output of the diagonal matrix is
[U, S, Vt] = svd(B);
% disp(S(1:size(S,2),:));


%now that we have the singular values of the array we know the first 3
%variables are most important, and really only the 1st one based on values

%simplify the data by removing a certain number of variables. n is #
%of variables remaining
n = 1;

U_new = U(:, 1:n);
S_new = S(1:n, 1:n);
Vt_new = Vt(1:n, :);

A_rom = U_new * S_new * Vt_new;

% %find the means of columns so as to have the mean-adjusted data
% means_array = mean(B, 1);
% B_ma = B - means_array;
% 
% 
% %find the covariance matrix between each vector to find connections
% n_samples = length(B);
% disp(n_samples);
% cov = 1/(n_samples) * (B_ma'*B_ma);


%display the first value in the A matrix to see if our singular values show
%that we can represent the SMD system with just first column
% fprintf("\nOriginal data:\n");
% disp(B);
% 
% fprintf("\nFirst column vector of A:\n");
% disp(A_rom(:,1));
% 
% figure(1);
A_vec1 = A_rom(:,1);
% plot(time_data, A_vec1);
% 
% figure(2);
% A_vec2 = B(:,1);
% plot(time_data, A_vec2);


%plot the original x vs. y relationships so as to visualize
% figure(1);
% plot(time_data, B(:,1));
% figure(2);
% plot(time_data, B(:,2));
% figure(3);
% plot(time_data, B(:,3));
% figure(4);
% plot(time_data, B(:,4));
% figure(5);
% plot(time_data, B(:,5));
% figure(6);
% plot(time_data, B(:,6));

% corrplot(B);

% disp(cov);

% fprintf("\nReduced order model:\n")
% disp(A_rom);
% 
% fprintf("\nOriginal position data:\n")
% disp(posn_data);