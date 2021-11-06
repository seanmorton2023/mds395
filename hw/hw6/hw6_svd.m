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
X = multidata(:, 2:7);
time_data = multidata(:, 1);

%find the means of columns so as to have the mean-adjusted data
means_array = mean(X, 1);
B = X - means_array;

% 
% % disp(mean(B));
% 
% %perform svd on data and see what the output of the diagonal matrix is
% [U, S, Vt] = svd(B);
% 
% 
% %now that we have the singular values of the array we know the first 3
% %variables are most important, and really only the 1st one based on values
% 
% %simplify the data by removing a certain number of variables. n is #
% %of variables remaining
% n = 1;
% 
% U_new = U(:, 1:n);
% S_new = S(1:n, 1:n);
% Vt_new = Vt(1:n, :);
% 
% A_rom = U_new * S_new * Vt_new;
% 
% 
% %find the covariance matrix between each vector to find connections
% n_samples = length(X);
% 
% cov1 = 1/(n_samples-1) * (B'*B);
% 
% %try this out: factor in that B*V should have approx. orthogonal
% %components, cov(B*V) should be diagonal; if cov(B*V) has only 1 large
% %value then B*V_1 should be a reduced order model
% A_new = B*Vt';
% 
% V = Vt';
% g1 = V(:,1);
% 
% data2 = B*g1;




%retry: decompose using SVD
%find the covariance matrix of B, then decompose that using SVD
Cb = cov(B);
% disp(Cb);

%Cb is a square matrix so we can decompose using either SVD or PCA
%PCA gives similar results to SVDbut the order of vectors is different
[U, S, Vt] = svd(Cb);
%[eigenvectors, eigenvalues] = eig(Cb);
disp(S);
% disp(Vt);

%U and Vt are the same matrix for an input of the square covariance matrix

%need to rotate B by multiplying it by rotation matrix V.
%covariance matrix C_bv should be diagonal
B_rot = B*U;
disp(cov(B_rot));
%covariance matrix of new rotated matrix is the same values as Sigma array
%with some rounding errors


%else, we'll try PCA












% disp(data2); %later, figure out: is this the reduced order 1D data?
% 
% disp(cov(data2));

% figure(3);
% plot(time_data, data2)



% A_v3 = B*Vt(:,1);
% disp(A_v3);
% disp(cov(A_v3));

% figure(1);
% plot(time_data, X(:,1));
% figure(2);
% plot(time_data, A_new(:,1));

%disp(S(1:6,:));

%%NAH, don't do this approach below. you can't just delete dimensions to
%%get a matrix that's _approximately_ diagonal. We need to use SVD to
%%rotate the original vectors in 6 dimensions to a space in 1 or 2D that
%%has totally independent vectors: cov(matrix) is exactly diagonal, such
%%that the vectors have covariances of 0 between each other and they're
%%totally independent

% %reduce the number of dimensions to get closer to a diagonal covariance matrix
% B_new = B_ma(:, [2 3 6]);
% disp(B_new);
% 
% cov2 = 1/(n_samples - 1) * (B_new' * B_new);
% disp(cov2);

%display the first value in the A matrix to see if our singular values show
%that we can represent the SMD system with just first column
% fprintf("\nOriginal data:\n");
% disp(B);
% 
% fprintf("\nFirst column vector of A:\n");
% disp(A_rom(:,1));


% figure(1);
% A_vec1 = A_rom(:,1);
% plot(time_data, A_vec1);
% 
% figure(2);
% A_vec2 = B(:,1);
% plot(time_data, A_vec2);
% 
% figure(3);
% A_vec3 = A_vec2-A_vec1;
% plot(time_data, A_vec3);


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

% fprintf("\nReduced order model:\n")
% disp(A_rom);
% 
% fprintf("\nOriginal position data:\n")
% disp(posn_data);
