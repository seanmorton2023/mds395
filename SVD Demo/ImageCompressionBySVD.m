clc; clear all; format compact;

k = 50; % rank of SVD approximation

image = 4; % choose a number 1 to 4 identifying a file name
names = ["dice","cat","piano","street"];
A = imread(names(image)+".jpg");
% A = imread('dice.jpg');

% Display the Orginal Image
A = rgb2gray(A);
figure(1)
imshow(A)
title(sprintf('Original (%d by %d)',size(A)))

% SVD Factorization and Truncation
[U1, S1, V1] = svd(double(A));
U = U1(:, 1:k);
V = V1(:, 1:k);
S = S1(1:k, 1:k);

% Display the Approximated Image
A_new = uint8(U*S*V');
figure(2)
imshow(uint8(A_new))
title(sprintf('Rank %d Approximation',size(S,1)))

% Singular Value Plot
i = 1:min(size(S1));
figure (3)
plot(max(S1(:,i)), 'LineWidth', 2)
title('Singular Values in Descending Order')
