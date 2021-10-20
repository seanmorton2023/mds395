#typical scipy stuff
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import random
from scipy.stats import pearsonr

from general_nonlinear_regression import MortonRegression

#sklearn module is helpful for machine learning
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge


num_values = 40
scale_factor = 10
offset = -5
noise_stdev = 1.2

#make a random array of x values
x_array = scale_factor * np.random.rand(1, num_values)[0] + offset
x_array.sort()
y_array = [1/4*math.exp(x) - 1/8*math.sin(x) - math.cos(x) + 0.1*x**3 for x in x_array ]

#make a gaussian noise array and add it to y_array. let mean, stdev of 
#gaussian distribution be the first inputs
noise = np.random.normal(0,noise_stdev,num_values)
y_plus_noise = y_array + noise

#---------------------------------------------------#
#part 1: pretrained model w/ no L1 norm regulatization

#preprocessing on polynomial regression sets up all the
#x variables to n degrees: 1, x, x^2, ..., x^n
pre_process = PolynomialFeatures(degree=3)

#transforms input to a vector that factors in 1, x, x^2, etc.
x_poly = pre_process.fit_transform(x_array.reshape(-1, 1) )

#fir the data to a model. this uses the LinearRegression()
#function but obv. it works with polynomials too
pr_model = LinearRegression()
pr_model.fit(x_poly, y_plus_noise)

#put predicted values of y created by model into an array
y_predicted = pr_model.predict(x_poly)

r_a = pearsonr(y_array, y_predicted)[0]
r_sq_sklinear = r_a**2

plt.figure(1)
plt.scatter(x_array, y_plus_noise)
plt.plot(x_array, y_predicted, c='black',
         label='R_squared: ' + str(round(r_sq_sklinear, 4)))
plt.title('SKLearn LinearRegression model')
plt.legend()
#---------------------------------------------------------#
#part 2: pretrained model with L1 norm regularization

lasso_model = Lasso()
lasso_model.fit(x_poly, y_plus_noise)

#put predicted values of y created by model into an array
y_lasso = lasso_model.predict(x_poly)
r_a1 = pearsonr(y_array, y_lasso)[0]
r_sq_sklasso = r_a1**2


plt.figure(2)
plt.scatter(x_array, y_plus_noise)
plt.plot(x_array, y_lasso, c='black',
         label='R_squared: ' + str(round(r_sq_sklasso, 4)))
plt.title('SKLearn Lasso Model')
plt.legend()


#------------------------------------------------------#
#part 3: pretrained model with L2 norm regularization

ridge_model = Ridge()
ridge_model.fit(x_poly, y_plus_noise)

#put predicted values of y created by model into an array
y_ridge = ridge_model.predict(x_poly)
r_b = pearsonr(y_array, y_ridge)[0]
r_sq_skridge = r_b**2


plt.figure(3)
plt.scatter(x_array, y_plus_noise)
plt.plot(x_array, y_ridge, c='black',
         label='R_squared: ' + str(round(r_sq_skridge, 4)))
plt.title('SKLearn Ridge Model')
plt.legend()


#--------------------------------------------------------#
#part 4: the nonlinear regression model I built for practice

morton_model = MortonRegression(degree=3)
morton_model.train_model(x_array, y_plus_noise, lambda_ = 0.8)
coeffs = morton_model.coeffs

#take our coefficients and show what y would equal
y_morton = [sum([coeffs[j] * x_n**(len(coeffs) - j - 1) \
        for j in range(len(coeffs))]) for x_n in x_array]

r_c = pearsonr(y_array, y_morton)[0]
r_sq_morton = r_c**2

plt.figure(4)
plt.scatter(x_array, y_plus_noise)
plt.plot(x_array, y_array, c='gray')
plt.plot(x_array, y_morton, c='black',
         label='R_squared: ' + str(round(r_sq_morton, 4)))
plt.title('Morton Model, No Regularization')
plt.legend()

#plt.show()

#---------------------------------------------------------------#
#part 5: my nonlinear regression model with L1 reg.

morton_lasso = MortonRegression(degree=3)
morton_lasso.train_model(x_array, y_plus_noise, lambda_ = 1.8, L1 = True)
coeffs = morton_lasso.coeffs

#take our coefficients and show what y would equal
y_L1 = [sum([coeffs[j] * x_n**(len(coeffs) - j - 1) \
        for j in range(len(coeffs))]) for x_n in x_array]

r_d = pearsonr(y_array, y_L1)[0]
r_sq_L1 = r_d**2

plt.figure(5)
plt.scatter(x_array, y_plus_noise)
plt.plot(x_array, y_array, c='gray')
plt.plot(x_array, y_L1, c='black',
         label='R_squared: ' + str(round(r_sq_L1, 4)))
plt.title('Morton Model, L1 Regularization')
plt.legend()

#plt.show()

#----------------------------------------------------------#

#part 6: my nonlinear regression model with L2 reg.

morton_ridge = MortonRegression(degree=3)
morton_ridge.train_model(x_array, y_plus_noise, lambda_ = 1.8, L2 = True)
coeffs = morton_ridge.coeffs

#take our coefficients and show what y would equal
y_L2 = [sum([coeffs[j] * x_n**(len(coeffs) - j - 1) \
        for j in range(len(coeffs))]) for x_n in x_array]

r_e = pearsonr(y_array, y_L2)[0]
r_sq_L2 = r_e**2

plt.figure(6)
plt.scatter(x_array, y_plus_noise)
plt.plot(x_array, y_array, c='gray')
plt.plot(x_array, y_L2, c='black',
         label='R_squared: ' + str(round(r_sq_L2, 4)))
plt.title('Morton Model, L2 Regularization')
plt.legend()

plt.show()

