#typical scipy stuff
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import random

#sklearn module is helpful for machine learning
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


num_values = 40
scale_factor = 10
offset = -5
noise_stdev = 2

#make a random array of x values
x_array = scale_factor * np.random.rand(1, num_values)[0] + offset
x_array.sort()
y_array = [1/4*math.exp(x) - 1/8*math.sin(x) - math.cos(x) + 0.1*x**3 for x in x_array ]

#make a gaussian noise array and add it to y_array. let mean, stdev of 
#gaussian distribution be the first inputs
noise = np.random.normal(0,noise_stdev,num_values)
y_plus_noise = y_array + noise

x_ordered = np.arange(offset, offset + scale_factor, 0.2)
y_function = [1/4*math.exp(x) - 1/8*math.sin(x) - math.cos(x) + 0.1*x**3 for x in x_ordered ]


#---------------------------------------------------#

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

plt.scatter(x_array, y_plus_noise)
#plt.plot(x_ordered, y_function, color='black')
plt.plot(x_array, y_predicted, c='black')
plt.title('Placeholder')
plt.show()