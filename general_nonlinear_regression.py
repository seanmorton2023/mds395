
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import random
import statistics
from scipy.stats import pearsonr

num_values = 50
x_scale_factor = 4
x_offset = -1.5
noise_stdev = 0.8

#---------------------------------------------------------#

#set up learning algorithm for regression.
lambda_ = 0.0003
coeffs = [1, 1, 1, 1, 1, 1]
max_iters = 10000
iter = 0
mse_threshold = 0.01
error = 1000

def MSE(coeffs, x_array, y_array):
    '''different values of omega will be the diff. coefficients in the array
    assume (n+1) coefficients for a nth-order polynomial

    assume length of "coeffs" array is unknown, but that
    coeffs corresponds to highest order down to lowest order
    this means where n = length of xoeffs array, coeffs has 
    coefficients for x^(n-1) down to x^0'''

    num_values = len(x_array)
    if len(x_array) != len(y_array):
        raise ValueError("x and y arrays aren't the same size, 3head")

    #iterate through each value in the array
    mse_sum = 0
    for i in range(len(x_array)):
        x_n = x_array[i]
        y_n = y_array[i]
        y_pred = sum([coeffs[j] * x_n**(len(coeffs) - j - 1) for j in range(len(coeffs))])
        mse_sum += (y_n - y_pred)**2

    return mse_sum / num_values

def r_squared(x_array, y_array, y_stars):
    num_values = len(x_array)
    if len(x_array) != len(y_array):
        raise ValueError("x and y arrays aren't the same size, 3head")

    #print(y_array)
    #print(type(y_array))

    #find mean of the y_values array
    y_mean = statistics.mean(y_array)

    #iterate through each value in the array
    RSS = 0
    REGSS = 0
    TSS = 0
    for i in range(len(x_array)):
        y_n = y_array[i]
        y_star = y_stars[i]
        #RSS += (y_n - y_star)**2
        REGSS += (y_star - y_mean)**2
        TSS += (y_n - y_mean)**2

    #return 1 - RSS/TSS
    return REGSS/TSS
        


#make our functions for partial derivatives
#the partial derivative is proportional to the sum of all values,
#so the iteration happens here

#so the iteration happens here
def derivatives(coeffs, x_array, y_array):
    #iterate through all the derivatives in the nth-order list of coeffs
    #different values of omega will be the diff. coefficients in the array
    #assume 6 coefficients for a 5th-order polynomial
    #iterate through each value in the array
    num_values = len(x_array)
    if len(x_array) != len(y_array):
        raise ValueError("x and y arrays aren't the same size, 3head")

    deriv_array = []
    for n in range(len(coeffs)):

        xn_power = len(coeffs) - n - 1
        deriv_sum = 0
        for i in range(len(x_array)):
            x_n = x_array[i]
            y_n = y_array[i]
            y_pred = sum([coeffs[j] * x_n**(len(coeffs) - j - 1) for j in range(len(coeffs))])
            deriv_sum += x_n**xn_power * (y_n - y_pred)

        nth_deriv = deriv_sum * (-2 / num_values)
        deriv_array.append(nth_deriv)

    return deriv_array

#main function; otherwise we just import the functions
#from this module
if __name__ == '__main__':

    #start with a vector of x
    x_array = x_scale_factor * np.random.rand(1, num_values)[0] + x_offset
    x_array.sort()

    #make y_array with actual values of first, second weight
    y_array = 1.8*x_array**5 - 4.2*x_array**4 + 1.6*x_array**3 + x_array**2 \
            -5.2*x_array + 2.1

    #add some gaussian noise to the dataset
    noise = np.random.normal(0, noise_stdev, num_values)
    y_plus_noise = y_array + noise

    print(x_array)
    print(y_array)

    #iterate through each time and improve model
    while iter < max_iters and error > mse_threshold:
       
        #make our first guess as to the model. w0*x_n^5 + w1*x_n^4 + ... + w5*x_n^0, e.g.
        y_pred = [sum([coeffs[j] * x_n**(len(coeffs) - j - 1) \
            for j in range(len(coeffs))]) for x_n in x_array]
        r = pearsonr(y_array, y_pred)[0]
        r_sq = r**2

        #find error of the current approximation
        error = MSE(coeffs, x_array, y_array)
        derivs = derivatives(coeffs, x_array, y_array)

        #update coefficients based on gradient descent
        coeffs = [coeffs[ii] - lambda_ * derivs[ii] for ii in range(len(coeffs))]

        if iter % 200 == 0:
            print('\nCoeffs: ', coeffs)
            print('Error: ', error)
            print('R_squared: ', r_sq)
            print('Iteration: ', iter)
        iter += 1
        #time.sleep(0.5)
    

    #plt.scatter(x_array, y_array)
    plt.scatter(x_array, y_plus_noise)
    plt.plot(x_array, y_pred)
    plt.show()
