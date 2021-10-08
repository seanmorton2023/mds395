import numpy as np
import matplotlib.pyplot as plt
import math
import time
import random
import statistics
from scipy.stats import pearsonr

num_values = 50
x_scale_factor = 10
x_offset = -5
noise_stdev = 2

#---------------------------------------------------------#

#set up learning algorithm for regression.
lambda_ = 0.01
coeffs = [1, 1]
max_iters = 500
iter = 0
mse_threshold = 0.01
error = 1000

def MSE(omega_0, omega_1, x_array, y_array):
    num_values = len(x_array)
    if len(x_array) != len(y_array):
        raise ValueError("x and y arrays aren't the same size, 3head")

    #iterate through each value in the array
    sum = 0
    for i in range(len(x_array)):
        x_n = x_array[i]
        y_n = y_array[i]
        sum += (y_n - (omega_0 * x_n + omega_1))**2

    return sum / num_values

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
        


#make our functions for both partial derivatives
#the partial derivative is proportional to the sum of all values,
#so the iteration happens here
def dE_d_omega_0(omega_0, omega_1, x_array, y_array):
    num_values = len(x_array)
    if len(x_array) != len(y_array):
        raise ValueError("x and y arrays aren't the same size")

    #iterate through each value in the array
    sum = 0
    for i in range(len(x_array)):
        x_n = x_array[i]
        y_n = y_array[i]
        sum += x_n * (y_n - (omega_0 * x_n + omega_1))

    return sum * (-2 / num_values)

def dE_d_omega_1(omega_0, omega_1, x_array, y_array):
    num_values = len(x_array)
    if len(x_array) != len(y_array):
        raise ValueError("x and y arrays aren't the same size")

    #iterate through each value in the array
    sum = 0
    for i in range(len(x_array)):
        x_n = x_array[i]
        y_n = y_array[i]
        sum += (y_n - (omega_0 * x_n + omega_1))

    return sum * (-2 / num_values)


#main function; otherwise we just import the functions
#from this module
if __name__ == '__main__':

    #start with a vector of x
    x_array = x_scale_factor * np.random.rand(1, num_values)[0] + x_offset
    x_array.sort()

    #make y_array with actual values of first, second weight
    y_array = 9.8*x_array + 32.2

    #add some gaussian noise to the dataset
    noise = np.random.normal(0, noise_stdev, num_values)
    y_plus_noise = y_array + noise

    print(x_array)
    print(y_array)



    #test out our rsquared function
    #r_sq1 = r_squared(x_array, y_array, y_pred)
    #r_sq2 = (pearsonr(y_array, y_pred))**2
    #print('\nR squared v1 and v2: ', r_sq1, r_sq2)

    #iterate through each time and improve model
    while iter < max_iters and error > mse_threshold:
       
        #make our first guess as to the model
        y_pred = coeffs[0]*x_array + coeffs[1]
        r_sq1 = r_squared(x_array, y_array, y_pred)

        #find error of the current approximation
        error = MSE(coeffs[0], coeffs[1], x_array, y_array)
        deriv0 = dE_d_omega_0(coeffs[0], coeffs[1], x_array, y_array)
        deriv1 = dE_d_omega_1(coeffs[0], coeffs[1], x_array, y_array)

        print('\nCoeffs: ', coeffs)
        print('Error: ', error)
        print('R_squared: ', r_sq1)

        #update omega 0 and 1 based on gradient descent
        coeffs[0] = coeffs[0] - lambda_*deriv0
        coeffs[1] = coeffs[1] - lambda_*deriv1

        iter += 1
        print('Iteration: ', iter)
        #time.sleep(0.5)
    

    #plt.scatter(x_array, y_array)
    plt.scatter(x_array, y_plus_noise)
    plt.plot(x_array, y_pred)
    plt.show()
