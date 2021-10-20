
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import random
import statistics
from scipy.stats import pearsonr

class MortonRegression():

    def __init__(self, degree=3):
        self.coeffs = [0.1] * (degree + 1)
        self.max_iters = 10000
        self.mse_threshold = 0.01

    def MSE(self, x_array, y_array):
        '''different values of omega will be the diff. coefficients in the array
        assume (n+1) coefficients for a nth-order polynomial

        assume length of "coeffs" array is unknown, but that
        coeffs corresponds to highest order down to lowest order
        this means where n = length of xoeffs array, coeffs has 
        coefficients for x^(n-1) down to x^0'''

        num_values = len(x_array)
        if len(x_array) != len(y_array):
            raise ValueError("x and y arrays aren't the same size")

        #iterate through each value in the array
        mse_sum = 0
        for i in range(len(x_array)):
            x_n = x_array[i]
            y_n = y_array[i]
            y_pred = sum([self.coeffs[j] * x_n**(len(self.coeffs) - j - 1) for j in range(len(self.coeffs))])
            mse_sum += (y_n - y_pred)**2

        return mse_sum / num_values     

    #so the iteration happens here
    def derivatives(self, x_array, y_array, lambda_ = 0, L1 = False, L2 = False):
        '''iterate through all the derivatives in the nth-order list of coeffs
        -different values of omega will be the diff. coefficients in the array
        -assume 6 coefficients for a 5th-order polynomial
        -iterate through each value in the array'''

        num_values = len(x_array)
        if len(x_array) != len(y_array):
            raise ValueError("x and y arrays aren't the same size")

        deriv_array = []
        for n in range(len(self.coeffs)):

            xn_power = len(self.coeffs) - n - 1
            deriv_sum = 0
            for i in range(len(x_array)):
                x_n = x_array[i]
                y_n = y_array[i]
                y_pred = sum([self.coeffs[j] * x_n**(len(self.coeffs) - j - 1) for j in range(len(self.coeffs))])
                deriv_sum += x_n**xn_power * (y_n - y_pred)

            nth_deriv = deriv_sum * (-2 / num_values)

            #implement L1 or L2 regularization
            if L1:
                if self.coeffs[n] >= 0:
                    nth_deriv += lambda_
                else:
                    nth_deriv -= lambda_

            elif L2:
                temp = lambda_ * self.coeffs[n] * (sum([x**2 for x in self.coeffs]) ** -0.5)
                nth_deriv += temp

            deriv_array.append(nth_deriv)

        return deriv_array

    def train_model(self, x_array, y_array, alpha = 0.0003, 
                    lambda_ = 0, L1= False, L2 = False):

        iter = 0
        error = 1000

        #iterate through each time and improve model
        while iter < self.max_iters and error > self.mse_threshold:
       
            #make our first guess as to the model. w0*x_n^5 + w1*x_n^4 + ... + w5*x_n^0, e.g.
            y_pred = [sum([self.coeffs[j] * x_n**(len(self.coeffs) - j - 1) \
                for j in range(len(self.coeffs))]) for x_n in x_array]
            r = pearsonr(y_array, y_pred)[0]
            r_sq = r**2

            #find error of the current approximation
            error = self.MSE(x_array, y_array)
            derivs = self.derivatives(x_array, y_array, lambda_, L1, L2) 

            #update coefficients based on gradient descent
            self.coeffs = [self.coeffs[ii] - alpha * derivs[ii] for ii in range(len(self.coeffs))]

            if iter % 1000 == 0:
                print('\nCoeffs: ', self.coeffs)
                print('Error: ', error)
                print('R_squared: ', r_sq)
                print('Iteration: ', iter)
            iter += 1
            #time.sleep(0.5)

        #return coeffs

#main function; otherwise we just import the functions
#from this module
if __name__ == '__main__':

    #parameters for array
    num_values = 50
    x_scale_factor = 4
    x_offset = -1.5
    noise_stdev = 0.5

    #start with a vector of x
    x_array = x_scale_factor * np.random.rand(1, num_values)[0] + x_offset
    x_array.sort()

    #make y_array with actual values of first, second weight
    #y_array = 1.8*x_array**5 - 4.2*x_array**4 + 1.6*x_array**3 + x_array**2 \
    #        -5.2*x_array + 2.1
    y_array = 0.21*x_array**5 - 2.2*x_array**2 + 0.87

    #add some gaussian noise to the dataset
    noise = np.random.normal(0, noise_stdev, num_values)
    y_plus_noise = y_array + noise

    #----------------------------------------#

    #set up learning algorithm for regression.
    reg = MortonRegression(degree=5)
    alpha = 0.00003
    lambda_ = 0.8
    reg.train_model(x_array, y_array, L2 = True)
    coeffs = reg.coeffs

    #take our coefficients and show what y would equal
    y_pred = [sum([coeffs[j] * x_n**(len(coeffs) - j - 1) \
            for j in range(len(coeffs))]) for x_n in x_array]

    r = pearsonr(y_array, y_pred)[0]
    r_sq = r**2

    x_plot = np.arange(x_offset, x_offset + x_scale_factor, 0.01)
    y_plot = [sum([coeffs[j] * x_n**(len(coeffs) - j - 1) \
            for j in range(len(coeffs))]) for x_n in x_plot]

    plt.figure(1)
    #plt.scatter(x_array, y_array)
    plt.scatter(x_array, y_plus_noise)
    plt.plot(x_plot, y_plot,
             label='R_squared: ' + str(round(r_sq, 2)))
    plt.title('Sample nonlinear regression results')
    plt.legend()
    plt.show()
